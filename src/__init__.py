from .utils import FileLogger, DefaultDataCollator, makedirs, split_file_dir_name_ext, clear_dir, get_max_length_in_nested_lists, pad_nested_lists, mask_nested_lists, normalize_text, wrap_text, load_json, save_json, load_pickle, save_pickle, add_eos, remove_eos, format_numel_str
from .chat import apply_chat_template
from .args import ModelArgs
from .data import Data
from .modeling_utils import evaluate_perplexity, evaluate_generation, evaluate_nll, move_to_device, get_shifted_labels

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def get_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):    
    import os
    import glob
    import torch
    import transformers
    from dataclasses import asdict
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PretrainedConfig
    from transformers.utils import logging
    from transformers.integrations import is_deepspeed_zero3_enabled
    from packaging import version

    from .args import ModelArgs

    logger = logging.get_logger(__name__)

    model_args: ModelArgs

    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)
    
    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    access_token = model_args_dict["access_token"]

    # Resolve local paths early to avoid Hugging Face Hub repo_id validation errors.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if isinstance(model_name_or_path, str):
        model_name_or_path = os.path.expanduser(model_name_or_path)
        is_local_hint = model_name_or_path.startswith(".") or "/" in model_name_or_path

        if any(ch in model_name_or_path for ch in ["*", "?", "["]):
            patterns = [model_name_or_path]
            if not os.path.isabs(model_name_or_path):
                patterns.append(os.path.join(repo_root, model_name_or_path))
            matches = []
            for pattern in patterns:
                matches.extend(glob.glob(pattern))
            if len(matches) == 1:
                model_name_or_path = matches[0]
            elif len(matches) > 1:
                # deterministically pick the latest path to keep behavior stable.
                model_name_or_path = sorted(matches)[-1]
            else:
                raise FileNotFoundError(
                    f"No local path matched model_name_or_path pattern: {model_name_or_path}. "
                    f"Checked patterns: {patterns}"
                )

        is_local_model_path = os.path.exists(model_name_or_path)
        if not is_local_model_path and is_local_hint and not os.path.isabs(model_name_or_path):
            project_relative_path = os.path.join(repo_root, model_name_or_path)
            if os.path.exists(project_relative_path):
                model_name_or_path = project_relative_path
                is_local_model_path = True

        if is_local_hint and not is_local_model_path:
            snapshot_hints = []
            for candidate in [model_name_or_path, os.path.join(repo_root, model_name_or_path)]:
                snapshots_dir = os.path.dirname(candidate)
                if os.path.basename(snapshots_dir) == "snapshots" and os.path.isdir(snapshots_dir):
                    snapshot_hints = sorted(glob.glob(os.path.join(snapshots_dir, "*")))
                    break
            hint_text = f" Available snapshots: {snapshot_hints[:5]}" if snapshot_hints else ""
            raise FileNotFoundError(
                f"Local model path does not exist: {model_name_or_path}. "
                f"Please check symlink/absolute path and retry.{hint_text}"
            )

        if is_local_model_path:
            model_name_or_path = os.path.abspath(model_name_or_path)
    else:
        is_local_model_path = False

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

    tokenizer_kwargs = {}
    if model_args_dict["no_use_fast"]:
        tokenizer_kwargs = {"use_fast": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        padding_side=model_args_dict["padding_side"], 
        token=access_token, 
        trust_remote_code=True,
        local_files_only=is_local_model_path,
        **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if return_tokenizer_only:
        return tokenizer

    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}
    
    rope_kwargs = {}
    rope_theta = model_args_dict["rope_theta"]
    if rope_theta is not None:
        rope_kwargs["rope_theta"] = rope_theta
    rope_method = model_args_dict["rope_method"]
    if rope_method is not None:
        rope_factor = model_args_dict["rope_factor"]
        rope_scaling = {
            "type": rope_method,
            "factor": rope_factor
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    attn_kwargs = {}
    attn_impl = model_args_dict["attn_impl"]
    if attn_impl is not None:
        if version.parse(transformers.__version__) <= version.parse("4.36"):
            if attn_impl == "flash_attention_2":
                attn_kwargs["use_flash_attention_2"] = True
        else:
            attn_kwargs["attn_implementation"] = attn_impl

    # from_pretrained_kwargs = {}
    # if attn_impl == "flash_attention_2" and version.parse(transformers.__version__) <= version.parse("4.36"):
    #     from_pretrained_kwargs["use_flash_attention_2"] = True

    beacon_kwargs = {}
    for k, v in model_args_dict.items():
        if k.startswith("beacon") and v is not None:
            beacon_kwargs[k] = v

    # Probe config.json without requiring transformers to recognize model_type.
    # This avoids crashes for newer checkpoints (e.g. qwen3_5) on older transformers.
    probe_config_dict, _ = PretrainedConfig.get_config_dict(
        model_name_or_path,
        cache_dir=cache_dir,
        token=access_token,
        local_files_only=is_local_model_path,
    )
    architectures = probe_config_dict.get("architectures", None) or []
    architecture = architectures[0] if len(architectures) else None
    model_type = probe_config_dict.get("model_type", None)

    extra_kwargs = {}
    if model_args_dict["max_position_embeddings"] is not None:
        extra_kwargs["max_position_embeddings"] = model_args_dict["max_position_embeddings"]
    if architecture == "MistralForCausalLM" and model_args_dict["mistral_sliding_window"] is not None:
        extra_kwargs["sliding_window"] = model_args_dict["mistral_sliding_window"]
    if model_args_dict["load_in_4_bit"]:
        extra_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = None

    if model_args_dict["enable_beacon"]:
        from .llama import LlamaForCausalLM, LlamaConfig
        from .mistral import MistralForCausalLM, MistralConfig
        from .qwen2 import Qwen2ForCausalLM, Qwen2Config
        from .qwen3 import Qwen3ForCausalLM, Qwen3Config
        from .qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        ARCHITECTURE_TO_CLASS = {
            'LlamaForCausalLM': (LlamaConfig, LlamaForCausalLM),
            'MistralForCausalLM': (MistralConfig, MistralForCausalLM),
            'Qwen2ForCausalLM': (Qwen2Config, Qwen2ForCausalLM),
            'Qwen3ForCausalLM': (Qwen3Config, Qwen3ForCausalLM),
            'Qwen3_5ForCausalLM': (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
        }
        MODEL_TYPE_TO_CLASS = {
            "llama": (LlamaConfig, LlamaForCausalLM),
            "mistral": (MistralConfig, MistralForCausalLM),
            "qwen2": (Qwen2Config, Qwen2ForCausalLM),
            "qwen3": (Qwen3Config, Qwen3ForCausalLM),
            "qwen3_5_text": (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
        }

        if architecture in ARCHITECTURE_TO_CLASS:
            config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]
        elif model_type in MODEL_TYPE_TO_CLASS:
            config_class, model_class = MODEL_TYPE_TO_CLASS[model_type]
        else:
            supported = ", ".join(sorted(ARCHITECTURE_TO_CLASS.keys()))
            raise NotImplementedError(
                f"Beacon model for architecture/model_type '{architecture}'/'{model_type}' is not implemented yet. "
                f"Supported architectures: {supported}"
            )

        if config_class is Qwen3_5TextConfig and model_type == "qwen3_5":
            # qwen3.5 checkpoints may store a top-level multimodal config with nested text_config.
            text_config_dict = probe_config_dict.get("text_config", probe_config_dict)
            config = config_class.from_dict(
                text_config_dict,
                torch_dtype=dtype,
                **beacon_kwargs,
                **rope_kwargs,
                **attn_kwargs,
                **extra_kwargs,
            )
        else:
            config = config_class.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                token=access_token,
                # NOTE: keep the torch_dtype in config consistent with that in model
                torch_dtype=dtype,
                local_files_only=is_local_model_path,
                **beacon_kwargs,
                **rope_kwargs,
                **attn_kwargs,
                **extra_kwargs,
            )
        model = model_class.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
            local_files_only=is_local_model_path,
        )

    else:
        if model_args_dict["enable_vllm"]:
            from .vllm_utils import HFStyleVllmModel
            if model_args_dict["dtype"] == "fp32":
                vllm_dtype = "float32"
            elif model_args_dict["dtype"] == "fp16":
                vllm_dtype = "float16"
            elif model_args_dict["dtype"] == "bf16":
                vllm_dtype = "bfloat16"

            vllm_kwargs = {}
            if model_args_dict["vllm_len"] is not None:
                vllm_kwargs["max_model_len"] = model_args_dict["vllm_len"]

            model = HFStyleVllmModel(
                model=model_name_or_path,
                dtype=vllm_dtype,
                gpu_memory_utilization=model_args_dict["vllm_mem"],
                tensor_parallel_size=model_args_dict["vllm_tp"],
                disable_custom_all_reduce=model_args_dict["vllm_disable_ar"],
                enforce_eager=False,
                trust_remote_code=True,
                **rope_kwargs,
                **vllm_kwargs,
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir, 
                torch_dtype=dtype,
                device_map=device_map,
                token=access_token,
                trust_remote_code=True,
                local_files_only=is_local_model_path,

                # NOTE: do not destroy the default rope_scaling of the model
                **rope_kwargs,
                **attn_kwargs,
                **extra_kwargs,
            )

    # load lora
    if model_args_dict["lora"] is not None:
        logger.info(f"loading lora from {model_args_dict['lora']}...")

        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, 
            model_args_dict["lora"],
            torch_dtype=dtype,
            device_map=device_map,
        )
        if model_args_dict["lora_unload"]:
            model = model.merge_and_unload()

    if model_args_dict["enable_tp"]:
        import tensor_parallel as tp
        logger.info("enabling tensor parallelism...")
        
        # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
        model = tp.tensor_parallel(model, sharded=True)

        if model.generation_config.eos_token_id == 128001:
            model.generation_config.eos_token_id = [128001, 128009]

    if isinstance(model, transformers.modeling_utils.PreTrainedModel):
        model = model.eval()
        if evaluation_mode:
            # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
            model.requires_grad_(False)
        logger.info(model.config)

    # override the default generation config
    generation_config = model_args.get_generation_config()
    if len(generation_config):
        model.generation_config.update(**generation_config)
    logger.info(f"Specified generation config: {generation_config}")

    return model, tokenizer
