import logging
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from src import ( 
    Data,
    DefaultDataCollator,
    ModelArgs,
    FileLogger,
    get_model_and_tokenizer,
    makedirs,
    format_numel_str
)
from src.args import TrainingArgs
from src.metrics import Metric
from src.trainer import ActivationBeaconTrainer

logger = logging.getLogger(__name__)


def _patch_deepspeed_grad_fn_compat() -> bool:
    """Patch DeepSpeed/Torch grad-fn probes to avoid NoneType crashes on frozen params."""
    try:
        import torch
        import torch.autograd.graph as torch_graph
        from deepspeed.runtime import utils as ds_runtime_utils
    except Exception:
        return False

    patched = False

    if not getattr(torch_graph, "_activation_beacon_grad_fn_patch", False):
        original_get_grad_fn = torch_graph._get_grad_fn_or_grad_acc

        def _safe_get_grad_fn_or_grad_acc(t):
            if t is None:
                return None
            try:
                return original_get_grad_fn(t)
            except AttributeError:
                if not getattr(t, "requires_grad", False):
                    return None
                # On some torch/deepspeed combinations, t.view_as(t).grad_fn can be None.
                with torch.enable_grad():
                    grad_fn = t.view_as(t).grad_fn
                if grad_fn is None or len(grad_fn.next_functions) == 0:
                    return None
                return grad_fn.next_functions[0][0]

        torch_graph._get_grad_fn_or_grad_acc = _safe_get_grad_fn_or_grad_acc
        torch_graph._activation_beacon_grad_fn_patch = True
        patched = True

    if not getattr(ds_runtime_utils, "_activation_beacon_grad_fn_patch", False):
        # Keep DeepSpeed's reference in sync with patched torch helper.
        ds_runtime_utils._get_grad_fn_or_grad_acc = torch_graph._get_grad_fn_or_grad_acc
        ds_runtime_utils._activation_beacon_grad_fn_patch = True
        patched = True

    if not getattr(ds_runtime_utils, "_activation_beacon_count_patch", False):
        def _safe_count_used_parameters_in_backward(params):
            used = 0
            for param in params:
                try:
                    grad_fn = ds_runtime_utils._get_grad_fn_or_grad_acc(param)
                except Exception:
                    grad_fn = None
                if grad_fn is not None:
                    used += 1
            return used

        ds_runtime_utils.count_used_parameters_in_backward = _safe_count_used_parameters_in_backward
        ds_runtime_utils._activation_beacon_count_patch = True
        patched = True

    return patched or getattr(ds_runtime_utils, "_activation_beacon_grad_fn_patch", False)


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.deepspeed is not None and _patch_deepspeed_grad_fn_compat():
        logger.warning(
            "Applied DeepSpeed grad-fn compatibility patch for current torch/deepspeed versions."
        )

    model, tokenizer = get_model_and_tokenizer(model_args, device="cuda", evaluation_mode=False)

    beacon_param_names = [name for name, _ in model.named_parameters() if "beacon" in name]
    if model_args.enable_beacon:
        logger.info(
            f"Detected beacon parameter tensors: {len(beacon_param_names)} "
            f"(beacon_param={model_args.beacon_param})"
        )

    if model_args.enable_beacon and training_args.only_train_beacon:
        if len(beacon_param_names) == 0:
            raise RuntimeError(
                "No beacon parameter tensors were found, but only_train_beacon=True. "
                f"Current beacon_param={model_args.beacon_param}. "
                "Please check --enable_beacon / --beacon_param."
            )
        for name, param in model.named_parameters():
            param.requires_grad_("beacon" in name)

    if training_args.lora_tune:
        from peft import (
            LoraConfig,
            get_peft_model,
        )
        # copied from LongLoRA
        config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_targets,
            modules_to_save=training_args.lora_extra_params,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_numel = sum(getattr(p, "ds_numel", p.numel()) for p in trainable_params)
    logger.info(
        f"Trainable Model params: {format_numel_str(trainable_numel)} "
        f"({len(trainable_params)} tensors)"
    )
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found. Check --enable_beacon / --only_train_beacon "
            "and whether beacon parameters exist in the loaded model."
        )

    with training_args.main_process_first():
        train_dataset = Data.prepare_train_data(
            model_args.train_data, 
            tokenizer=tokenizer,
            max_length=model_args.max_length,
            min_length=training_args.min_length,
            chat_template=model_args.chat_template,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    with training_args.main_process_first():
        if is_deepspeed_zero3_enabled() and training_args.eval_method != "perplexity":
            logger.warning(f"In deepspeed zero3, evaluation with generation is may lead to hang because of the unequal number of forward passes across different devices.")
        eval_dataset = Data.prepare_eval_data(
            model_args.eval_data, 
            tokenizer=tokenizer,
            max_length=training_args.eval_max_length,
            min_length=training_args.eval_min_length,
            chat_template=model_args.chat_template,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    trainer = ActivationBeaconTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(tokenizer),
        file_logger=FileLogger(makedirs(training_args.log_path)),
        compute_metrics=Metric.get_metric_fn(
            metrics=training_args.metrics,
            save_path=Metric.get_save_path(
                model_args.eval_data,
                training_args.output_dir
            ) if model_args.eval_data is not None else None
        )
    )
    if train_dataset is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    elif eval_dataset is not None:
        trainer.evaluate()

if __name__ == "__main__":
    main()
