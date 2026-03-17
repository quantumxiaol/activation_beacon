from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.integrations import is_deepspeed_zero3_enabled

from ..modeling_beacon import Memory
from ..modeling_utils import ModelOutput, compute_loss, optional_grad_ctx
from .configuration_qwen3_5 import Qwen3_5TextConfig
from .modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class BeaconQwen3_5Attention(Qwen3_5Attention):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
            )
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in config.beacon_param:
            self.beacon_k_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.beacon_k_proj.weight.data.zero_()
            self.beacon_k_proj._is_hf_initialized = True
        if "v" in config.beacon_param:
            self.beacon_v_proj = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.beacon_v_proj.weight.data.zero_()
            self.beacon_v_proj._is_hf_initialized = True
        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
            )
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

    def _init_beacon_proj(self, missing_keys):
        beacon_param = self.config.beacon_param
        if is_deepspeed_zero3_enabled():
            import deepspeed

            if "q" in beacon_param:
                params = [self.beacon_q_proj.weight, self.q_proj.weight]
                if self.q_proj.bias is not None:
                    params.extend([self.beacon_q_proj.bias, self.q_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_q_proj.weight.sum(-1) == 0).any() or (self.beacon_q_proj.weight > 1e29).any():
                        self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                        if self.q_proj.bias is not None:
                            self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data

            if "k" in beacon_param:
                params = [self.beacon_k_proj.weight, self.k_proj.weight]
                if self.k_proj.bias is not None:
                    params.extend([self.beacon_k_proj.bias, self.k_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_k_proj.weight.sum(-1) == 0).any() or (self.beacon_k_proj.weight > 1e29).any():
                        self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                        if self.k_proj.bias is not None:
                            self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data

            if "v" in beacon_param:
                params = [self.beacon_v_proj.weight, self.v_proj.weight]
                if self.v_proj.bias is not None:
                    params.extend([self.beacon_v_proj.bias, self.v_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_v_proj.weight.sum(-1) == 0).any() or (self.beacon_v_proj.weight > 1e29).any():
                        self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                        if self.v_proj.bias is not None:
                            self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data

            if "o" in beacon_param:
                params = [self.beacon_o_proj.weight, self.o_proj.weight]
                if self.o_proj.bias is not None:
                    params.extend([self.beacon_o_proj.bias, self.o_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_o_proj.weight.sum(-1) == 0).any() or (self.beacon_o_proj.weight > 1e29).any():
                        self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                        if self.o_proj.bias is not None:
                            self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data
        else:
            with torch.no_grad():
                if "q" in beacon_param and any("beacon_q_proj" in k for k in missing_keys):
                    self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                    if self.q_proj.bias is not None:
                        self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
                if "k" in beacon_param and any("beacon_k_proj" in k for k in missing_keys):
                    self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                    if self.k_proj.bias is not None:
                        self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
                if "v" in beacon_param and any("beacon_v_proj" in k for k in missing_keys):
                    self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                    if self.v_proj.bias is not None:
                        self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
                if "o" in beacon_param and any("beacon_o_proj" in k for k in missing_keys):
                    self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                    if self.o_proj.bias is not None:
                        self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data

    def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices):
        if beacon_size > 0:
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1] :]

            if "q" in self.config.beacon_param:
                ordinal_q = self.q_proj(hidden_states)
                beacon_q = self.beacon_q_proj(hidden_states)
                q_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_q, beacon_q)
                if (cur_beacon_indices == 2).any():
                    source = beacon_q[:, cur_beacon_indices == 1]
                    if source.shape[1] > 0:
                        q_states[:, cur_beacon_indices == 2] = source[:, : (cur_beacon_indices == 2).sum()]
            else:
                q_states = self.q_proj(hidden_states)

            if "k" in self.config.beacon_param:
                ordinal_k = self.k_proj(hidden_states)
                beacon_k = self.beacon_k_proj(hidden_states)
                k_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_k, beacon_k)
                if (cur_beacon_indices == 2).any():
                    source = beacon_k[:, cur_beacon_indices == 1]
                    if source.shape[1] > 0:
                        k_states[:, cur_beacon_indices == 2] = source[:, : (cur_beacon_indices == 2).sum()]
            else:
                k_states = self.k_proj(hidden_states)

            if "v" in self.config.beacon_param:
                ordinal_v = self.v_proj(hidden_states)
                beacon_v = self.beacon_v_proj(hidden_states)
                v_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_v, beacon_v)
                if (cur_beacon_indices == 2).any():
                    source = beacon_v[:, cur_beacon_indices == 1]
                    if source.shape[1] > 0:
                        v_states[:, cur_beacon_indices == 2] = source[:, : (cur_beacon_indices == 2).sum()]
            else:
                v_states = self.v_proj(hidden_states)
        else:
            q_states = self.q_proj(hidden_states)
            k_states = self.k_proj(hidden_states)
            v_states = self.v_proj(hidden_states)

        return q_states, k_states, v_states

    def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0 and "o" in self.config.beacon_param:
            cur_beacon_indices = beacon_indices[-attn_output.shape[1] :]
            ordinal_out = self.o_proj(attn_output)
            beacon_out = self.beacon_o_proj(attn_output)
            out = torch.where((cur_beacon_indices == 0)[:, None], ordinal_out, beacon_out)
            if (cur_beacon_indices == 2).any():
                source = beacon_out[:, cur_beacon_indices == 1]
                if source.shape[1] > 0:
                    out[:, cur_beacon_indices == 2] = source[:, : (cur_beacon_indices == 2).sum()]
            return out
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        if isinstance(past_key_values, tuple):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            past_key, past_value, beacon_size, beacon_indices = past_key_values

            q_states, k_states, v_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)
            query_states, gate = torch.chunk(q_states.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(k_states.view(hidden_shape)).transpose(1, 2)
            value_states = v_states.view(hidden_shape).transpose(1, 2)

            # Apply RoPE BEFORE concatenation — past keys are already RoPE-encoded
            cos, sin = position_embeddings
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            present_key_value = (key_states, value_states, beacon_size, beacon_indices)

            if past_key is not None:
                key_states = torch.cat([past_key, key_states], dim=2)
                value_states = torch.cat([past_value, value_states], dim=2)

            # Resolve attention implementation with compat fallback
            if hasattr(ALL_ATTENTION_FUNCTIONS, 'get_interface'):
                attention_impl = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward
                )
            elif isinstance(ALL_ATTENTION_FUNCTIONS, dict):
                attention_impl = ALL_ATTENTION_FUNCTIONS.get(
                    self.config._attn_implementation, eager_attention_forward
                )
            else:
                attention_impl = eager_attention_forward

            attn_output, attn_weights = attention_impl(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)
            return attn_output, attn_weights, present_key_value

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    # keep same callable used by use_kernelized_func decorator
    apply_rotary_pos_emb = staticmethod(apply_rotary_pos_emb)


class BeaconQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.layer_type == "full_attention":
            self.self_attn = BeaconQwen3_5Attention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        beacon_mode = isinstance(past_key_values, tuple)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        layer_past = None
        if self.layer_type == "linear_attention":
            # Keep linear attention stateless in beacon mode to avoid cache structure mismatch.
            cache_params = None if beacon_mode else past_key_values
            linear_attn_mask = None if beacon_mode else attention_mask
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=linear_attn_mask,
            )
            if beacon_mode:
                layer_past = past_key_values
        else:
            if beacon_mode:
                hidden_states, _, layer_past = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            else:
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if beacon_mode:
            return hidden_states, layer_past
        return hidden_states


class BeaconQwen3_5TextModel(Qwen3_5TextModel):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [BeaconQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Keep beacon embedding independent from model pad token id to avoid invalid padding_idx
        # when pad_token_id is outside [0, num_embeddings).
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size)
        self.beacon_embed_tokens._is_hf_initialized = True
        self.post_init()

    def _init_beacon_embed(self, missing_keys):
        if is_deepspeed_zero3_enabled():
            import deepspeed

            params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if (self.beacon_embed_tokens.weight == 0).all():
                    if self.config.beacon_embed_init == "bos" and self.config.bos_token_id is not None:
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    else:
                        eos_token_id = self.config.eos_token_id
                        if isinstance(eos_token_id, list):
                            eos_token_id = eos_token_id[0]
                        if eos_token_id is None:
                            eos_token_id = self.config.vocab_size - 1
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
        else:
            with torch.no_grad():
                if any("beacon_embed_tokens" in k for k in missing_keys):
                    if self.config.beacon_embed_init == "bos" and self.config.bos_token_id is not None:
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    else:
                        eos_token_id = self.config.eos_token_id
                        if isinstance(eos_token_id, list):
                            eos_token_id = eos_token_id[0]
                        if eos_token_id is None:
                            eos_token_id = self.config.vocab_size - 1
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        if isinstance(past_key_values, list):
            if inputs_embeds is None:
                past_key, _, beacon_size, beacon_indices = past_key_values[0]
                if beacon_size > 0:
                    cur_beacon_indices = beacon_indices[-input_ids.shape[1] :]
                    ordinal_input_ids = input_ids[:, cur_beacon_indices == 0]
                    beacon_input_ids = input_ids[:, cur_beacon_indices > 0]
                    ordinal_inputs_embeds = self.embed_tokens(ordinal_input_ids)
                    beacon_inputs_embeds = self.beacon_embed_tokens(beacon_input_ids - self.config.vocab_size)
                    inputs_embeds = beacon_inputs_embeds.new_zeros(*input_ids.shape, beacon_inputs_embeds.shape[-1])
                    inputs_embeds[:, cur_beacon_indices == 0] = ordinal_inputs_embeds
                    inputs_embeds[:, cur_beacon_indices > 0] = beacon_inputs_embeds
                else:
                    inputs_embeds = self.embed_tokens(input_ids)

            if position_ids is None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
                position_ids = position_ids.expand(inputs_embeds.shape[0], -1)

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            next_decoder_cache = []

            for layer_idx, decoder_layer in enumerate(self.layers):
                layer_past = past_key_values[layer_idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    # Bypass GradientCheckpointingLayer.__call__ which strips
                    # past_key_values, breaking beacon mode detection.
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        position_ids,
                        layer_past,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=layer_past,
                        use_cache=use_cache,
                        **kwargs,
                    )

                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    next_decoder_cache.append(layer_outputs[1] if len(layer_outputs) > 1 else layer_past)
                else:
                    hidden_states = layer_outputs
                    next_decoder_cache.append(layer_past)

            hidden_states = self.norm(hidden_states)
            return Qwen3_5ModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class Qwen3_5ForCausalLM(HFQwen3_5ForCausalLM):
    config_class = Qwen3_5TextConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = BeaconQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)
        model.memory = Memory(
            model_config=model.config,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        missing_keys = loading_info["missing_keys"]
        model.model._init_beacon_embed(missing_keys)
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn._init_beacon_proj(missing_keys)
        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ModelOutput]:
        if past_key_values is None:
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        loss = None
        batch_loss = None
        token_loss = None
        if labels is not None:
            loss, batch_loss, token_loss = compute_loss(logits, labels, shift=False)

        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            token_loss=token_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def _beacon_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        **kwargs,
    ):
        # Beacon mode requires returned KV cache to update memory each step.
        use_cache = True
        self.memory.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            skip_first=beacon_skip_first,
            skip_last=beacon_skip_last,
        )

        while not self.memory.finish:
            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()
            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                **kwargs,
            )
            if outputs.past_key_values is None:
                raise RuntimeError(
                    "Beacon forward requires `past_key_values`, but got None. "
                    "Ensure `use_cache=True` in beacon mode."
                )
            self.memory.update_memory(outputs.past_key_values)
            if labels is not None:
                self.memory.update_loss(outputs.batch_loss, (labels != -100).sum(-1))

        return self.memory.output(outputs)

    def forward(self, **kwargs):
        with optional_grad_ctx(with_grad=self.training):
            if hasattr(self, "_enable_beacon") and self._enable_beacon is False:
                return self._native_forward(**kwargs)
            return self._beacon_forward(**kwargs)


__all__ = ["Qwen3_5ForCausalLM"]
