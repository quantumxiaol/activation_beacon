from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as HFQwen3Attention,
    Qwen3DecoderLayer as HFQwen3DecoderLayer,
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
    Qwen3Model as HFQwen3Model,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ..modeling_beacon import Memory
from ..modeling_utils import ModelOutput, compute_loss, optional_grad_ctx
from .configuration_qwen3 import Qwen3Config


class BeaconQwen3Attention(HFQwen3Attention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        qkv_bias = getattr(config, "attention_bias", self.q_proj.bias is not None)
        o_bias = getattr(config, "attention_bias", self.o_proj.bias is not None)

        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=qkv_bias,
            )
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True

        if "k" in config.beacon_param:
            self.beacon_k_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * self.head_dim,
                bias=qkv_bias,
            )
            self.beacon_k_proj.weight.data.zero_()
            self.beacon_k_proj._is_hf_initialized = True

        if "v" in config.beacon_param:
            self.beacon_v_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * self.head_dim,
                bias=qkv_bias,
            )
            self.beacon_v_proj.weight.data.zero_()
            self.beacon_v_proj._is_hf_initialized = True

        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim,
                config.hidden_size,
                bias=o_bias,
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
                ordinal_query_states = self.q_proj(hidden_states)
                beacon_query_states = self.beacon_q_proj(hidden_states)
                query_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_query_states, beacon_query_states)
                if (cur_beacon_indices == 2).any():
                    query_states[:, cur_beacon_indices == 2] = beacon_query_states[:, cur_beacon_indices == 1][
                        :, : (cur_beacon_indices == 2).sum()
                    ]
            else:
                query_states = self.q_proj(hidden_states)

            if "k" in self.config.beacon_param:
                ordinal_key_states = self.k_proj(hidden_states)
                beacon_key_states = self.beacon_k_proj(hidden_states)
                key_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_key_states, beacon_key_states)
                if (cur_beacon_indices == 2).any():
                    key_states[:, cur_beacon_indices == 2] = beacon_key_states[:, cur_beacon_indices == 1][
                        :, : (cur_beacon_indices == 2).sum()
                    ]
            else:
                key_states = self.k_proj(hidden_states)

            if "v" in self.config.beacon_param:
                ordinal_value_states = self.v_proj(hidden_states)
                beacon_value_states = self.beacon_v_proj(hidden_states)
                value_states = torch.where(
                    (cur_beacon_indices == 0)[:, None], ordinal_value_states, beacon_value_states
                )
                if (cur_beacon_indices == 2).any():
                    value_states[:, cur_beacon_indices == 2] = beacon_value_states[:, cur_beacon_indices == 1][
                        :, : (cur_beacon_indices == 2).sum()
                    ]
            else:
                value_states = self.v_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states

    def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0 and "o" in self.config.beacon_param:
            cur_beacon_indices = beacon_indices[-attn_output.shape[1] :]
            ordinal_attn_output = self.o_proj(attn_output)
            beacon_attn_output = self.beacon_o_proj(attn_output)
            return torch.where((cur_beacon_indices == 0)[:, None], ordinal_attn_output, beacon_attn_output)
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if isinstance(past_key_values, tuple):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            past_key, past_value, beacon_size, beacon_indices = past_key_values
            query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
            value_states = value_states.view(hidden_shape).transpose(1, 2)

            # Apply RoPE BEFORE concatenation — past keys are already RoPE-encoded
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            present_key_value = (key_states, value_states, beacon_size, beacon_indices)

            if past_key is not None:
                key_states = torch.cat([past_key, key_states], dim=2)
                value_states = torch.cat([past_value, value_states], dim=2)

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)
            return attn_output, attn_weights, present_key_value

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )


class BeaconQwen3DecoderLayer(HFQwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = BeaconQwen3Attention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if isinstance(past_key_values, tuple):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _, layer_past = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, layer_past

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class BeaconQwen3Model(HFQwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [BeaconQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size)
        self.beacon_embed_tokens._is_hf_initialized = True
        self.post_init()

    def _init_beacon_embed(self, missing_keys):
        if is_deepspeed_zero3_enabled():
            import deepspeed

            params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if (self.beacon_embed_tokens.weight == 0).all():
                    if self.config.beacon_embed_init == "bos":
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    elif self.config.beacon_embed_init == "eos":
                        eos_token_id = self.config.eos_token_id
                        if isinstance(eos_token_id, list):
                            eos_token_id = eos_token_id[0]
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                    else:
                        raise NotImplementedError(
                            f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}"
                        )
        else:
            if any("beacon_embed_tokens" in missing_key for missing_key in missing_keys):
                if self.config.beacon_embed_init == "bos":
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                elif self.config.beacon_embed_init == "eos":
                    eos_token_id = self.config.eos_token_id
                    if isinstance(eos_token_id, list):
                        eos_token_id = eos_token_id[0]
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                else:
                    raise NotImplementedError(
                        f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}"
                    )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if isinstance(past_key_values, list):
            # Beacon memory update relies on per-layer KV cache outputs.
            use_cache = True
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

            if inputs_embeds is None:
                past_key, _, beacon_size, beacon_indices = past_key_values[0]
                if beacon_size > 0:
                    cur_beacon_indices = beacon_indices[-input_ids.shape[1] :]
                    ordinal_input_ids = input_ids[:, cur_beacon_indices == 0]
                    beacon_input_ids = input_ids[:, cur_beacon_indices > 0]
                    ordinal_inputs_embeds = self.embed_tokens(ordinal_input_ids)
                    beacon_input_embeds = self.beacon_embed_tokens(beacon_input_ids - self.config.vocab_size)
                    inputs_embeds = beacon_input_embeds.new_zeros(*input_ids.shape, beacon_input_embeds.shape[-1])
                    inputs_embeds[:, cur_beacon_indices == 0] = ordinal_inputs_embeds
                    inputs_embeds[:, cur_beacon_indices > 0] = beacon_input_embeds
                else:
                    inputs_embeds = self.embed_tokens(input_ids)

            if position_ids is None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

            hidden_states = inputs_embeds
            # Memory.step() may provide position_ids for [memory + current chunk], while
            # RoPE here should only be applied to current chunk queries/keys.
            rope_position_ids = position_ids[:, -hidden_states.shape[1] :]
            position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
            next_decoder_cache = []

            for layer_idx, decoder_layer in enumerate(self.layers):
                layer_past = past_key_values[layer_idx] if past_key_values is not None else None

                layer_attention_mask = attention_mask
                if isinstance(attention_mask, dict):
                    layer_attention_mask = attention_mask.get(
                        decoder_layer.attention_type, attention_mask.get("full_attention")
                    )

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        layer_attention_mask,
                        position_ids,
                        layer_past,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_values=layer_past,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]
                next_decoder_cache.append(layer_outputs[1] if len(layer_outputs) > 1 else None)

            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache if use_cache else None,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


class Qwen3ForCausalLM(HFQwen3ForCausalLM):
    config_class = Qwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = BeaconQwen3Model(config)
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
        cache_position: Optional[torch.LongTensor] = None,
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
            cache_position=cache_position,
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


__all__ = ["Qwen3ForCausalLM"]
