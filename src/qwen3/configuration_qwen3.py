from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as HFQwen3Config


class Qwen3Config(HFQwen3Config):
    model_type = "qwen3"

    def __init__(
        self,
        rope_parameters=None,
        layer_types=None,
        beacon_window=1024,
        beacon_stride=1024,
        beacon_attn="full-coverage",
        beacon_ratio=None,
        beacon_ratio_mix="step-random",
        beacon_param=None,
        beacon_embed_init="eos",
        beacon_sink_size=0,
        beacon_attend_prev=True,
        beacon_pos="interleave",
        beacon_parallel_window=1,
        **kwargs,
    ):
        if rope_parameters is not None:
            if "rope_theta" in rope_parameters and "rope_theta" not in kwargs:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
            if "rope_scaling" in rope_parameters and "rope_scaling" not in kwargs:
                kwargs["rope_scaling"] = rope_parameters["rope_scaling"]

        if layer_types is not None and "layer_types" not in kwargs:
            kwargs["layer_types"] = layer_types

        super().__init__(**kwargs)

        self.rope_parameters = rope_parameters
        self.beacon_window = beacon_window
        self.beacon_stride = beacon_stride
        self.beacon_attn = beacon_attn
        self.beacon_ratio = [2, 4, 8, 16, 32] if beacon_ratio is None else beacon_ratio
        self.beacon_ratio_mix = beacon_ratio_mix
        self.beacon_param = ["q", "k", "v"] if beacon_param is None else beacon_param
        self.beacon_embed_init = beacon_embed_init
        self.beacon_sink_size = beacon_sink_size
        self.beacon_attend_prev = beacon_attend_prev
        self.beacon_pos = beacon_pos
        self.beacon_parallel_window = beacon_parallel_window


__all__ = ["Qwen3Config"]
