from ..qwen2.configuration_qwen2 import Qwen2Config


class Qwen3Config(Qwen2Config):
    model_type = "qwen3"

    def __init__(self, rope_parameters=None, layer_types=None, **kwargs):
        # Qwen3 configs in newer transformers may provide rope_parameters/layer_types.
        if rope_parameters is not None:
            if "rope_theta" in rope_parameters and "rope_theta" not in kwargs:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
            if "rope_scaling" in rope_parameters and "rope_scaling" not in kwargs:
                kwargs["rope_scaling"] = rope_parameters["rope_scaling"]
        super().__init__(**kwargs)
        self.layer_types = layer_types
        self.rope_parameters = rope_parameters


__all__ = ["Qwen3Config"]
