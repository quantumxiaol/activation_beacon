from ..qwen2.modeling_qwen2 import Qwen2ForCausalLM
from .configuration_qwen3 import Qwen3Config


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen3Config


__all__ = ["Qwen3ForCausalLM"]
