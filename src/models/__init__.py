# from transformers import AutoProcessor
import torch
from .base import BaseVLMAdapter

# Build MODEL_REGISTRY dynamically, only including adapters that successfully import
MODEL_REGISTRY = {}

try:
    from .gemma3 import Gemma3Adapter
    MODEL_REGISTRY["gemma3"] = Gemma3Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .qwen2 import Qwen2Adapter
    MODEL_REGISTRY["qwen2vl"] = Qwen2Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .qwen2_5 import Qwen2_5Adapter
    MODEL_REGISTRY["qwen2_5vl"] = Qwen2_5Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .medvlm import MedVLM_Adapter
    MODEL_REGISTRY["medvlm"] = MedVLM_Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .lingshu import Lingshu_Adapter
    MODEL_REGISTRY["lingshu"] = Lingshu_Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .qwen3 import Qwen3Adapter
    MODEL_REGISTRY["qwen3vl"] = Qwen3Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .internvl3_5 import InternVL35Adapter
    MODEL_REGISTRY["intern"] = InternVL35Adapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .llava import LlavaAdapter
    MODEL_REGISTRY["llava"] = LlavaAdapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .octomed import OctoMedAdapter
    MODEL_REGISTRY["octomed"] = OctoMedAdapter
except (ImportError, SyntaxError, AttributeError):
    pass

try:
    from .llava_med import LlavaMedAdapter
    MODEL_REGISTRY["llavamed"] = LlavaMedAdapter
except (ImportError, SyntaxError, AttributeError):
    pass

def load_model_adapter(model_type, model_name, device, cache_dir):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](model_name, device, cache_dir)