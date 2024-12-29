from .load_audio_tensor import LoadAudioTensor
from .save_result import SaveResult
from .save_audio_tensor import SaveAudioTensor

NODE_CLASS_MAPPINGS = {"LoadAudioTensor": LoadAudioTensor, "SaveResult": SaveResult, "SaveAudioTensor": SaveAudioTensor}

__all__ = ["NODE_CLASS_MAPPINGS"]
