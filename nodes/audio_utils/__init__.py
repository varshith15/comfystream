from .apply_whisper import ApplyWhisper
from .load_audio_tensor import LoadAudioTensor
from .save_asr_response import SaveASRResponse
from .save_audio_tensor import SaveAudioTensor

NODE_CLASS_MAPPINGS = {"LoadAudioTensor": LoadAudioTensor, "SaveASRResponse": SaveASRResponse, "ApplyWhisper": ApplyWhisper, "SaveAudioTensor": SaveAudioTensor}

__all__ = ["NODE_CLASS_MAPPINGS"]
