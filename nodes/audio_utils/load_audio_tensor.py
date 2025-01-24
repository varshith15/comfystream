import numpy as np

from comfystream import tensor_cache

class LoadAudioTensor:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"

    def __init__(self):
        self.audio_buffer = np.array([])

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "buffer_size": ("FLOAT", {"default": 500.0})
            }
        }

    @classmethod
    def IS_CHANGED():
        return float("nan")

    def execute(self, buffer_size):
        audio = tensor_cache.audio_inputs.get(block=True)
        self.audio_buffer = np.concatenate((self.audio_buffer, audio))
        
        buffer_size_samples = int(buffer_size * 48)

        if self.audio_buffer.size >= buffer_size_samples:
            buffered_audio = self.audio_buffer[:buffer_size_samples]
            self.audio_buffer = self.audio_buffer[buffer_size_samples:]
            return (buffered_audio,)
        else:
            return (None,)