import librosa
import numpy as np
from .whisper_online import FasterWhisperASR, VACOnlineASRProcessor

class ApplyWhisper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    CATEGORY = "whisper_utils"
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "apply_whisper"

    def __init__(self):
        self.asr = FasterWhisperASR(
            lan="en", 
            modelsize="large-v3", 
            cache_dir=None, 
            model_dir=None, 
            logfile=None
        )
        self.asr.use_vad()
        
        self.online = VACOnlineASRProcessor(
            online_chunk_size=0.5,
            asr=self.asr,
            tokenizer=None,
            logfile=None,
            buffer_trimming=("segment", 15)
        )
        
    def apply_whisper(self, audio):
        audio = librosa.resample(audio.astype(np.float32), 48000, 16000)
        self.online.insert_audio_chunk(audio)
        text = self.online.process_iter()
        return (text,)
