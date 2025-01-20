import torch
import av
import numpy as np
import fractions

from av import AudioFrame
from typing import Any, Dict, Optional, Union, List
from comfystream.client import ComfyStreamClient

WARMUP_RUNS = 5
import logging
display_logger = logging.getLogger('display_logger')
display_logger.setLevel(logging.INFO)
handler = logging.FileHandler('display_logs.txt')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
display_logger.addHandler(handler)


class VideoPipeline:
    def __init__(self, **kwargs):
        self.client = ComfyStreamClient(**kwargs, type="image")

    async def warm(self):
        frame = torch.randn(1, 512, 512, 3)

        for _ in range(WARMUP_RUNS):
            await self.predict(frame)

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    def preprocess(self, frame: av.VideoFrame) -> torch.Tensor:
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return torch.from_numpy(frame_np).unsqueeze(0)

    async def predict(self, frame: torch.Tensor) -> torch.Tensor:
        return await self.client.queue_prompt(frame)

    def postprocess(self, frame: torch.Tensor) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(
            (frame * 255.0).clamp(0, 255).to(dtype=torch.uint8).squeeze(0).cpu().numpy()
        )

    async def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        pre_output = self.preprocess(frame)
        pred_output = await self.predict(pre_output)
        post_output = self.postprocess(pred_output)

        post_output.pts = frame.pts
        post_output.time_base = frame.time_base

        return post_output


class AudioPipeline:
    def __init__(self, **kwargs):
        self.client = ComfyStreamClient(**kwargs, type="audio")
        self.resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=48000)
        self.sample_rate = 48000
        self.frame_size = int(self.sample_rate * 0.02)
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.curr_pts = 0

    async def warm(self):
        dummy_audio = np.random.randint(-32768, 32767, 48000 * 1, dtype=np.int16)
        for _ in range(WARMUP_RUNS):
            await self.predict(dummy_audio)

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    def preprocess(self, frames: List[av.AudioFrame]) -> torch.Tensor:
        audio_arrays = []
        for frame in frames:
            audio_arrays.append(self.resampler.resample(frame)[0].to_ndarray())
        return np.concatenate(audio_arrays, axis=1).flatten()

    def postprocess(self, out_np) -> Optional[Union[av.AudioFrame, str]]:
        frames = []
        for idx in range(0, len(out_np), self.frame_size):
            frame_samples = out_np[idx:idx + self.frame_size]
            frame_samples = frame_samples.reshape(1, -1)
            frame = AudioFrame.from_ndarray(frame_samples, layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = self.curr_pts
            frame.time_base = self.time_base
            self.curr_pts += 960

            frames.append(frame)
        return frames
        
    async def predict(self, frame) -> torch.Tensor:
        return await self.client.queue_prompt(frame)

    async def __call__(self, frames: List[av.AudioFrame]):
        pre_audio = self.preprocess(frames)
        pred_audio, text = await self.predict(pre_audio)
        if text[-1] != "":
            display_logger.info(f"{text[0]} {text[1]} {text[2]}")
        pred_audios = self.postprocess(pred_audio)
        return pred_audios