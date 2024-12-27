import torch
import av
import numpy as np

from typing import Any, Dict, Optional, Union
from comfystream.client import ComfyStreamClient

WARMUP_RUNS = 5


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

    async def warm(self):
        dummy_audio = torch.randn(16000)
        for _ in range(WARMUP_RUNS):
            await self.predict(dummy_audio)

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    def preprocess(self, frame: av.AudioFrame) -> torch.Tensor:
        self.sample_rate = frame.sample_rate
        samples = frame.to_ndarray(format="s16", layout="stereo")
        samples = samples.astype(np.float32) / 32768.0
        return torch.from_numpy(samples)

    def postprocess(self, output: torch.Tensor) -> Optional[Union[av.AudioFrame, str]]:
        out_np = output.cpu().numpy()
        out_np = np.clip(out_np * 32768.0, -32768, 32767).astype(np.int16)
        audio_frame = av.AudioFrame.from_ndarray(out_np, format="s16", layout="stereo")
        return audio_frame
        
    async def predict(self, frame: torch.Tensor) -> torch.Tensor:
        return await self.client.queue_prompt(frame)

    async def __call__(self, frame: av.AudioFrame):
        pre_output = self.preprocess(frame)
        pred_output = await self.predict(pre_output)
        post_output = self.postprocess(pred_output)
        post_output.sample_rate = self.sample_rate
        post_output.pts = frame.pts
        post_output.time_base = frame.time_base
        return post_output
