import torch
import av
import numpy as np
import fractions
import asyncio

from av import AudioFrame
from typing import Any, Dict, Optional, Union, List
from comfystream.client import ComfyStreamClient
from comfystream import tensor_cache

WARMUP_RUNS = 5



class Pipeline:
    def __init__(self, **kwargs):
        self.client = ComfyStreamClient(**kwargs, max_workers=5) # hardcoded max workers

        self.video_futures = asyncio.Queue()
        self.audio_futures = asyncio.Queue()

        self.audio_output_frames = []
        
        self.resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=48000) # find a better way to convert to mono
        self.sample_rate = 48000 # instead of hardcoding, find a clean way to set from audio frame
        self.frame_size = int(self.sample_rate * 0.02)
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.curr_pts = 0 # figure out a better way to set back pts to processed audio frames

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    async def warm(self):
        dummy_video_frame = torch.randn(1, 512, 512, 3)
        dummy_audio_frame = np.random.randint(-32768, 32767, 48000 * 1, dtype=np.int16)

        for _ in range(WARMUP_RUNS):
            image_out_fut = asyncio.Future()
            audio_out_fut = asyncio.Future()
            tensor_cache.image_outputs.put(image_out_fut)
            tensor_cache.audio_outputs.put(audio_out_fut)

            tensor_cache.image_inputs.put(dummy_video_frame)
            tensor_cache.audio_inputs.put(dummy_audio_frame)

            await image_out_fut
            await audio_out_fut

    def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(prompts, dict):
            self.client.set_prompts([prompts])
        else:
            self.client.set_prompts(prompts)

    async def put_video_frame(self, frame: av.VideoFrame):
        inp_tensor = self.video_preprocess(frame)
        out_future = asyncio.Future()
        tensor_cache.image_outputs.put(out_future)
        tensor_cache.image_inputs.put(inp_tensor)
        await self.video_futures.put((out_future, frame.pts, frame.time_base))

    async def put_audio_frame(self, frame: av.AudioFrame):
        inp_tensor = self.audio_preprocess(frame)
        out_future = asyncio.Future()
        tensor_cache.audio_outputs.put(out_future)
        tensor_cache.audio_inputs.put(inp_tensor)
        await self.audio_futures.put(out_future)

    def video_preprocess(self, frame: av.VideoFrame) -> torch.Tensor:
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return torch.from_numpy(frame_np).unsqueeze(0)
    
    def audio_preprocess(self, frame: av.AudioFrame) -> torch.Tensor:
        return self.resampler.resample(frame)[0].to_ndarray().flatten()
    
    def video_postprocess(self, output: torch.Tensor) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(
            (output * 255.0).clamp(0, 255).to(dtype=torch.uint8).squeeze(0).cpu().numpy()
        )

    def audio_postprocess(self, output: torch.Tensor) -> av.AudioFrame:
        frames = []
        for idx in range(0, len(output), self.frame_size):
            frame_samples = output[idx:idx + self.frame_size]
            frame_samples = frame_samples.reshape(1, -1).astype(np.int16)
            frame = AudioFrame.from_ndarray(frame_samples, layout="mono")
            frame.sample_rate = self.sample_rate
            frame.pts = self.curr_pts
            frame.time_base = self.time_base
            self.curr_pts += 960

            frames.append(frame)
        return frames
    
    async def get_processed_video_frame(self):
        out_fut, pts, time_base = await self.video_futures.get()
        frame = self.video_postprocess(await out_fut)
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def get_processed_audio_frame(self):
        while not self.audio_output_frames:
            out_fut = await self.audio_futures.get()
            output = await out_fut
            if output is None:
                print("No Audio output")
                continue
            self.audio_output_frames.extend(self.audio_postprocess(output))
        return self.audio_output_frames.pop(0)
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        nodes_info = await self.client.get_available_nodes()
        return nodes_info