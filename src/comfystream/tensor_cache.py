import asyncio
import torch
from typing import List

image_inputs: List[torch.Tensor] = []
image_outputs: List[asyncio.Future] = []

audio_inputs: List[torch.Tensor] = []
audio_outputs: List[asyncio.Future] = []
