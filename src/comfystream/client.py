import torch
import asyncio

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfystream import tensor_cache
from comfystream.utils import convert_prompt


class ComfyStreamClient:
    def __init__(self, type: str = "image", **kwargs):
        config = Configuration(**kwargs)
        # TODO: Need to handle cleanup for EmbeddedComfyClient if not using async context manager?
        self.comfy_client = EmbeddedComfyClient(config)
        self.prompt = None
        self.type = type.lower()
        if self.type not in {"image", "audio"}:
            raise ValueError(f"Unsupported type: {self.type}. Supported types are 'image' and 'audio'.")
        
        self.input_cache = getattr(tensor_cache, f"{self.type}_inputs", None)
        self.output_cache = getattr(tensor_cache, f"{self.type}_outputs", None)
        
        if self.input_cache is None or self.output_cache is None:
            raise AttributeError(f"tensor_cache does not have attributes for type '{self.type}'.")

    def set_prompt(self, prompt: PromptDictInput):
        self.prompt = convert_prompt(prompt)

    async def queue_prompt(self, input: torch.Tensor) -> torch.Tensor:
        self.input_cache.append(input)

        output_fut = asyncio.Future()
        self.output_cache.append(output_fut)

        await self.comfy_client.queue_prompt(self.prompt)

        return await output_fut
