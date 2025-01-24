import torch
import asyncio
from typing import List

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfystream.utils import convert_prompt


class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, **kwargs):
        config = Configuration(**kwargs)
        # TODO: Need to handle cleanup for EmbeddedComfyClient if not using async context manager?
        self.comfy_client = EmbeddedComfyClient(config, max_workers=max_workers)

    def set_prompts(self, prompts: List[PromptDictInput]):
        for prompt in [convert_prompt(prompt) for prompt in prompts]:
            asyncio.create_task(self.run_prompt(prompt))

    async def run_prompt(self, prompt: PromptDictInput):
        # TODO: Handle workflow changing mid stream
        while True:
            await self.comfy_client.queue_prompt(prompt)
