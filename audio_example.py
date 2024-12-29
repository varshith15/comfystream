import json
import asyncio
import torchaudio

from comfystream.client import ComfyStreamClient

async def main():
    cwd = "/home/user/ComfyUI"        
    client = ComfyStreamClient(cwd=cwd, type="audio")
    with open("./workflows/audio-whsiper-example-workflow.json", "r") as f:
        prompt = json.load(f)

    client.set_prompt(prompt)
    waveform, sr = torchaudio.load("/home/user/harvard.wav")
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)
    sr = 16000
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    chunk_ms = 20
    chunk_size = int(sr * (chunk_ms / 1000.0))

    total_samples = waveform.shape[1]
    offset = 0

    results = []
    while offset < total_samples:
        end = min(offset + chunk_size, total_samples)
        chunk = waveform[:, offset:end]
        offset = end
        results.append(await client.queue_prompt(chunk.numpy().squeeze()))

    print("Result:")
    for result in results:
        if result[0] is not None:
            print(result[-1])

if __name__ == "__main__":
    asyncio.run(main())