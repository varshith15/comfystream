from comfystream import tensor_cache

class SaveResult:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "result": ("RESULT",),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, result):
        fut = tensor_cache.audio_outputs.pop()
        fut.set_result(result)
        return result