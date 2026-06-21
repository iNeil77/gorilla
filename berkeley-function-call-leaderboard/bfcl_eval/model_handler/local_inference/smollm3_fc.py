from bfcl_eval.model_handler.local_inference.octolong_fc import OctoLongFCHandler
from overrides import override


SMOLLM3_NO_THINK_SYSTEM_PROMPT = "/no_think"


class SmolLM3FCHandler(OctoLongFCHandler):
    """
    Handler for HuggingFaceTB/SmolLM3-3B in function-calling mode.

    SmolLM3 emits tool calls in the exact same shape as the OctoLong/Qwen
    family — a `<tools>...</tools>` block injected into the system message
    and `<tool_call>{"name": ..., "arguments": ...}</tool_call>` for
    invocations — so the OctoLong template/parsing logic carries over.

    The two SmolLM3-specific quirks we have to address:

    1. SmolLM3 has thinking on by default; we set the system prompt to
       `/no_think` so the model skips reasoning traces (the chat template
       reads this flag and injects an empty `<think></think>` block).
       OctoLongFCHandler already preseeds an empty think block at the
       generation cue, which lines up with /no_think behaviour.
    2. We drop the "You are OctoLong, ..." identity — SmolLM3 isn't
       OctoLong and shouldn't be told it is.
    """

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {
            "message": [
                {"role": "system", "content": SMOLLM3_NO_THINK_SYSTEM_PROMPT}
            ],
            "function": functions,
        }
