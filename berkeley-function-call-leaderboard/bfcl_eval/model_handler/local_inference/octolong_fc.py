from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override


OCTOLONG_SYSTEM_PROMPT = "You are OctoLong, a helpful and interactive tool-calling agent."


class OctoLongFCHandler(QwenFCHandler):
    """
    Handler for the OctoLong-Qwen3 / OctoLong-Instruct family in FC mode.

    OctoLong's chat_template.jinja is byte-identical to Qwen3-4B-Instruct-2507's
    template (verified — same `last_query_index`/`<tool_call>`/`<tool_response>`
    framing, same `<|im_start|>assistant\\n` generation cue, no `<think>` block).
    The only OctoLong-specific need is injecting the identity system prompt
    `"You are OctoLong, a helpful and interactive tool-calling agent."` so the
    Jinja's `tools`-branch prepends it to the `# Tools` system message.
    """

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {
            "message": [{"role": "system", "content": OCTOLONG_SYSTEM_PROMPT}],
            "function": functions,
        }
