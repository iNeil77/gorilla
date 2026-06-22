from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from overrides import override


OCTOLONG_SYSTEM_PROMPT = "You are OctoLong, a helpful and interactive tool-calling agent."


class OctoLongHandler(QwenHandler):
    """
    Handler for the OctoLong-Qwen3 family in prompt mode.

    OctoLong's chat_template.jinja is byte-identical to Qwen3-4B-Instruct-2507's
    template, so QwenHandler's `_format_prompt` (with the `last_query_index`
    mechanism and the plain `<|im_start|>assistant\\n` generation cue) renders
    correctly verbatim. The only OctoLong-specific need is prepending the
    identity string `"You are OctoLong, a helpful and interactive tool-calling
    agent."` to the synthesized system message that the base prompt-mode
    pipeline injects with the function docs.
    """

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        # super() (OSSHandler) injects a system message with the BFCL function
        # docs into test_entry["question"][0]. Prepend OctoLong identity to it.
        first_turn = test_entry["question"][0]
        if first_turn and first_turn[0]["role"] == "system":
            first_turn[0]["content"] = (
                OCTOLONG_SYSTEM_PROMPT + "\n\n" + first_turn[0]["content"]
            )
        return inference_data
