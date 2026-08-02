from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override


class QwenFCNoThinkHandler(QwenFCHandler):
    """
    Qwen3 (hybrid-reasoning) in FC mode, forced into NON-thinking mode for
    self-hosted / local vLLM inference.

    Background: Qwen3's chat_template supports an ``enable_thinking`` kwarg —
    when false it appends an empty ``<think>\\n\\n</think>\\n\\n`` block right
    after the assistant generation cue, which forces the model to skip its
    reasoning trace and answer directly. BFCL's local path reimplements the
    template in Python and drives vLLM's *completions* endpoint with a raw
    ``prompt=`` string, so the ``chat_template_kwargs.enable_thinking`` flag
    (which only works on the API/chat path) never reaches the model. As a
    result the stock ``QwenFCHandler`` always runs Qwen3 in *thinking* mode.

    This handler mirrors the template's ``enable_thinking is false`` branch by
    appending the empty think block to the generation cue produced by the
    parent's ``_format_prompt``. This is the same trick ``SmolLM3FCHandler``
    uses for its forced ``/no_think`` mode.

    Everything else (tool-call extraction, response parsing which strips any
    ``</think>`` prefix, chat-history handling) is inherited unchanged.
    """

    _GENERATION_CUE = "<|im_start|>assistant\n"
    _EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = super()._format_prompt(messages, function)
        # Parent always ends the prompt with the bare assistant generation
        # cue; splice in the empty think block exactly as the Jinja
        # `enable_thinking is false` branch would.
        assert formatted_prompt.endswith(self._GENERATION_CUE), (
            "Unexpected prompt tail from QwenFCHandler._format_prompt; "
            "cannot inject the non-thinking block safely."
        )
        return formatted_prompt + self._EMPTY_THINK_BLOCK
