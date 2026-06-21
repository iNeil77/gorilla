import json

from bfcl_eval.model_handler.local_inference.granite_3 import Granite3FCHandler
from overrides import override


class Granite33FCHandler(Granite3FCHandler):
    """
    Handler for Granite-3.3-8B-Instruct.

    The chat template is identical to Granite-3.1/3.2 except that the
    tool-schema block is emitted under the `available_tools` role rather
    than `tools` (the 3.3 Jinja aliases `tools` -> `available_tools` at
    render time, so the model has only seen the renamed form during
    training). bfcl bypasses the chat template and builds the prompt
    manually, so we override the role header to match.

    The new <think>/<response> reasoning blocks are opt-in via
    `thinking=True` in `apply_chat_template`, which the BFCL pipeline
    never sets — so the default tool-calling path stays untouched.
    """

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        else:
            system_prompt = (
                "Knowledge Cutoff Date: April 2024.\n"
                "Today's Date: April 29, 2025.\n"
                "You are Granite, developed by IBM."
            )
            if function:
                system_prompt += (
                    " You are a helpful AI assistant with access "
                    "to the following tools. When a tool is required to answer the user's query, respond "
                    "with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist "
                    "in the provided list of tools, notify the user that you do not have the ability to fulfill the request."
                )

        formatted_prompt += (
            f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
        )

        if function:
            formatted_prompt += (
                "<|start_of_role|>available_tools<|end_of_role|>"
                + json.dumps(function, indent=4)
                + "<|end_of_text|>\n"
            )

        for msg in messages:
            formatted_prompt += (
                "<|start_of_role|>"
                + msg["role"]
                + "<|end_of_role|>"
                + msg["content"]
                + "<|end_of_text|>\n"
            )

        return formatted_prompt
