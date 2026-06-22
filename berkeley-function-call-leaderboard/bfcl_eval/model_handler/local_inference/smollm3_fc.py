import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


SMOLLM3_NO_THINK_SYSTEM_PROMPT = "/no_think"


class SmolLM3FCHandler(OSSHandler):
    """
    Handler for HuggingFaceTB/SmolLM3-3B in function-calling mode.

    Mirrors SmolLM3's chat_template.jinja byte-for-byte (verified against
    `transformers.apply_chat_template`). SmolLM3's template has several
    quirks that don't match the OctoLong/Qwen3 family:

    - The system block always emits a Metadata + Custom Instructions header,
      not just the bare system message.
    - Assistant turns are wrapped with `<think>\\n\\n</think>\\n` in /no_think
      mode (and the same prefix appears at the generation cue).
    - Tool calls inside an assistant message are NOT rendered by the
      template's main loop — they get dropped. We emit them inline with
      the assistant content so the wire format still includes them.
    - Tool role messages are rendered as plain user turns (no
      `<tool_response>` wrapping); BFCL passes raw JSON in the content.
    - The system tools block is closed with a literal `\\n\\n<|im_end|>\\n`
      separator (not `<|im_end|>\\n` directly).
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        # Resolve system message + reasoning mode (matches Jinja lines 13-24)
        system_message = ""
        reasoning_mode = "/no_think"  # we always force /no_think for FC eval
        custom_instructions = ""
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"] or ""
            # /no_think and /think substrings are stripped from the rendered
            # custom_instructions even when present in the system message.
            custom_instructions = (
                system_message.replace("/no_think", "").replace("/think", "").rstrip()
            )

        # Header (Jinja lines 13-69). We always render the standard
        # Metadata/Custom Instructions form (no /system_override path).
        header = "<|im_start|>system\n"
        header += "## Metadata\n\n"
        header += "Knowledge Cutoff Date: June 2025\n"
        header += "Today Date: 01 January 2025\n"
        header += f"Reasoning Mode: {reasoning_mode}\n\n"

        header += "## Custom Instructions\n\n"
        if custom_instructions:
            header += custom_instructions + "\n\n"
        else:
            # /no_think default
            header += "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n"

        # Tools section
        if function:
            header += "### Tools\n\n"
            header += (
                "You may call one or more functions to assist with the user query.\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n\n"
                "<tools>\n"
            )
            # SmolLM3's Jinja uses `{tool | string}` (Python repr-style)
            # rather than `tojson` — single-quoted keys, no backslash-escapes.
            for tool in function:
                header += str(tool) + "\n"
            header += (
                "</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            )
            header += "\n\n"
            header += "<|im_end|>\n"

        formatted_prompt = header

        # Main loop (Jinja lines 70-86)
        for idx, message in enumerate(messages):
            if idx == 0 and message["role"] == "system":
                continue  # already consumed above
            role = message["role"]
            content = message["content"] or ""

            if role == "user":
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                # /no_think prefix
                formatted_prompt += "<|im_start|>assistant\n<think>\n\n</think>\n"
                # The Jinja template only renders `content` for assistant
                # turns — it silently drops tool_calls. Mirroring that exactly
                # would lose the tool calls from history. To stay faithful
                # while still preserving information, we splice tool_calls
                # into the content as the same `<tool_call>...</tool_call>`
                # blocks the model is trained to emit.
                inline = content.lstrip("\n") if content else ""
                if "tool_calls" in message and message["tool_calls"]:
                    tc_chunks = []
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        args = tool_call["arguments"]
                        if not isinstance(args, str):
                            args = json.dumps(args)
                        tc_chunks.append(
                            '<tool_call>\n{"name": "' + tool_call["name"]
                            + '", "arguments": ' + args + "}\n</tool_call>"
                        )
                    tc_text = "\n".join(tc_chunks)
                    inline = (inline + ("\n" if inline else "") + tc_text)
                formatted_prompt += inline + "<|im_end|>\n"
            elif role == "tool":
                # Template renders tool messages as plain user turns
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"

        # Generation prompt (Jinja lines 87-94, /no_think branch)
        formatted_prompt += "<|im_start|>assistant\n<think>\n\n</think>\n"
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {
            "message": [{"role": "system", "content": SMOLLM3_NO_THINK_SYSTEM_PROMPT}],
            "function": functions,
        }

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": model_response,
            }

        return {
            "model_responses": model_response,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception:
                pass
        return result
