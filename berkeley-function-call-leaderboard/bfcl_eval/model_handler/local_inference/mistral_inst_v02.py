from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class MistralInstV02Handler(OSSHandler):
    """
    Handler for Mistral-7B-Instruct-v0.2 derivatives (e.g.
    aws-prototyping/MegaBeam-Mistral-7B-512k).

    The v0.2 chat template only supports user/assistant alternation with an
    optional initial system message — there are no tool branches. The system
    message, if present, is folded into the first user turn:

        <s>[INST] {system}\n\n{user1} [/INST] {assistant1}</s>[INST] {user2} [/INST] ...

    The base v0.2 tokenizer rejects any role other than user/assistant/system,
    so this is prompt-mode only. The BFCL prompt-mode pipeline injects function
    docs into the system message, which we then fold into the first user turn
    per the template.
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

    @override
    def _format_prompt(self, messages, function):
        bos_token = "<s>"
        eos_token = "</s>"

        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            loop_messages = messages[1:]
        else:
            system_message = None
            loop_messages = messages

        user_messages = [m for m in loop_messages if m["role"] == "user"]
        last_user = user_messages[-1] if user_messages else None

        formatted_prompt = bos_token
        for message in loop_messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                if message is last_user and system_message is not None:
                    formatted_prompt += f"[INST] {system_message}\n\n{content} [/INST]"
                else:
                    formatted_prompt += f"[INST] {content} [/INST]"
            elif role == "assistant":
                formatted_prompt += f" {content}{eos_token}"

        return formatted_prompt
