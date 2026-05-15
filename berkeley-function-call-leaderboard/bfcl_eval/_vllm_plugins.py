"""vLLM plugin entry points for BFCL.

vLLM 0.21 only registers ``Qwen3_5ForConditionalGeneration`` (the multimodal
class) for the Qwen3.5 architecture, so checkpoints whose ``config.json``
declares ``architectures=["Qwen3_5ForCausalLM"]`` get routed through that
multimodal class and crash on ``config.vision_config`` for text-only
checkpoints (e.g. our Yantri/OctoLong Qwen3.5 fine-tunes).

The text-only ``Qwen3_5ForCausalLM`` class already exists in
``vllm.model_executor.models.qwen3_5`` — it just isn't wired into the
registry. This plugin registers it under that arch name so vLLM picks it
for text-only checkpoints.

The plugin runs in every vLLM process (the ``bfcl generate`` parent, the
``vllm serve`` API process, and the engine-core / worker subprocesses).
"""


def register_qwen3_5_text_arch_alias() -> None:
    try:
        from vllm import ModelRegistry
    except ImportError:
        return

    if "Qwen3_5ForCausalLM" in ModelRegistry.get_supported_archs():
        existing = ModelRegistry.models.get("Qwen3_5ForCausalLM")
        existing_class = getattr(existing, "class_name", None)
        if existing_class == "Qwen3_5ForCausalLM":
            return

    try:
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
        )
    except Exception:
        pass
