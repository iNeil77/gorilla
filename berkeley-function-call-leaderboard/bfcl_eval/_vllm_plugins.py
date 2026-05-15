"""vLLM plugin entry points for BFCL.

vLLM 0.21 only registers ``Qwen3_5ForConditionalGeneration`` (the multimodal
class) for the Qwen3.5 architecture, so checkpoints whose ``config.json``
declares ``architectures=["Qwen3_5ForCausalLM"]`` get routed through that
multimodal class and crash on ``config.vision_config`` for text-only
checkpoints (e.g. our Yantri/OctoLong Qwen3.5 fine-tunes).

The text-only ``Qwen3_5ForCausalLM`` class already exists in
``vllm.model_executor.models.qwen3_5`` — it just isn't wired into the
registry, and the upstream class also forgets to inherit from
``IsHybrid``, which means vLLM never runs the hybrid-mamba config hook
that populates ``mamba_block_size``.

We define a thin subclass with ``is_hybrid = True`` and register it
under the ``Qwen3_5ForCausalLM`` arch name. Because vLLM introspects
the registered class in a subprocess via the lazy-import string
(``<module>:<class>``), the class definition has to live in an
importable module rather than be patched at runtime — otherwise the
subprocess re-imports the upstream class without our patch.

The plugin runs in every vLLM process (the ``bfcl generate`` parent, the
``vllm serve`` API process, and the engine-core / worker subprocesses).
"""

try:
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM as _Qwen3_5ForCausalLMTextBase,
    )
except Exception:
    _Qwen3_5ForCausalLMTextBase = None


if _Qwen3_5ForCausalLMTextBase is not None:

    class Qwen3_5ForCausalLMHybrid(_Qwen3_5ForCausalLMTextBase):
        """Text-only Qwen3.5 causal LM with the IsHybrid marker upstream forgot.

        See module docstring for the full rationale.
        """

        is_hybrid = True

else:
    Qwen3_5ForCausalLMHybrid = None  # type: ignore[assignment]


def register_qwen3_5_text_arch_alias() -> None:
    try:
        from vllm import ModelRegistry
    except ImportError:
        return

    if Qwen3_5ForCausalLMHybrid is None:
        return

    existing = ModelRegistry.models.get("Qwen3_5ForCausalLM")
    existing_class = getattr(existing, "class_name", None)
    if existing_class == "Qwen3_5ForCausalLMHybrid":
        return

    try:
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "bfcl_eval._vllm_plugins:Qwen3_5ForCausalLMHybrid",
        )
    except Exception:
        pass
