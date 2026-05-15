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
that populates ``mamba_block_size``. This plugin:

1. Registers ``Qwen3_5ForCausalLM`` under that arch name so vLLM picks
   the text-only class instead of the multimodal one.
2. Stamps ``is_hybrid = True`` on the class so vLLM treats it as a
   hybrid attention/mamba model and applies the right cache defaults.

The plugin runs in every vLLM process (the ``bfcl generate`` parent, the
``vllm serve`` API process, and the engine-core / worker subprocesses).
"""


def register_qwen3_5_text_arch_alias() -> None:
    try:
        from vllm import ModelRegistry
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
    except ImportError:
        return

    # Mark as hybrid so HybridAttentionMambaModelConfig.verify_and_update_config
    # runs and populates mamba_block_size for the GDN linear-attention layers.
    if not getattr(Qwen3_5ForCausalLM, "is_hybrid", False):
        try:
            Qwen3_5ForCausalLM.is_hybrid = True
        except Exception:
            pass

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
