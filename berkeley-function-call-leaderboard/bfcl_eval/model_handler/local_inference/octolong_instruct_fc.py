from bfcl_eval.model_handler.local_inference.octolong_fc import OctoLongFCHandler


class OctoLongInstructFCHandler(OctoLongFCHandler):
    """
    Alias for OctoLongFCHandler.

    Historically this was a separate class for the OctoLong-Qwen3-Instruct
    M1/M2/LC and yantri-tool/Qwen3.5-*-FC family because their chat template
    differed from the OctoLong-128K-Stage3 template. After verifying against
    HF, the templates are identical (byte-for-byte equal to
    Qwen3-4B-Instruct-2507's chat_template.jinja), so the same handler covers
    both groups.
    """
    pass
