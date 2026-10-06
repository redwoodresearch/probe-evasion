"""Modal resources: a vLLM sampler on GPU and a network-blocked CPU executor for
model-written code."""

import modal

from probe_evasion import config

app = modal.App(config.MODAL_APP)

hf_cache = modal.Volume.from_name(config.HF_CACHE_VOLUME, create_if_missing=True)
ACTS_DIR = "/acts"
acts_volume = modal.Volume.from_name(config.ACTIVATIONS_VOLUME, create_if_missing=True)
PROBE_LAYERS = (16, 24, 32, 40, 48)  # of 64; select by validation AUROC

# CUDA devel base: FlashInfer JIT-compiles its sampling and Qwen3.5 linear-attention kernels with nvcc.
vllm_base_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HOME": "/hf-cache"})
)
vllm_image = vllm_base_image.add_local_python_source("probe_evasion")
# HF forward passes for activations; FLA provides fast kernels for Qwen3.5's linear attention.
extract_image = vllm_base_image.uv_pip_install("flash-linear-attention", "accelerate").add_local_python_source("probe_evasion")

# No secrets, volumes, network, or Modal API access: this runs untrusted code.
exec_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("probe_evasion")


@app.function(
    image=exec_image,
    cpu=1,
    timeout=20 * 60,
    block_network=True,
    restrict_modal_access=True,
    max_containers=100,
)
def execute(job: dict) -> dict:
    from probe_evasion.executor import check

    return check(job)


@app.cls(
    image=vllm_image,
    gpu="H100",
    volumes={"/hf-cache": hf_cache},
    timeout=2 * 60 * 60,
    scaledown_window=5 * 60,
    max_containers=40,
)
class Sampler:
    model_id: str = modal.parameter(default=config.TARGET_MODEL)

    @modal.enter()
    def load(self):
        from huggingface_hub import snapshot_download
        from vllm import LLM

        # Persist weights before engine start so a failed init doesn't re-download them.
        snapshot_download(self.model_id)
        hf_cache.commit()
        self.llm = LLM(
            model=self.model_id,
            max_model_len=config.MAX_MODEL_LEN,
            gpu_memory_utilization=0.92,
            # Qwen3.5's linear-attention layers need one state slot per running sequence;
            # an H100 holds ~358, so stay below vLLM's default of 1024.
            max_num_seqs=256,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )

    @modal.method()
    def chat(
        self, conversations: list[list[dict]], n: int = 1, seed: int = 0, thinking: bool = True,
        max_tokens: int = config.MAX_NEW_TOKENS,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        sampling = config.SAMPLING if thinking else config.SAMPLING_NO_THINKING
        params = SamplingParams(n=n, seed=seed, max_tokens=max_tokens, **sampling)
        outs = self.llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": thinking})
        return [[o.text for o in out.outputs] for out in outs]
