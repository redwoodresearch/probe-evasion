"""Shared constants. Cloud resource names follow the Redwood convention
{project-id}_{owner-id}_{label}_kill-{YYYY-MM-DD}."""

TARGET_MODEL = "Qwen/Qwen3.5-27B"
OFF_POLICY_GENERATOR = "claude-sonnet-5-5"

APPS_DATASET = "RoganInglis/apps-control-arena"
SPLIT_SEED = "probe-evasion-v1"  # salt for the deterministic train/test split
TEST_FRACTION = 0.5

_PREFIX = "probe-evasion_nabil-baugher"
_KILL = "kill-2026-11-30"
MODAL_APP = f"{_PREFIX}_experiments_{_KILL}"
HF_CACHE_VOLUME = f"{_PREFIX}_hf-cache_{_KILL}"
ACTIVATIONS_VOLUME = f"{_PREFIX}_activations_{_KILL}"

# Qwen3.5 recommended sampling for thinking mode on precise coding tasks.
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
# Qwen3.5 reasons at length on APPS: at 16k tokens most traces were cut off before any code.
MAX_NEW_TOKENS = 32_768
MAX_MODEL_LEN = 36_864
# Prompts x samples per Sampler container; smaller chunks spread work across more GPUs.
CHUNK_SEQS = 64

MAX_TESTS_PER_PROBLEM = 100
TEST_TIMEOUT_S = 4.0
