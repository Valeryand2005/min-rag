
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"




@dataclass
class ExperimentConfig:
    # one experiment setup
    name: str
    embedding_model: str
    chunk_size: int
    generator_model: str
    tokenizer_model: Optional[str] = None

    # for chat models like tinyllama / phi / gemma
    use_chat_format: bool = False 

    def __post_init__(self):
        if self.tokenizer_model is None:
            self.tokenizer_model = self.embedding_model

    @property
    def exp_dir(self) -> Path:
        safe_name = (
            self.name.replace(" ", "_")
            .replace("/", "-")
            .replace("(", "")
            .replace(")", "")
        )
        return EXPERIMENTS_DIR / safe_name


# baseline experiment
# this is our reference point to compare everything else
BASELINE = ExperimentConfig(
    name="baseline",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256,
    generator_model="distilgpt2",
)


# different embedding models
# we expect better retrieval quality compared to baseline
EMBED_MPNET = ExperimentConfig(
    name="embed_all-mpnet-base-v2",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    chunk_size=256,
    generator_model="distilgpt2",
)

EMBED_BGE = ExperimentConfig(
    name="embed_bge-small-en-v1.5",
    embedding_model="BAAI/bge-small-en-v1.5",
    chunk_size=256,
    generator_model="distilgpt2",
)

# different chunk sizes
# goal is to see how context granularity affects retrieval
CHUNK_128 = ExperimentConfig(
    name="chunk_128",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=128,
    generator_model="distilgpt2",
)

CHUNK_512 = ExperimentConfig(
    name="chunk_512",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=512,
    generator_model="distilgpt2",
)

# stronger generator models
# expected to produce better answers but may be slower
GEN_TINYLLAMA = ExperimentConfig(
    name="gen_TinyLlama-1.1B",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256,
    generator_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_chat_format=True,
)

GEN_PHI2 = ExperimentConfig(
    name="gen_phi-2",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256,
    generator_model="microsoft/phi-2",
    use_chat_format=True,
)

GEN_GEMMA2B = ExperimentConfig(
    name="gen_gemma-2b",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256,
    generator_model="google/gemma-2b",
    use_chat_format=True,
)

# list of all experiments we run
# each one is a separate test case for comparison
ALL_EXPERIMENTS: List[ExperimentConfig] = [
    BASELINE,
    EMBED_MPNET,
    EMBED_BGE,
    CHUNK_128,
    CHUNK_512,
    GEN_TINYLLAMA,
    GEN_PHI2,
    GEN_GEMMA2B,
]

QUICK_EXPERIMENTS: List[ExperimentConfig] = [
    BASELINE,
    CHUNK_128,
    CHUNK_512,
]