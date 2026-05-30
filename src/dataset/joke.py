from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class DatasetSource(Enum):
    RUOZHIBA = "ruozhiba"
    RUOZHIBA_RAW = "ruozhiba_raw"
    OOGIRI = "oogiri"
    ONE_LINER = "one_liner"
    R_JOKES = "r_jokes"
    CHUMOR = "chumor"


class JokeLanguage(Enum):
    ENGLISH = "en"
    CHINESE = "zh"


@dataclass(slots=True)
class JokeEntry:
    """A single joke with its setup/punchline segmentation."""

    text: str
    setup: str
    punchline: str
    source: DatasetSource
    language: JokeLanguage
    joke_id: str


@dataclass(slots=True)
class DistributionalMetrics:
    """Distributional properties extracted from a model for a single joke."""

    punchline_surprisal: np.float32  # mean neg-log-prob of punchline tokens
    pre_punchline_entropy: np.float32  # Shannon entropy at last pre-punchline position
    embedding_distance: np.float32  # cosine distance between setup and punchline reps
    punchline_token_surprisals: NDArray[np.float32]  # per-token surprisals
    entropy_trajectory: NDArray[np.float32]  # entropy at each position


@dataclass(slots=True)
class HumorJudgment:
    """Multi-dimensional humor rating from LLM-as-judge or human."""

    unexpectedness: np.float32
    cleverness: np.float32
    amusement: np.float32
    overall: np.float32
    rationale: str = ""


@dataclass(slots=True)
class JokeAnalysisEntry:
    """Complete analysis record for one joke under one model."""

    joke: JokeEntry
    model_name: str
    metrics: DistributionalMetrics
    judgment: HumorJudgment | None = None


@dataclass
class DecodingConfig:
    """Decoding strategy parameters for controlled generation experiments."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = []
            if self.temperature != 1.0:
                parts.append(f"t{self.temperature}")
            if self.top_p != 1.0:
                parts.append(f"p{self.top_p}")
            if self.top_k > 0:
                parts.append(f"k{self.top_k}")
            self.label = "_".join(parts) if parts else "greedy"


@dataclass(slots=True)
class ChumorEntry:
    """A Chumor joke paired with an explanation and its correctness label.

    Used for linear probing: does the model's internal representation
    distinguish correct from incorrect humor explanations?
    """

    joke_id: str
    joke: str
    explanation: str
    explanation_correct: bool  # True = explanation correctly accounts for the joke
    explanation_source: str    # "G" (GPT-4o) or "E" (ERNIE-4-turbo)
    language: JokeLanguage = JokeLanguage.CHINESE


# Standard decoding configs for the temperature sweep experiment
DECODING_CONFIGS = [
    DecodingConfig(temperature=0.3, label="low_temp"),
    DecodingConfig(temperature=0.7, label="mid_temp"),
    DecodingConfig(temperature=1.0, label="default"),
    DecodingConfig(temperature=1.3, label="high_temp"),
    DecodingConfig(temperature=1.6, label="very_high_temp"),
    DecodingConfig(temperature=1.0, top_p=0.5, label="narrow_nucleus"),
    DecodingConfig(temperature=1.0, top_p=0.9, label="wide_nucleus"),
]
