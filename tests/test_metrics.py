"""Tests for distributional metrics.

These run without GPU by mocking the model outputs.
For integration tests with real models, run with --integration flag.
"""

import numpy as np
import pytest

from src.analysis.inverted_u import fit_inverted_u
from src.analysis.alignment_comparison import compare_distributions
from src.dataset.joke import DecodingConfig, JokeEntry, DatasetSource, JokeLanguage
from src.dataset.loader import _split_one_liner


class TestOneLinerSplitting:
    def test_dash_split(self):
        text = "I told my wife she was drawing her eyebrows too high — she looked surprised."
        setup, punchline = _split_one_liner(text)
        assert "eyebrows" in setup
        assert "surprised" in punchline

    def test_colon_split(self):
        text = "My dating life: it's like a fairy tale, except the dragon wins."
        setup, punchline = _split_one_liner(text)
        assert "dating" in setup

    def test_fallback_split(self):
        text = "I used to be indecisive but now I'm not so sure"
        setup, punchline = _split_one_liner(text)
        assert len(setup) > 0
        assert len(punchline) > 0


class TestInvertedU:
    def test_perfect_inverted_u(self):
        np.random.seed(42)
        x = np.linspace(-3, 3, 200)
        y = -x**2 + 5 + np.random.normal(0, 0.5, 200)

        result = fit_inverted_u(x, y, metric_name="test")
        assert result.is_inverted_u
        assert result.quadratic_coeff < 0
        assert result.r_squared > 0.8
        assert result.peak_x is not None
        assert abs(result.peak_x) < 1.0  # peak near 0

    def test_linear_relationship(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 1, 100)

        result = fit_inverted_u(x, y, metric_name="linear")
        # Should not detect inverted-U for a linear relationship
        assert not result.is_inverted_u or result.r_squared < 0.1

    def test_too_few_samples(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 1.0])

        result = fit_inverted_u(x, y, metric_name="small")
        assert result.n_samples == 3
        assert not result.is_inverted_u


class TestAlignmentComparison:
    def test_significant_shift(self):
        np.random.seed(42)
        base = np.random.normal(5.0, 1.0, 500)
        aligned = np.random.normal(3.0, 0.8, 500)

        result = compare_distributions(base, aligned, "test_metric")
        assert result.p_value < 0.001
        assert result.mean_shift < 0  # aligned is lower
        assert abs(result.cohens_d) > 0.5  # large effect

    def test_no_shift(self):
        np.random.seed(42)
        base = np.random.normal(5.0, 1.0, 500)
        aligned = np.random.normal(5.0, 1.0, 500)

        result = compare_distributions(base, aligned, "test_metric")
        assert result.p_value > 0.01
        assert abs(result.cohens_d) < 0.3


class TestDecodingConfig:
    def test_auto_label(self):
        config = DecodingConfig(temperature=0.5)
        assert config.label == "t0.5"

    def test_combined_label(self):
        config = DecodingConfig(temperature=0.7, top_p=0.9)
        assert "t0.7" in config.label
        assert "p0.9" in config.label

    def test_explicit_label(self):
        config = DecodingConfig(temperature=0.5, label="custom")
        assert config.label == "custom"


class TestJokeEntry:
    def test_creation(self):
        joke = JokeEntry(
            text="Setup\nPunchline",
            setup="Setup",
            punchline="Punchline",
            source=DatasetSource.R_JOKES,
            language=JokeLanguage.ENGLISH,
            joke_id="test_001",
        )
        assert joke.source == DatasetSource.R_JOKES
        assert joke.language == JokeLanguage.ENGLISH
