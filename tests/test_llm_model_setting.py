"""GF_LLM_MODEL: missing, empty or blank means the default model; an explicit choice is kept. No API calls."""
import os
import subprocess
import sys

import pytest
from dotenv import dotenv_values

from golden_fighter import config
from golden_fighter.llm import LLMSetupError, build_llm

SONNET = "anthropic/claude-sonnet-5-5"


@pytest.mark.parametrize("value", [None, "", " ", "   ", "\t", " \t\n "])
def test_missing_empty_or_blank_setting_uses_the_default(value):
    assert config.llm_model_from(value) == config.DEFAULT_LLM_MODEL == "anthropic/claude-haiku-5-5"


@pytest.mark.parametrize("value, expected", [(SONNET, SONNET), (f"  {SONNET}\t", SONNET),
                                             ("anthropic/claude-haiku-5-5", "anthropic/claude-haiku-5-5")])
def test_explicit_model_choice_is_kept(value, expected):
    assert config.llm_model_from(value) == expected


def test_env_example_as_shipped_selects_the_default():
    values = dotenv_values(config.ROOT / ".env.example")
    assert "GF_LLM_MODEL" in values and not (values["GF_LLM_MODEL"] or "").strip()
    assert config.llm_model_from(values["GF_LLM_MODEL"]) == config.DEFAULT_LLM_MODEL


@pytest.mark.parametrize("value, expected", [("", "anthropic/claude-haiku-5-5"), ("   ", "anthropic/claude-haiku-5-5"),
                                             (SONNET, SONNET)])
def test_config_reads_the_setting_at_import(value, expected):
    """A fresh interpreter, as in a real run: only imports config (no API call, no key needed)."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"} | {"GF_LLM_MODEL": value}
    out = subprocess.run([sys.executable, "-c", "from golden_fighter import config; print(config.LLM_MODEL)"],
                         env=env, capture_output=True, text=True, check=True, cwd=config.ROOT)
    assert out.stdout.strip() == expected


def test_empty_setting_no_longer_hides_the_missing_key_message(monkeypatch):
    """The original bug: an empty GF_LLM_MODEL failed with "got ''" before the API key was even checked."""
    monkeypatch.setattr(config, "LLM_MODEL", config.llm_model_from(""))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMSetupError, match="ANTHROPIC_API_KEY is not set"):
        build_llm()
