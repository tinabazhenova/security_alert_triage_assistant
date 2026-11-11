"""Offline tests for triage.llm. No credentials or network needed."""

import pytest

from triage import llm
from triage.config import ConfigError, Settings

FAKE = Settings(
    watsonx_url="https://example.invalid",
    watsonx_project_id="test-project",
    watsonx_apikey="test-key",
)


def test_get_llm_passes_config_through(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "WatsonxLLM", lambda **kw: captured.update(kw))
    llm.get_llm(FAKE)
    assert captured["model_id"] == llm.LLM_MODEL_ID
    assert captured["url"] == FAKE.watsonx_url
    assert captured["project_id"] == FAKE.watsonx_project_id
    assert captured["api_key"] == FAKE.watsonx_apikey


def test_get_embeddings_passes_config_through(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "WatsonxEmbeddings", lambda **kw: captured.update(kw))
    llm.get_embeddings(FAKE)
    assert captured["model_id"] == llm.EMBEDDING_MODEL_ID
    assert captured["url"] == FAKE.watsonx_url
    assert captured["project_id"] == FAKE.watsonx_project_id
    assert captured["api_key"] == FAKE.watsonx_apikey


def test_get_llm_without_env_fails_with_config_error(monkeypatch):
    monkeypatch.delenv("WATSONX_APIKEY", raising=False)
    with pytest.raises(ConfigError):
        llm.get_llm()
