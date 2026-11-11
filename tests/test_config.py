import pytest

from triage.config import REQUIRED_VARS, ConfigError, load_settings


@pytest.fixture
def full_env(monkeypatch):
    monkeypatch.setenv("WATSONX_URL", "https://example.invalid")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")
    monkeypatch.setenv("WATSONX_APIKEY", "test-key")


def test_load_settings_reads_env(full_env):
    settings = load_settings()
    assert settings.watsonx_url == "https://example.invalid"
    assert settings.watsonx_project_id == "test-project"
    assert settings.watsonx_apikey == "test-key"


@pytest.mark.parametrize("missing", REQUIRED_VARS)
def test_missing_variable_raises_clear_error(full_env, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ConfigError, match=missing):
        load_settings()


def test_blank_variable_counts_as_missing(full_env, monkeypatch):
    monkeypatch.setenv("WATSONX_APIKEY", "   ")
    with pytest.raises(ConfigError, match="WATSONX_APIKEY"):
        load_settings()
