import pytest

from app.config import Settings


def test_settings_load_gemini_environment_mapping(monkeypatch):
    values = {
        "LLM_PROVIDER": "gemini",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "GEMINI_API_KEY": "",
        "PROVIDER_TIMEOUT_SECONDS": "15",
        "MAX_PROVIDER_CONCURRENCY": "3",
        "MAX_MODEL_TURNS_PER_ANSWER": "3",
        "MAX_RETRIES_PER_TURN": "1",
        "RETRY_BASE_DELAY_SECONDS": "1",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_environment()
    assert settings.provider == "gemini"
    assert settings.model == "gemini-2.5-flash"
    assert settings.gemini_api_key is None
    assert settings.provider_timeout_seconds == 15
    assert settings.max_provider_concurrency == 3
    assert settings.max_model_turns_per_answer == 3
    assert settings.max_retries_per_turn == 1


@pytest.mark.parametrize("kwargs", [
    {"provider": "openai"},
    {"provider_timeout_seconds": 14},
    {"max_provider_concurrency": 4},
    {"max_model_turns_per_answer": 4},
    {"max_retries_per_turn": 2},
    {"retry_base_delay_seconds": -1},
    {"default_agent_version": "v3"},
])
def test_settings_reject_out_of_policy_values(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)