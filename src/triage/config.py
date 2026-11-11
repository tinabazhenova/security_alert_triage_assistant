"""Reads configuration from environment variables."""

import os
from dataclasses import dataclass

REQUIRED_VARS = ("WATSONX_URL", "WATSONX_PROJECT_ID", "WATSONX_APIKEY")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    watsonx_url: str
    watsonx_project_id: str
    watsonx_apikey: str


def load_settings() -> Settings:
    """Read watsonx settings from the environment.

    Called explicitly (not at import time) so importing this module never
    requires credentials.
    """
    values = {name: os.environ.get(name, "").strip() for name in REQUIRED_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in your shell or in .env (see .env.example)."
        )
    return Settings(
        watsonx_url=values["WATSONX_URL"],
        watsonx_project_id=values["WATSONX_PROJECT_ID"],
        watsonx_apikey=values["WATSONX_APIKEY"],
    )
