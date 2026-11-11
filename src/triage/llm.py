"""LLM and embedding clients: get_llm(), get_embeddings()."""

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM

from triage.config import Settings, load_settings

LLM_MODEL_ID = "ibm/granite-4-h-small"
EMBEDDING_MODEL_ID = "ibm/granite-embedding-278m-multilingual"

LLM_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}

EMBEDDING_PARAMS = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
    EmbedParams.RETURN_OPTIONS: {"input_text": True},
}


def get_llm(settings: Settings | None = None) -> WatsonxLLM:
    """Return a Granite LLM client. Credentials come from config, not code."""
    settings = settings or load_settings()
    return WatsonxLLM(
        model_id=LLM_MODEL_ID,
        url=settings.watsonx_url,
        project_id=settings.watsonx_project_id,
        api_key=settings.watsonx_apikey,
        params=LLM_PARAMS,
    )


def get_embeddings(settings: Settings | None = None) -> WatsonxEmbeddings:
    """Return a Granite embedding client. Credentials come from config, not code."""
    settings = settings or load_settings()
    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        url=settings.watsonx_url,
        project_id=settings.watsonx_project_id,
        api_key=settings.watsonx_apikey,
        params=EMBEDDING_PARAMS,
    )
