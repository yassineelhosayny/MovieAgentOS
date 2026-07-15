import os
from typing import Any

import streamlit as st

#modello fallback, funziona sempre però è lento,viene usato solo se modulo in .toml non funziona, 1min di lantency i totale
DEFAULT_MODEL_ID = "openai/gpt-oss-20b:free"


# Legge un valore dai secrets Streamlit, None se assente.
def leggi_secret(nome: str) -> str | None:
    try:
        valore = st.secrets.get(nome)
    except Exception:
        return None
    if valore is None:
        return None
    valore = str(valore).strip()
    return valore or None


# OpenRouter richiede l'ID con prefisso provider ("openai/gpt-4.1-nano").
# Se nel .toml manca il prefisso, lo deduciamo dal nome del modello.
_PREFISSI_PROVIDER = (
    (("gpt", "chatgpt", "o1", "o3", "o4"), "openai/"),
    (("claude",), "anthropic/"),
    (("gemini", "gemma"), "google/"),
    (("llama",), "meta-llama/"),
    (("mistral", "mixtral"), "mistralai/"),
    (("qwen",), "qwen/"),
    (("deepseek",), "deepseek/"),
)


def _normalizza_model_id(model_id: str) -> str:
    model_id = (model_id or "").strip().strip('"').strip("'")
    if not model_id or "/" in model_id:
        return model_id
    basso = model_id.lower()
    for prefissi, provider in _PREFISSI_PROVIDER:
        if basso.startswith(prefissi):
            return provider + model_id
    return model_id


# Restituisce l'ID del modello LLM (da secret o default), valido per TUTTI
# gli agenti del progetto (router, memoria, rerank, chiarimenti).
def get_model_id(nome_agente: str | None = None) -> str:
    return _normalizza_model_id(leggi_secret("MODEL_ID_LLM") or DEFAULT_MODEL_ID)


# Imposta la chiave OpenRouter come variabile d'ambiente; errore se manca.
def set_llm_api_key(nome_agente: str | None = None) -> None:
    api_key = leggi_secret("API_KEY_LLM") or leggi_secret("OPENROUTER_API_KEY")
    if not api_key:
        nome = nome_agente or "LLM"
        raise ValueError(f"API key OpenRouter mancante per agente '{nome}'.")
    os.environ["OPENROUTER_API_KEY"] = api_key


# Crea l'istanza del modello LLM usata dagli agenti agno.
def crea_model_agente(nome_agente: str | None = None) -> Any:
    model_id = get_model_id(nome_agente)
    set_llm_api_key(nome_agente)
    try:
        from agno.models.openrouter import OpenRouter
    except ModuleNotFoundError as e:
        nome = nome_agente or "LLM"
        raise RuntimeError(
            "Dipendenza Python mancante per OpenRouter. "
            f"Agente: {nome}. Modello configurato: {model_id}. "
            "Installa le dipendenze Agno/OpenRouter."
        ) from e
    return OpenRouter(id=model_id)
