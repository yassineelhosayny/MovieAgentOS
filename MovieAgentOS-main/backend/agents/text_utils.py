"""
Utility di testo condivise tra gli agenti.
Centralizza l'estrazione di JSON dall'output dell'LLM.
"""
import json
import re
from typing import Any


def estrai_json_da_testo(testo: str) -> dict[str, Any]:
    """
    Estrae il PRIMO oggetto JSON valido dal testo prodotto da un LLM.
    Tollera: fence markdown (```json ... ```), testo prima/dopo il JSON,
    piu' oggetti JSON concatenati ("Extra data"). Solleva JSONDecodeError
    solo se nel testo non c'e' alcun oggetto JSON valido.
    """
    testo_pulito = (testo or "").strip()
    if testo_pulito.startswith("```"):
        testo_pulito = testo_pulito.strip("`").strip()
        if testo_pulito.lower().startswith("json"):
            testo_pulito = testo_pulito[4:].strip()

    try:
        risultato = json.loads(testo_pulito)
        if isinstance(risultato, dict):
            return risultato
    except json.JSONDecodeError:
        pass

    # primo oggetto JSON nel testo: raw_decode ignora il contenuto successivo
    decoder = json.JSONDecoder()
    indice = testo_pulito.find("{")
    while indice != -1:
        try:
            risultato, _ = decoder.raw_decode(testo_pulito[indice:])
            if isinstance(risultato, dict):
                return risultato
        except json.JSONDecodeError:
            pass
        indice = testo_pulito.find("{", indice + 1)

    # riparazione: alcuni modelli scrivono JSON in stile JavaScript, con le
    # chiavi senza virgolette ({ azione: "x" }). Le quotiamo e riproviamo.
    riparato = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', testo_pulito)
    if riparato != testo_pulito:
        indice = riparato.find("{")
        while indice != -1:
            try:
                risultato, _ = decoder.raw_decode(riparato[indice:])
                if isinstance(risultato, dict):
                    return risultato
            except json.JSONDecodeError:
                pass
            indice = riparato.find("{", indice + 1)

    raise json.JSONDecodeError("Nessun oggetto JSON valido nel testo.", testo_pulito, 0)
