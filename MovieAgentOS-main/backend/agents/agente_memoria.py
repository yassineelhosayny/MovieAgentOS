"""MemoryAgent di MovieAgentOS.

Estrazione preferenze LLM-first: l'LLM interpreta sempre il testo dell'utente
(cosi' qualsiasi parafrasi di "mi piace <genere>" viene riconosciuta); le
regole deterministiche del modulo condiviso `linguaggio` servono da fallback
offline e da validazione (un genere salvato deve essere un genere TMDB reale).

Divisione dei ruoli della memoria:
- SQLite (profilo_utente, film_visti, feedback): dati precisi e vincoli.
- ChromaDB: memoria semantica morbida (gusti, temi, film apprezzati/non
  apprezzati) usata SOLO per personalizzare il rerank, mai come vincolo.
"""

import re
from typing import Any

from agno.agent import Agent

from backend.agents.linguaggio import (
    estrai_generi_da_testo,
    estrai_generi_film,
    estrai_generi_per_sentimento,
    estrai_valore_film,
    filtra_generi_validi,
    chiave_genere,
    limita_score,
    normalizza_testo,
)
from backend.agents.llm_config import crea_model_agente, get_model_id, set_llm_api_key
from backend.agents.schemas import MemoryAgentOutput, PreferenzeEstratteMemoria
from backend.agents.text_utils import estrai_json_da_testo
from backend.db import crud
from backend.db.database import SessionLocal
from backend.tools import chroma_tools

_PAROLE_NON_NOME = {
    "stanco", "stanca", "triste", "felice", "contento", "contenta",
    "arrabbiato", "arrabbiata", "annoiato", "annoiata", "confuso", "confusa",
}


# Estrae con prudenza il nome dell'utente dal testo (fallback offline).
def cerca_nome_in_testo(testo: str) -> str | None:
    match = re.search(
        r"\b(?:mi\s+chiamo|chiamami|il\s+mio\s+nome\s+(?:è|e'|e)|io\s+sono|sono)\s+"
        r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]*){0,2})"
        r"(?=\s*(?:[,.;:!?-]|$))",
        testo or "",
        re.IGNORECASE,
    )
    if not match:
        return None
    nome = match.group(1).strip(" .,!?:;")
    if normalizza_testo(nome) in _PAROLE_NON_NOME:
        return None
    return nome[:1].upper() + nome[1:] if nome else None


# Estrazione deterministica di fallback (usata solo se l'LLM non risponde).
def estrai_memoria_semplice(testo: str) -> PreferenzeEstratteMemoria | None:
    testo_pulito = (testo or "").strip()
    if not testo_pulito:
        return None

    nome = cerca_nome_in_testo(testo_pulito)
    if nome:
        return PreferenzeEstratteMemoria(
            is_memoria=True,
            tipo_memoria="fatto_personale",
            summary_memoria=f"Il nome dell'utente è {nome}.",
            confidence=0.9,
        )

    generi_positivi, generi_negativi = estrai_generi_per_sentimento(testo_pulito)
    if not generi_positivi and not generi_negativi:
        return None

    preferenze_json: dict[str, float] = {}
    for genere in generi_positivi:
        preferenze_json[chiave_genere(genere) or genere.lower()] = 0.5
    for genere in generi_negativi:
        preferenze_json[chiave_genere(genere) or genere.lower()] = -0.7

    parti = []
    if generi_positivi:
        parti.append("generi preferiti: " + ", ".join(generi_positivi))
    if generi_negativi:
        parti.append("generi da evitare: " + ", ".join(generi_negativi))

    return PreferenzeEstratteMemoria(
        is_memoria=True,
        tipo_memoria="vincolo" if generi_negativi else "preferenza",
        generi_preferiti=generi_positivi,
        generi_da_evitare=generi_negativi,
        preferenze_positive=[testo_pulito] if generi_positivi else [],
        preferenze_negative=[testo_pulito] if generi_negativi else [],
        vincoli=[testo_pulito] if generi_negativi else [],
        preferenze_json=preferenze_json,
        summary_memoria="L'utente ha indicato " + "; ".join(parti) + ".",
        confidence=0.75,
    )


ISTRUZIONI_MEMORIA = [
    "Sei il MemoryAgent di MovieAgentOS.",
    "Rispondi solo con JSON valido, senza markdown e senza testo prima o dopo.",
    "Il JSON deve rispettare i campi di PreferenzeEstratteMemoria.",
    "Il tuo compito è leggere un testo dell'utente (con il contesto della conversazione) ed estrarre preferenze cinematografiche utili.",
    "Usa il CONTESTO per capire anche le preferenze INDIRETTE ma chiare: se l'utente chiede ripetutamente commedie, gli piacciono le commedie; se rifiuta i film proposti ('non mi sono piaciuti'), salva cosa non gli è piaciuto di quei film (visibili nel contesto).",
    "Salva solo ciò che è chiaramente una preferenza dell'utente: mai inventare.",
    "Riconosci QUALSIASI parafrasi: 'adoro la fantascienza', 'il fantasy mi annoia', 'mi piace un genere leggero', 'guarderei volentieri qualcosa di romantico'.",
    "Estrai solo memoria utile per il profilo cinematografico o fatti personali semplici (es. il nome).",
    "Non cercare film e non raccomandare film.",
    "Se il testo è una richiesta di raccomandazione senza nuove preferenze, imposta is_memoria=False.",
    "tipo_memoria: preferenza | avversione | vincolo | feedback | fatto_personale | non_memoria.",
    "generi_preferiti: solo generi esplicitamente apprezzati.",
    "generi_da_evitare: solo generi esplicitamente rifiutati (anche con negazioni come 'ma non horror', 'tranne horror', typo inclusi).",
    "Un genere in generi_da_evitare NON deve mai comparire anche in generi_preferiti: prevale il rifiuto.",
    "preferenze_positive/negative: gusti o stili non riducibili a un genere (es. 'film lenti', 'finali tristi').",
    "vincoli: limiti come durata, lingua, tono o contenuti da evitare.",
    "Un feedback negativo su un singolo film non diventa avversione per tutti i suoi generi.",
    "Se l'utente dice che gli è piaciuto un film specifico, salva il desiderio di film con storia/trama/atmosfera simili a quel film.",
    "preferenze_json: segnali morbidi tra -1 e 1 (chiave = genere o concetto, valore = intensità).",
    "summary_memoria: una frase breve, completa e utile da salvare.",
    "confidence tra 0 e 1.",
]


def Errore_MemoryAgentOutput(errore: str, id_utente: int | None = None, query: str = "", log_notes: dict[str, Any] | None = None) -> MemoryAgentOutput:
    return MemoryAgentOutput(
        success=False,
        id_utente=id_utente,
        query_originale=query,
        errors=[errore],
        log_notes=log_notes or {},
    )


def leggi_contesto_memoria(id_utente: int, query: str, num_memoria: int = 5) -> MemoryAgentOutput:
    """Preferenze precise da SQLite + memorie semantiche da ChromaDB."""
    log_notes: dict[str, Any] = {"agent": "MemoryAgent", "azione": "leggi_contesto_memoria"}

    if id_utente is None or id_utente <= 0:
        return Errore_MemoryAgentOutput("id_utente non valido o mancante.", id_utente, query, log_notes)
    if not query or not query.strip():
        return Errore_MemoryAgentOutput("Query memoria vuota o non valida.", id_utente, "", log_notes)

    query_pulita = query.strip()

    db = SessionLocal()
    try:
        profilo_utente = crud.ottieni_profilo(db, id_utente)
    except Exception as e:
        return Errore_MemoryAgentOutput(f"Errore durante la lettura del profilo utente database: {e}", id_utente, query_pulita, log_notes)
    finally:
        db.close()

    if profilo_utente is None:
        return Errore_MemoryAgentOutput("Profilo utente non trovato nel database.", id_utente, query_pulita, log_notes)

    try:
        memoria_chroma = chroma_tools.cerca_memorie_utente(
            id_utente=id_utente, query=query_pulita, num_results=num_memoria
        )
    except Exception as e:
        memoria_chroma = []
        log_notes["warning_chroma"] = str(e)

    if memoria_chroma and isinstance(memoria_chroma[0], dict) and "errore" in memoria_chroma[0]:
        log_notes["warning_chroma"] = memoria_chroma[0]["errore"]
        memoria_chroma = []

    memorie_rilevanti = [
        {
            "testo": mem.get("testo") or mem.get("documento") or "",
            "tipo": mem.get("metadata", {}).get("tipo", "") if isinstance(mem.get("metadata"), dict) else mem.get("tipo", ""),
            "distance": mem.get("distance"),
        }
        for mem in (memoria_chroma or [])
        if isinstance(mem, dict) and (mem.get("testo") or mem.get("documento") or "").strip()
    ]

    log_notes["numero_memorie_rilevanti"] = len(memorie_rilevanti)
    return MemoryAgentOutput(
        success=True,
        id_utente=id_utente,
        query_originale=query_pulita,
        generi_preferiti=filtra_generi_validi(profilo_utente.get("generi_preferiti", [])),
        generi_da_evitare=filtra_generi_validi(profilo_utente.get("generi_da_evitare", [])),
        preferenze_json=profilo_utente.get("preferenze", {}),
        summary_testuale=profilo_utente.get("summary_testuale") or "",
        memorie_rilevanti=memorie_rilevanti,
        errors=[],
        log_notes=log_notes,
    )


def unisci_liste_senza_duplicati(lista_base: list[str], nuovi_valori: list[str]) -> list[str]:
    risultato: list[str] = []
    visti: set[str] = set()
    for valore in list(lista_base or []) + list(nuovi_valori or []):
        if not valore:
            continue
        chiave = chiave_genere(valore) or normalizza_testo(valore)
        if chiave and chiave not in visti:
            visti.add(chiave)
            risultato.append(str(valore))
    return filtra_generi_validi(risultato) or risultato


def aggiorna_preferenze_json(preferenze_attuali: dict[str, Any], nuove_preferenze: dict[str, float]) -> dict[str, float]:
    risultato: dict[str, float] = {}
    for chiave, valore in (preferenze_attuali or {}).items():
        try:
            risultato[str(chiave)] = float(valore)
        except Exception:
            continue
    for chiave, valore in (nuove_preferenze or {}).items():
        chiave_pulita = chiave_genere(chiave) or normalizza_testo(chiave).replace(" ", "_")
        if not chiave_pulita:
            continue
        try:
            nuovo_score = limita_score(float(valore))
        except Exception:
            continue
        if chiave_pulita in risultato:
            risultato[chiave_pulita] = limita_score((risultato[chiave_pulita] + nuovo_score) / 2)
        else:
            risultato[chiave_pulita] = nuovo_score
    return risultato


def aggiorna_summary_testuale(summary_attuale: str | None, nuova_summary: str) -> str:
    summary_attuale = summary_attuale or ""
    nuova_summary = (nuova_summary or "").strip()
    if not nuova_summary:
        return summary_attuale

    parti = [riga.strip() for riga in summary_attuale.splitlines() if riga.strip()]
    parti.append(nuova_summary)

    deduplicate: list[str] = []
    viste: set[str] = set()
    for parte in parti:
        chiave = normalizza_testo(parte)
        if chiave and chiave not in viste:
            deduplicate.append(parte)
            viste.add(chiave)

    summary_finale = "\n".join(deduplicate)
    if len(summary_finale) <= 1500:
        return summary_finale

    mantenute: list[str] = []
    lunghezza = 0
    for parte in reversed(deduplicate):
        aggiunta = len(parte) + (1 if mantenute else 0)
        if lunghezza + aggiunta > 1500:
            continue
        mantenute.insert(0, parte)
        lunghezza += aggiunta
    return "\n".join(mantenute) if mantenute else summary_finale[:1500]


def _generi_eco_assistente(contesto: str, testo_utente: str) -> set[str]:
    """Generi nominati SOLO dall'assistente nella conversazione (es. proposti
    in una domanda: 'ti va anche il romance?') e mai dall'utente: non sono
    preferenze dell'utente e l'LLM non deve salvarli come tali."""
    righe_assistente: list[str] = []
    righe_utente: list[str] = [testo_utente or ""]
    for riga in (contesto or "").splitlines():
        ruolo = riga.split("]:", 1)[0].lstrip("[").strip().lower() if "]:" in riga else ""
        if ruolo in ("assistant", "assistente"):
            righe_assistente.append(riga)
        else:
            righe_utente.append(riga)
    citati_assistente = {chiave_genere(g) for g in estrai_generi_da_testo(" ".join(righe_assistente))}
    citati_utente = {chiave_genere(g) for g in estrai_generi_da_testo(" ".join(righe_utente))}
    return citati_assistente - citati_utente - {None}


def _valida_estrazione(estrazione: PreferenzeEstratteMemoria, testo: str,
                       generi_eco: set[str] | None = None) -> PreferenzeEstratteMemoria:
    """Validazione deterministica dell'output LLM col modulo condiviso:
    generi canonici reali, negazioni locali rispettate, rifiuto > preferenza,
    niente generi 'eco' (proposti dall'assistente, mai detti dall'utente)."""
    generi_pos_det, generi_neg_det = estrai_generi_per_sentimento(testo)
    eco = generi_eco or set()

    estrazione.generi_preferiti = unisci_liste_senza_duplicati(
        [g for g in filtra_generi_validi(estrazione.generi_preferiti) if chiave_genere(g) not in eco],
        generi_pos_det,
    )
    estrazione.generi_da_evitare = unisci_liste_senza_duplicati(
        [g for g in filtra_generi_validi(estrazione.generi_da_evitare) if chiave_genere(g) not in eco],
        generi_neg_det,
    )

    # la negazione deterministica prevale: un genere negato localmente
    # ("ma non horror") non puo' restare tra i preferiti
    negati = {chiave_genere(g) for g in estrazione.generi_da_evitare}
    estrazione.generi_preferiti = [
        g for g in estrazione.generi_preferiti if chiave_genere(g) not in negati
    ]

    if estrazione.generi_preferiti or estrazione.generi_da_evitare:
        estrazione.is_memoria = True
        if estrazione.tipo_memoria in ("non_memoria", ""):
            estrazione.tipo_memoria = "vincolo" if estrazione.generi_da_evitare else "preferenza"
        estrazione.confidence = max(float(estrazione.confidence or 0.0), 0.75)

    preferenze_json = dict(estrazione.preferenze_json or {})
    for genere in estrazione.generi_preferiti:
        preferenze_json[chiave_genere(genere) or genere.lower()] = max(
            float(preferenze_json.get(chiave_genere(genere) or genere.lower(), 0.0)), 0.5
        )
    for genere in estrazione.generi_da_evitare:
        preferenze_json[chiave_genere(genere) or genere.lower()] = -0.7
    estrazione.preferenze_json = preferenze_json

    if not estrazione.summary_memoria.strip() and (estrazione.generi_preferiti or estrazione.generi_da_evitare):
        parti = []
        if estrazione.generi_preferiti:
            parti.append("generi preferiti: " + ", ".join(estrazione.generi_preferiti))
        if estrazione.generi_da_evitare:
            parti.append("generi da evitare: " + ", ".join(estrazione.generi_da_evitare))
        estrazione.summary_memoria = "L'utente ha indicato " + "; ".join(parti) + "."

    return estrazione


def aggiorna_memoria_da_testo(id_utente: int, testo_utente: str, contesto: str = "") -> MemoryAgentOutput:
    """Estrae preferenze dal testo (LLM-first, col contesto della conversazione
    per capire anche le preferenze indirette) e aggiorna profilo SQLite + ChromaDB."""
    log_notes: dict[str, Any] = {
        "agent": "MemoryAgent",
        "azione": "aggiorna_memoria_da_testo",
        "id_utente": id_utente,
    }

    if id_utente is None or id_utente <= 0:
        return Errore_MemoryAgentOutput("id_utente non valido o mancante.", id_utente, testo_utente or "", log_notes)
    if not testo_utente or not testo_utente.strip():
        return Errore_MemoryAgentOutput("Testo utente vuoto: impossibile aggiornare memoria.", id_utente, "", log_notes)

    testo_pulito = testo_utente.strip()
    log_notes["testo_pulito"] = testo_pulito

    # LLM-first: l'LLM interpreta sempre; le regole sono solo fallback offline.
    estrazione: PreferenzeEstratteMemoria | None = None
    try:
        set_llm_api_key("Memoria")
        agente = Agent(
            model=crea_model_agente("memoria"),
            markdown=False,
            instructions=ISTRUZIONI_MEMORIA,
        )
        input_llm = testo_pulito
        if contesto.strip():
            input_llm = f"CONTESTO CONVERSAZIONE:\n{contesto.strip()}\n\nMESSAGGIO ATTUALE DELL'UTENTE:\n{testo_pulito}"
        risposta = agente.run(input_llm)
        content = risposta.content
        if isinstance(content, PreferenzeEstratteMemoria):
            estrazione = content
        elif isinstance(content, str):
            estrazione = PreferenzeEstratteMemoria.model_validate(estrai_json_da_testo(content))
        else:
            estrazione = PreferenzeEstratteMemoria.model_validate(content)
        log_notes["estrazione_llm"] = True
    except Exception as e:
        log_notes["errore_llm"] = f"{type(e).__name__}: {e}"
        estrazione = estrai_memoria_semplice(testo_pulito)
        log_notes["fallback_memoria_semplice"] = estrazione is not None
        if estrazione is None:
            return Errore_MemoryAgentOutput(
                f"Errore durante l'estrazione preferenze con LLM: {e}", id_utente, testo_pulito, log_notes
            )

    estrazione = _valida_estrazione(estrazione, testo_pulito,
                                    generi_eco=_generi_eco_assistente(contesto, testo_pulito))
    log_notes["estrazione"] = estrazione.model_dump()

    if not estrazione.is_memoria:
        return Errore_MemoryAgentOutput(
            "Il testo non contiene una memoria/preferenza utile da salvare.", id_utente, testo_pulito, log_notes
        )
    if estrazione.confidence < 0.35:
        return Errore_MemoryAgentOutput(
            "Confidence troppo bassa: preferenza non salvata per evitare memoria rumorosa.", id_utente, testo_pulito, log_notes
        )

    #profilo sqlite 
    db = SessionLocal()
    try:
        profilo_attuale = crud.ottieni_profilo(db, id_utente)
        if profilo_attuale is None:
            crud.crea_profilo_se_non_esiste(db, id_utente)
            profilo_attuale = crud.ottieni_profilo(db, id_utente)
        if profilo_attuale is None:
            return Errore_MemoryAgentOutput("Profilo utente non trovato e impossibile crearlo.", id_utente, testo_pulito, log_notes)

        generi_preferiti_finali = unisci_liste_senza_duplicati(
            profilo_attuale.get("generi_preferiti", []), estrazione.generi_preferiti
        )
        generi_da_evitare_finali = unisci_liste_senza_duplicati(
            profilo_attuale.get("generi_da_evitare", []), estrazione.generi_da_evitare
        )

        # una nuova dichiarazione sposta il genere dall'altra lista
        nuovi_pref = {chiave_genere(g) for g in estrazione.generi_preferiti}
        nuovi_evit = {chiave_genere(g) for g in estrazione.generi_da_evitare}
        generi_da_evitare_finali = [g for g in generi_da_evitare_finali if chiave_genere(g) not in nuovi_pref]
        generi_preferiti_finali = [g for g in generi_preferiti_finali if chiave_genere(g) not in nuovi_evit]

        preferenze_finali = aggiorna_preferenze_json(
            profilo_attuale.get("preferenze", {}), estrazione.preferenze_json
        )
        summary_finale = aggiorna_summary_testuale(
            profilo_attuale.get("summary_testuale"), estrazione.summary_memoria
        )

        crud.aggiorna_profilo(
            db=db,
            id_utente=id_utente,
            generi_preferiti=generi_preferiti_finali,
            generi_da_evitare=generi_da_evitare_finali,
            preferenze=preferenze_finali,
            summary_testuale=summary_finale,
        )
        print(f"[Memoria]    profilo aggiornato OK | preferiti: {', '.join(generi_preferiti_finali) or '-'} | da evitare: {', '.join(generi_da_evitare_finali) or '-'}")
    except Exception as e:
        return Errore_MemoryAgentOutput(f"Errore durante aggiornamento profilo SQLite: {e}", id_utente, testo_pulito, log_notes)
    finally:
        db.close()

    # --- memoria semantica Chroma (morbida) ---
    testo_memoria = estrazione.summary_memoria.strip() or f"L'utente ha espresso una preferenza cinematografica: {testo_pulito}"
    try:
        risultato_chroma = chroma_tools.salva_memoria_utente(
            id_utente=id_utente,
            testo=testo_memoria,
            tipo=estrazione.tipo_memoria,
            fonte="chat",
            metadata_extra={
                "testo_originale": testo_pulito,
                "confidence": estrazione.confidence,
                "generi_preferiti": estrazione.generi_preferiti,
                "generi_da_evitare": estrazione.generi_da_evitare,
            },
        )
    except Exception as e:
        # Chroma e' memoria morbida: un suo errore non blocca il turno
        risultato_chroma = {"success": False, "errore": f"{type(e).__name__}: {e}"}
    if not risultato_chroma.get("success"):
        log_notes["warning_chroma"] = risultato_chroma.get("errore", "Errore sconosciuto ChromaDB.")
    else:
        log_notes["id_memoria_chroma"] = risultato_chroma.get("id_memoria")

    return MemoryAgentOutput(
        success=True,
        id_utente=id_utente,
        query_originale=testo_pulito,
        generi_preferiti=generi_preferiti_finali,
        generi_da_evitare=generi_da_evitare_finali,
        preferenze_json=preferenze_finali,
        summary_testuale=summary_finale,
        memorie_rilevanti=[risultato_chroma] if risultato_chroma.get("success") else [],
        errors=[],
        log_notes=log_notes,
    )


def aggiorna_memoria_da_feedback(id_utente: int, film: Any, positivo: bool, commento: str | None = None) -> MemoryAgentOutput:
    """Feedback thumbs up/down su UN film.

    Il dato preciso (film visto, gradito o no) vive in SQLite (film_visti +
    feedback, gestiti dal chiamante). Qui si aggiornano:
    - i segnali morbidi del profilo (preferenze_json per genere);
    - la memoria semantica Chroma, usata poi SOLO nel rerank.
    """
    log_notes: dict[str, Any] = {
        "agent": "MemoryAgent",
        "azione": "aggiorna_memoria_da_feedback",
        "id_utente": id_utente,
        "positivo": positivo,
    }

    if id_utente is None or id_utente <= 0:
        return Errore_MemoryAgentOutput("id_utente non valido o mancante.", id_utente, "", log_notes)

    titolo_film = (
        estrai_valore_film(film, "titolo")
        or estrai_valore_film(film, "title")
        or estrai_valore_film(film, "titolo_originale")
        or estrai_valore_film(film, "original_title")
    )
    if not titolo_film or not str(titolo_film).strip():
        return Errore_MemoryAgentOutput("titolo film mancante: impossibile aggiornare memoria da feedback.", id_utente, "", log_notes)

    titolo_film = str(titolo_film).strip()
    generi = estrai_generi_film(film)
    commento_pulito = commento.strip() if commento and commento.strip() else None
    log_notes.update({"titolo_film": titolo_film, "generi": generi})
    print(f"[Feedback]   {'OK' if positivo else 'NO'} '{titolo_film}' - aggiorno segnali profilo e memoria semantica")

    # --- segnali morbidi nel profilo SQLite ---
    db = SessionLocal()
    try:
        profilo_attuale = crud.ottieni_profilo(db, id_utente)
        if profilo_attuale is None:
            crud.crea_profilo_se_non_esiste(db, id_utente)
            profilo_attuale = crud.ottieni_profilo(db, id_utente)
        if profilo_attuale is None:
            return Errore_MemoryAgentOutput("Profilo utente non trovato e impossibile crearlo.", id_utente, titolo_film, log_notes)

        preferenze_attuali = profilo_attuale.get("preferenze", {})
        delta = 0.12 if positivo else -0.12
        nuovi_segnali = {genere: delta for genere in generi}
        preferenze_finali: dict[str, float] = {}
        for chiave, valore in (preferenze_attuali or {}).items():
            try:
                preferenze_finali[str(chiave)] = float(valore)
            except Exception:
                continue
        for genere, delta_genere in nuovi_segnali.items():
            chiave = chiave_genere(genere) or normalizza_testo(genere)
            preferenze_finali[chiave] = limita_score(preferenze_finali.get(chiave, 0.0) + delta_genere)

        if positivo:
            summary_feedback = (
                f"All'utente è piaciuto '{titolo_film}'. Per i consigli futuri sono adatti "
                f"film con storia, trama, temi e atmosfera simili a '{titolo_film}'."
            )
        else:
            summary_feedback = (
                f"All'utente NON è piaciuto '{titolo_film}'. Film molto simili per storia, "
                f"trama o atmosfera sono meno adatti; questo non vieta i suoi generi in generale."
            )
        if commento_pulito:
            summary_feedback += f" Commento: {commento_pulito}."

        summary_finale = aggiorna_summary_testuale(profilo_attuale.get("summary_testuale"), summary_feedback)

        crud.aggiorna_profilo(
            db=db,
            id_utente=id_utente,
            preferenze=preferenze_finali,
            summary_testuale=summary_finale,
        )
        generi_preferiti_finali = filtra_generi_validi(profilo_attuale.get("generi_preferiti", []))
        generi_da_evitare_finali = filtra_generi_validi(profilo_attuale.get("generi_da_evitare", []))
    except Exception as e:
        return Errore_MemoryAgentOutput(f"Errore durante aggiornamento profilo SQLite da feedback: {e}", id_utente, titolo_film, log_notes)
    finally:
        db.close()

    # --- memoria semantica Chroma ---
    try:
        risultato_chroma = chroma_tools.salva_feedback_memoria(
            id_utente=id_utente,
            titolo_film=titolo_film,
            positivo=positivo,
            commento=commento_pulito,
            generi=generi,
        )
    except Exception as e:
        risultato_chroma = {"success": False, "errore": f"{type(e).__name__}: {e}"}
    if not risultato_chroma.get("success"):
        log_notes["warning_chroma"] = risultato_chroma.get("errore", "Errore sconosciuto ChromaDB.")
        print(f"[Feedback]   profilo aggiornato OK | ChromaDB errore: {str(log_notes['warning_chroma'])[:60]}")
    else:
        log_notes["id_memoria_chroma"] = risultato_chroma.get("id_memoria")
        print(f"[Feedback]   profilo aggiornato OK | memoria semantica salvata in ChromaDB OK")

    return MemoryAgentOutput(
        success=True,
        id_utente=id_utente,
        query_originale=titolo_film,
        generi_preferiti=generi_preferiti_finali,
        generi_da_evitare=generi_da_evitare_finali,
        preferenze_json=preferenze_finali,
        summary_testuale=summary_finale,
        memorie_rilevanti=[risultato_chroma] if risultato_chroma.get("success") else [],
        errors=[],
        log_notes=log_notes,
    )


def costruisci_testo_contesto_memoria(memoria_output: MemoryAgentOutput) -> str:
    """Contesto compatto per il router LLM. I dati precisi (film visti,
    esclusioni) NON passano da qui: viaggiano come tmdb_id in RichiestaRicerca."""
    if not memoria_output or not memoria_output.success:
        return ""

    sezioni: list[str] = []
    if memoria_output.generi_da_evitare:
        sezioni.append("Generi da evitare (vincolo forte): " + ", ".join(memoria_output.generi_da_evitare))
    if memoria_output.generi_preferiti:
        sezioni.append("Generi preferiti (preferenza morbida): " + ", ".join(memoria_output.generi_preferiti))
    if memoria_output.preferenze_json:
        segnali = [f"{chiave}={valore}" for chiave, valore in list(memoria_output.preferenze_json.items())[:12]]
        sezioni.append("Segnali morbidi: " + ", ".join(segnali))
    if memoria_output.summary_testuale and memoria_output.summary_testuale.strip():
        sezioni.append("Profilo: " + memoria_output.summary_testuale.strip()[:400])

    return "\n".join(sezioni)


def estrai_memorie_semantiche(memoria_output: MemoryAgentOutput, limite: int = 4) -> list[str]:
    """Testi delle memorie Chroma rilevanti: SOLO per personalizzare il rerank."""
    if not memoria_output or not memoria_output.success:
        return []
    testi: list[str] = []
    for mem in memoria_output.memorie_rilevanti or []:
        if not isinstance(mem, dict):
            continue
        testo = (mem.get("testo") or mem.get("documento") or "").strip()
        if testo:
            testi.append(testo[:250])
        if len(testi) >= limite:
            break
    return testi
