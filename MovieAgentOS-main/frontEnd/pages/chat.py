# pages/chat.py

from typing import Any

import streamlit as st

from backend.agents.agente_memoria import aggiorna_memoria_da_feedback
from backend.agents.orchestratorAgents import ConduttoreDiAgents
from backend.db import crud
from backend.db.database import SessionLocal
from gui.auth import is_loggato
from gui.mostra_pagina_loggato import mostra_pagina_loggato


def sintetizza_film_per_storico(film_list: list[dict[str, Any]]) -> str:
    righe: list[str] = []
    for film in film_list[:5]:
        titolo = film.get("titolo")
        if not titolo:
            continue
        anno = film.get("anno")
        generi = ", ".join(film.get("generi") or [])
        descrizione = " ".join(str(film.get("descrizione") or "").split())[:180]
        pezzi = [str(titolo)]
        if anno:
            pezzi.append(f"({anno})")
        if generi:
            pezzi.append(f"- generi: {generi}")
        if descrizione:
            pezzi.append(f"- trama: {descrizione}")
        righe.append("- " + " ".join(pezzi))
    return "\n".join(righe)


def carica_messaggi_sessione(db, id_sessione: int) -> list[dict[str, Any]]:
    messaggi_db = crud.ottieni_messaggi_sessione(db, id_sessione)
    messaggi: list[dict[str, Any]] = []
    for messaggio in messaggi_db:
        item = {
            "id_messaggio": messaggio["id_messaggio"],
            "ruolo": messaggio["ruolo"],
            "contenuto": messaggio["contenuto"],
        }
        if messaggio["ruolo"] == "assistant":
            item["film_consigliati"] = crud.ottieni_film_per_messaggio(db, messaggio["id_messaggio"])
        messaggi.append(item)
    return messaggi


def costruisci_storico_chat(messaggi: list[dict[str, Any]]) -> list[dict[str, str]]:
    storico: list[dict[str, str]] = []
    for messaggio in messaggi:
        if messaggio.get("ruolo") not in {"user", "assistant"} or not messaggio.get("contenuto"):
            continue

        contenuto = str(messaggio["contenuto"])
        film_storico = sintetizza_film_per_storico(messaggio.get("film_consigliati") or [])
        if film_storico:
            contenuto = f"{contenuto}\n\nFILM_CONSIGLIATI_STRUTTURATI:\n{film_storico}"

        storico.append(
            {
                "role": messaggio["ruolo"],
                "content": contenuto,
            }
        )
    return storico


def estrai_valore_film(film: Any, chiave: str, default=None):
    if isinstance(film, dict):
        return film.get(chiave, default)
    return getattr(film, chiave, default)


def normalizza_anno(anno: Any) -> int | None:
    if anno is None:
        return None
    try:
        return int(str(anno)[:4])
    except (TypeError, ValueError):
        return None


def crea_poster_url(poster_path: str | None) -> str | None:
    if not poster_path:
        return None
    if poster_path.startswith("http"):
        return poster_path
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def salva_film_consigliati(db, id_messaggio: int, id_utente: int, film_consigliati: list[Any]) -> None:
    for film in film_consigliati:
        tmdb_id = estrai_valore_film(film, "tmdb_id")
        titolo = estrai_valore_film(film, "titolo")

        if not tmdb_id or not titolo:
            continue

        poster_path = estrai_valore_film(film, "poster_path")
        voto_medio = estrai_valore_film(film, "voto_medio")

        crud.salva_film_consigliato(
            db=db,
            id_messaggio=id_messaggio,
            id_utente=id_utente,
            id_tmdb=tmdb_id,
            titolo=titolo,
            anno=normalizza_anno(estrai_valore_film(film, "anno")),
            descrizione=estrai_valore_film(film, "descrizione"),
            generi=estrai_valore_film(film, "generi", []),
            poster_path=poster_path,
            poster_url=crea_poster_url(poster_path),
            voto_medio=str(voto_medio) if voto_medio is not None else None,
            numero_voti=estrai_valore_film(film, "numero_voti"),
            motivo_raccomandazione=estrai_valore_film(film, "motivo_ricerca") or estrai_valore_film(film, "motivo_raccomandazione"),
        )


def prepara_film_per_feedback(film: dict[str, Any]) -> dict[str, Any]:
    return {
        "tmdb_id": film.get("id_tmdb"),
        "titolo": film.get("titolo"),
        "anno": film.get("anno"),
        "descrizione": film.get("descrizione"),
        "generi": film.get("generi") or [],
        "poster_path": film.get("poster_path"),
        "voto_medio": film.get("voto_medio"),
        "numero_voti": film.get("numero_voti"),
    }


def registra_feedback_film(db, id_utente: int, film: dict[str, Any], positivo: bool) -> None:
    """Feedback thumbs su UN film:
    - SQLite feedback (dato preciso) + film_visti (mai riproporre);
    - ChromaDB memoria semantica (personalizzazione futura, via MemoryAgent)."""
    crud.salva_feedback(
        db=db,
        id_utente=id_utente,
        id_consiglio=film["id_consiglio"],
        voto=10 if positivo else 1,
        commento=None,
    )
    if film.get("id_tmdb"):
        crud.segna_film_visto(
            db,
            id_utente=id_utente,
            id_tmdb=int(film["id_tmdb"]),
            titolo=film.get("titolo"),
            fonte="feedback",
            gradito=positivo,
        )
    print(f"\n[Feedback]   {'OK' if positivo else 'NO'} '{film.get('titolo')}' -> salvato in SQLite (feedback + film_visti: non verrà più riproposto)")
    aggiorna_memoria_da_feedback(
        id_utente=id_utente,
        film=prepara_film_per_feedback(film),
        positivo=positivo,
    )


def feedback_salvato_per_film(db, id_utente: int, film: dict[str, Any]) -> bool | None:
    """True = thumbs up, False = thumbs down, None = nessun feedback."""
    feedback = crud.ottieni_feedback_film(db, id_utente, film["id_consiglio"])
    if not feedback:
        return None
    return feedback["voto"] >= 7


def mostra_feedback_messaggio(db, id_utente: int, messaggio: dict[str, Any]) -> None:
    """Thumbs up/down separati per CIASCUN film consigliato nel messaggio."""
    if messaggio.get("ruolo") != "assistant" or not messaggio.get("id_messaggio"):
        return

    id_messaggio = messaggio["id_messaggio"]
    film_messaggio = crud.ottieni_film_per_messaggio(db, id_messaggio)
    if not film_messaggio:
        return

    st.caption("Ti sono piaciuti? Valuta ogni film:")
    disabilitato = st.session_state.get("agent_in_esecuzione", False)

    for film in film_messaggio:
        id_consiglio = film["id_consiglio"]
        stato_key = f"feedback_film_{id_consiglio}"
        salvato = st.session_state.get(stato_key)
        if salvato is None:
            salvato = feedback_salvato_per_film(db, id_utente, film)
            if salvato is not None:
                st.session_state[stato_key] = salvato

        col_titolo, col_up, col_down = st.columns([6, 1, 1])
        with col_titolo:
            anno = f" ({film.get('anno')})" if film.get("anno") else ""
            st.markdown(f"**{film.get('titolo')}**{anno}")

        if salvato is True:
            with col_up:
                st.markdown(":green[👍]")
            with col_down:
                st.markdown("")
        elif salvato is False:
            with col_up:
                st.markdown("")
            with col_down:
                st.markdown(":red[👎]")
        else:
            with col_up:
                if st.button("👍", key=f"up_{id_consiglio}", disabled=disabilitato,
                             help="Mi è piaciuto"):
                    registra_feedback_film(db, id_utente, film, positivo=True)
                    st.session_state[stato_key] = True
                    st.rerun()
            with col_down:
                if st.button("👎", key=f"down_{id_consiglio}", disabled=disabilitato,
                             help="Non mi è piaciuto"):
                    registra_feedback_film(db, id_utente, film, positivo=False)
                    st.session_state[stato_key] = False
                    st.rerun()


def avvia_elaborazione_agent() -> None:
    st.session_state.agent_in_esecuzione = True
    st.session_state.prompt_da_processare = st.session_state.get("chat_prompt_input", "")


def costruisci_risposta_errore(errors: list[str] | None) -> str:
    risposta_generica = (
        "Mi dispiace, non sono riuscito a completare la richiesta. "
        "Puoi riprovare o riformulare?"
    )

    if not errors:
        return risposta_generica

    dettaglio = "; ".join(str(errore) for errore in errors if errore)
    if not dettaglio:
        return risposta_generica

    dettaglio_lower = dettaglio.lower()
    if "tmdb_api_key" in dettaglio_lower or "chiave api tmdb" in dettaglio_lower or "tmdb mancante" in dettaglio_lower:
        return (
            "La ricerca dei film non e' pronta: manca o non e' valida la chiave TMDB. "
            "Configura TMDB_API_KEY nei secrets di Streamlit e riprova."
        )

    if "errore durante la richiesta a tmdb" in dettaglio_lower or "errore tmdb" in dettaglio_lower:
        return (
            "Ho capito la richiesta, ma in questo momento non riesco a contattare TMDB per cercare i film. "
            "Riprova tra poco; se continua, controlla connessione e chiave TMDB."
        )

    if "nessun film" in dettaglio_lower or "rispetta tutti i vincoli" in dettaglio_lower:
        return (
            "Non ho trovato film reali che rispettino tutti i vincoli. "
            "Prova ad allargare almeno un vincolo, ad esempio voto, anno, genere escluso o tema."
        )

    if "database" in dettaglio_lower or "sqlite" in dettaglio_lower or "no such table" in dettaglio_lower:
        return (
            "La risposta e' stata interrotta da un problema sul database locale. "
            "Inizializza il database e riprova."
        )

    if "resource_exhausted" in dettaglio_lower or "quota" in dettaglio_lower or "429" in dettaglio_lower:
        return (
            "Mi dispiace, in questo momento non posso completare la richiesta "
            "perche' il servizio di intelligenza artificiale ha raggiunto il limite temporaneo di utilizzo. "
            "Riprova tra qualche minuto."
        )

    if "api key openrouter" in dettaglio_lower or "openrouter_api_key" in dettaglio_lower:
        return "Il servizio di intelligenza artificiale non e' pronto: la chiave OpenRouter manca o non e' valida."

    return risposta_generica


def risposta_mostrabile(testo: str | None) -> bool:
    """Vero se il testo e' leggibile dall'utente finale (niente tecnicismi)."""
    if not testo or not testo.strip():
        return False
    basso = testo.lower()
    tecnici = ("traceback", "exception", "validation error", "jsondecode",
               "connecterror", "getaddrinfo", "typeerror", "keyerror", "http")
    return not any(token in basso for token in tecnici)


def costruisci_risposta_assistente(output) -> str:
    # la risposta dell'agente ha priorita' se e' pulita, anche in caso di errore
    if output.risposta and risposta_mostrabile(output.risposta):
        return output.risposta
    return costruisci_risposta_errore(output.errors)


def mostra_debug_errori(output) -> None:
    debug_errors = output.log_notes.get("debug_errors", []) if getattr(output, "log_notes", None) else []
    if not debug_errors:
        return
    with st.expander("Dettagli tecnici"):
        for errore in debug_errors:
            st.code(str(errore), language="text")


def genera_titolo_fallback(prompt: str) -> str:
    titolo = " ".join((prompt or "").strip().split())
    titolo = titolo.strip(" \t\n\r.,;:!?-")
    if not titolo:
        return "Nuova sessione"
    if len(titolo) > 42:
        titolo = titolo[:42].rstrip(" \t\n\r.,;:!?-") + "..."
    return titolo[0].upper() + titolo[1:] if titolo else "Nuova sessione"


def reimposta_stato_conversazione() -> None:
    """Stato di sessione esplicito: si azzera cambiando/creando sessione."""
    st.session_state.richiesta_precedente = None
    st.session_state.n_chiarimenti = 0
    st.session_state.n_raffinamenti = 0


if not is_loggato() or "id_utente" not in st.session_state:
    st.warning("Devi effettuare il login per usare la chat.")
    st.stop()


id_utente = st.session_state.id_utente
db = SessionLocal()

try:
    if "agent_in_esecuzione" not in st.session_state:
        st.session_state.agent_in_esecuzione = False
    if "richiesta_precedente" not in st.session_state:
        st.session_state.richiesta_precedente = None
    if "n_chiarimenti" not in st.session_state:
        st.session_state.n_chiarimenti = 0
    if "n_raffinamenti" not in st.session_state:
        st.session_state.n_raffinamenti = 0

    agent_in_esecuzione = st.session_state.agent_in_esecuzione
    prompt_da_processare = st.session_state.get("prompt_da_processare")

    with st.sidebar:
        st.markdown("### Sessioni")

        if agent_in_esecuzione:
            st.info("Attendi la fine della risposta prima di cambiare sessione.")

        if st.button("Nuova sessione", use_container_width=True, disabled=agent_in_esecuzione):
            st.session_state.id_sessione = crud.crea_sessione(db, id_utente, "Nuova sessione")
            st.session_state.messaggi = []
            reimposta_stato_conversazione()
            st.rerun()

        st.divider()

        sessioni = crud.ottieni_sessioni_utente(db, id_utente)

        for sessione in sessioni:
            col1, col2 = st.columns([4, 1])
            id_sessione_corrente = sessione["id_sessione"]
            sessione_attiva = id_sessione_corrente == st.session_state.get("id_sessione")
            etichetta = sessione["titolo"]

            with col1:
                if st.button(
                    etichetta,
                    key=f"sess_{id_sessione_corrente}",
                    use_container_width=True,
                    type="primary" if sessione_attiva else "secondary",
                    disabled=agent_in_esecuzione,
                ):
                    st.session_state.id_sessione = id_sessione_corrente
                    st.session_state.messaggi = carica_messaggi_sessione(db, id_sessione_corrente)
                    reimposta_stato_conversazione()
                    st.rerun()

            with col2:
                if st.button("X", key=f"del_{id_sessione_corrente}", disabled=agent_in_esecuzione):
                    crud.elimina_sessione(db, id_sessione_corrente)
                    if st.session_state.get("id_sessione") == id_sessione_corrente:
                        st.session_state.pop("id_sessione", None)
                        st.session_state.messaggi = []
                        reimposta_stato_conversazione()
                    st.rerun()

        st.divider()
        mostra_pagina_loggato()

    st.title("Chat")

    if "id_sessione" not in st.session_state:
        sessioni = crud.ottieni_sessioni_utente(db, id_utente)
        if sessioni:
            st.session_state.id_sessione = sessioni[0]["id_sessione"]
            st.session_state.messaggi = carica_messaggi_sessione(db, sessioni[0]["id_sessione"])
        else:
            st.session_state.id_sessione = crud.crea_sessione(db, id_utente, "Nuova sessione")
            st.session_state.messaggi = []

    if "messaggi" not in st.session_state:
        st.session_state.messaggi = carica_messaggi_sessione(db, st.session_state.id_sessione)

    for messaggio in st.session_state.messaggi:
        with st.chat_message(messaggio["ruolo"]):
            st.markdown(messaggio["contenuto"])
            mostra_feedback_messaggio(db, id_utente, messaggio)

    prompt = st.chat_input(
        "Che film vuoi vedere stasera?",
        disabled=agent_in_esecuzione,
        on_submit=avvia_elaborazione_agent,
        key="chat_prompt_input",
    )
    prompt_corrente = prompt_da_processare or prompt

    if prompt_corrente:
        storico_chat = costruisci_storico_chat(st.session_state.messaggi)
        sessione_vuota = len(st.session_state.messaggi) == 0
        output = None
        id_messaggio_assistente = None

        with st.chat_message("user"):
            st.markdown(prompt_corrente)

        id_messaggio_utente = crud.salva_messaggio(
            db,
            st.session_state.id_sessione,
            "user",
            prompt_corrente,
        )
        st.session_state.messaggi.append(
            {
                "id_messaggio": id_messaggio_utente,
                "ruolo": "user",
                "contenuto": prompt_corrente,
            }
        )

        try:
            # Fonte precisa SQLite: film gia' consigliati in questa sessione
            # e film gia' visti (feedback o dichiarati) non vanno mai riproposti.
            tmdb_id_da_evitare = sorted(set(
                crud.ottieni_tmdb_consigliati_sessione(db, st.session_state.id_sessione)
                + crud.ottieni_tmdb_visti(db, id_utente)
            ))
            with st.spinner("Ci sto pensando..."):
                output = ConduttoreDiAgents(
                    id_utente=id_utente,
                    id_sessione=st.session_state.id_sessione,
                    prompt=prompt_corrente,
                    storico_chat=storico_chat,
                    tmdb_id_da_evitare=tmdb_id_da_evitare,
                    richiesta_precedente=st.session_state.richiesta_precedente,
                    n_chiarimenti=st.session_state.n_chiarimenti,
                    n_raffinamenti=st.session_state.n_raffinamenti,
                )
            risposta_assistente = costruisci_risposta_assistente(output)
            film_consigliati = output.film_consigliati
            titolo_chat = output.titolo_chat

            # stato di sessione esplicito: l'ultima richiesta eseguita/parziale
            if output.richiesta_eseguita is not None:
                st.session_state.richiesta_precedente = output.richiesta_eseguita.model_dump()
            if output.azione == "chiarimento":
                st.session_state.n_chiarimenti += 1
                if output.log_notes.get("raffinamento_post_ricerca"):
                    st.session_state.n_raffinamenti += 1
            else:
                st.session_state.n_chiarimenti = 0
                st.session_state.n_raffinamenti = 0
        except Exception as e:
            risposta_assistente = costruisci_risposta_errore([f"Errore imprevisto: {type(e).__name__}: {e}"])
            film_consigliati = []
            titolo_chat = None

        try:
            id_messaggio_assistente = crud.salva_messaggio(
                db,
                st.session_state.id_sessione,
                "assistant",
                risposta_assistente,
            )

            if film_consigliati:
                salva_film_consigliati(
                    db=db,
                    id_messaggio=id_messaggio_assistente,
                    id_utente=id_utente,
                    film_consigliati=film_consigliati,
                )

            if sessione_vuota:
                titolo = titolo_chat or genera_titolo_fallback(prompt_corrente)
                crud.rinomina_sessione(db, st.session_state.id_sessione, titolo)
        except Exception as e:
            risposta_assistente = costruisci_risposta_errore([f"Risposta generata, ma salvataggio su database fallito: {e}"])

        with st.chat_message("assistant"):
            st.markdown(risposta_assistente)
            if output is not None:
                mostra_debug_errori(output)

        st.session_state.messaggi.append(
            {
                "id_messaggio": id_messaggio_assistente,
                "ruolo": "assistant",
                "contenuto": risposta_assistente,
                "film_consigliati": [
                    film.model_dump() if hasattr(film, "model_dump") else film
                    for film in (film_consigliati or [])
                ],
            }
        )

        st.session_state.agent_in_esecuzione = False
        st.session_state.pop("prompt_da_processare", None)
        st.rerun()

finally:
    db.close()
