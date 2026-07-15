"""ConduttoreDiAgents di MovieAgentOS.

Il router LLM e' il decisore primario: classifica l'intent, dialoga con
l'utente (sempre nel dominio film), decide quando chiedere chiarimenti e
compila la bozza di RichiestaRicerca. Python valida (generi TMDB reali,
numeri plausibili), fonde con lo stato di sessione (continuazioni e
raffinazioni) e applica il profilo utente. Nessuna pila di classificatori
regex: restano solo guardrail minimi di sicurezza.

Flusso dati strutturato: il conduttore passa al SearchAgent un oggetto
RichiestaRicerca, mai testo da ri-parsare. Le esclusioni precise (film gia'
visti o gia' consigliati) arrivano da SQLite come tmdb_id.
"""

import json
import time
from typing import Any

from agno.agent import Agent

from backend.agents.agente_memoria import (
    aggiorna_memoria_da_testo,
    cerca_nome_in_testo,
    costruisci_testo_contesto_memoria,
    estrai_memorie_semantiche,
    leggi_contesto_memoria,
)
from backend.agents.agente_ricerca import RicercaAgent
from backend.agents.linguaggio import (
    anni_negati,
    chiave_genere,
    estrai_anni,
    estrai_film_riferimento,
    estrai_generi_da_testo,
    estrai_generi_per_sentimento,
    estrai_voto_max,
    estrai_voto_min,
    filtra_generi_validi,
    generi_additivi,
    normalizza_titolo,
)
from backend.agents.llm_config import crea_model_agente, get_model_id, set_llm_api_key
from backend.agents.schemas import (
    FilmCandidato,
    ConduttoreOutput,
    DecisioneRouter,
    MemoryAgentOutput,
    RichiestaRicerca,
)
from backend.agents.text_utils import estrai_json_da_testo
from backend.db import crud
from backend.db.database import SessionLocal

MASSIMO_CHIARIMENTI = 1
# domande post-ricerca: si continua a restringere finche' i candidati validi
# sono piu' di 6 E ogni risposta fa davvero progressi (riduce il pool);
# questo numero e' solo una rete di sicurezza contro i cicli infiniti
MASSIMO_RAFFINAMENTI = 5

AZIONI_VALIDE = {
    "raccomandazione",
    "chiarimento",
    "conversazione_film",
    "domanda_memoria",
    "aggiornamento_memoria",
    "fuori_dominio",
    "errore",
}

TIPI_RICHIESTA_VALIDI = {"NUOVA", "CONTINUAZIONE", "RAFFINAZIONE"}
TIPI_RICERCA_VALIDI = {"genere", "tema", "simili", "titolo", "generica"}


def genera_titolo_chat(prompt: str) -> str:
    """Titolo sessione dai primi caratteri del primo messaggio."""
    if not prompt or not prompt.strip():
        return "Nuova conversazione"
    titolo = " ".join(prompt.strip().split()).strip(" \t\n\r.,;:!?-")
    if not titolo:
        return "Nuova conversazione"
    if len(titolo) > 42:
        titolo = titolo[:42].rstrip(" \t\n\r.,;:!?-") + "..."
    return titolo[0].upper() + titolo[1:]


def normalizza_errore_llm(errore: Any) -> str:
    testo = str(errore).lower()
    if "resource_exhausted" in testo or "quota" in testo or "429" in testo:
        return "Il servizio di intelligenza artificiale ha raggiunto il limite temporaneo di utilizzo. Riprova tra qualche minuto."
    if "api key" in testo or "openrouter_api_key" in testo:
        return "API key OpenRouter mancante o non valida."
    if "tmdb" in testo:
        return "Non riesco a completare la ricerca per un problema con TMDB. Controlla la chiave API o riprova tra poco."
    if "sqlite" in testo or "database" in testo or "no such table" in testo:
        return "Problema sul database locale: inizializza le tabelle e riprova."
    return "Errore durante la chiamata al modello LLM."


def crea_errore_conduttore_output(errore: str, azione: str = "errore",
                                  risposta: str = "Mi dispiace, non sono riuscito a completare la richiesta. Puoi riprovare o riformulare?",
                                  prompt: str = "", id_utente: int | None = None, id_sessione: int | None = None,
                                  log_notes: dict[str, Any] | None = None) -> ConduttoreOutput:
    return ConduttoreOutput(
        success=False,
        azione=azione,
        risposta=risposta,
        film_consigliati=[],
        memoria_aggiornata=False,
        titolo_chat=genera_titolo_chat(prompt) if prompt else None,
        errors=[errore],
        log_notes={
            "agent": "ConduttoreDiAgents",
            "id_utente": id_utente,
            "id_sessione": id_sessione,
            "prompt": prompt,
            **(log_notes or {}),
        },
    )


EMOJI_GENERI = {
    "horror": "👻", "azione": "💥", "commedia": "😂", "dramma": "🎭",
    "fantascienza": "🚀", "animazione": "🎨", "thriller": "🔍", "romance": "❤️",
    "avventura": "🗺️", "famiglia": "👨‍👩‍👧", "musica": "🎵", "crime": "🕵️",
    "mistero": "🔮", "fantasy": "🧙", "guerra": "⚔️", "documentario": "📽️",
    "storia": "📜", "western": "🤠",
}


def formatta_risposta_markdown_locale(film_consigliati: list) -> str:
    if not film_consigliati:
        return "🎬 Non ho trovato film adatti alla tua richiesta."

    def emoji_per_generi(generi: list[str]) -> str:
        for genere in generi:
            emoji = EMOJI_GENERI.get(genere.lower())
            if emoji:
                return emoji
        return "🎬"

    def stelle_voto(voto: float | None) -> str:
        if voto is None:
            return "N/D"
        return "⭐" * round(voto / 2) + f" **{voto:.1f}/10**"

    righe = ["## 🎬 Ecco cosa ti consiglio\n"]
    for i, film in enumerate(film_consigliati, start=1):
        anno = f" *({film.anno})*" if film.anno else ""
        generi = film.generi or []
        generi_testo = " · ".join(f"`{g}`" for g in generi[:4]) if generi else "_generi non disponibili_"
        descrizione = (film.descrizione or "Descrizione non disponibile.")[:300].rstrip(" .,") + "."
        motivo = film.motivo_ricerca or "corrisponde alla tua richiesta."
        righe.extend([
            "---",
            f"### {emoji_per_generi(generi)} {i}. **{film.titolo}**{anno}",
            f"> {descrizione}",
            "",
            f"🎭 **Generi:** {generi_testo}",
            f"⭐ **Voto TMDB:** {stelle_voto(film.voto_medio)}",
            f"💡 **Perché te lo consiglio:** {motivo}",
            "",
        ])
    righe.append("---")
    return "\n".join(righe)


def estrai_ultimi_scambi(storico_chat: list[dict[str, Any]] | None, n_scambi: int = 5) -> str:
    """Storico compatto per il contesto del router. I messaggi assistant con
    film vengono ridotti ai soli titoli (sezione strutturata dal frontend)."""
    if not storico_chat:
        return "Nessuna conversazione precedente."
    messaggi = list(storico_chat)[-(n_scambi * 2):]
    righe: list[str] = []
    for msg in messaggi:
        ruolo = (msg.get("role") or msg.get("ruolo") or "").lower()
        contenuto = (msg.get("content") or msg.get("contenuto") or "").strip()
        if not contenuto:
            continue
        if ruolo in ("assistant", "assistente") and "FILM_CONSIGLIATI_STRUTTURATI:" in contenuto:
            sezione = contenuto.split("FILM_CONSIGLIATI_STRUTTURATI:", 1)[1]
            titoli = [riga.strip(" -").split(" - ")[0] for riga in sezione.splitlines() if riga.strip(" -")]
            contenuto = "Film consigliati: " + ", ".join(t for t in titoli if t)[:300]
        else:
            contenuto = contenuto[:300]
        righe.append(f"[{ruolo}]: {contenuto}")
    return "\n".join(righe) if righe else "Nessuna conversazione precedente."
#istruzione del Routerr
ISTRUZIONI_ROUTER = [
    "Sei MovieAgentOS: un amico esperto di cinema. Tono caldo, naturale e conversazionale, sempre e solo nel dominio dei film.",
    "PROCEDURA: prima di tutto chiediti se il MESSAGGIO ATTUALE chiede consigli di film.",
    "Se NO — saluto ('ciao', 'come stai'), chiacchiera, opinione, preferenza, fatto personale, argomento non cinematografico — NON scegliere 'raccomandazione'.",
    "Un saluto merita un saluto: conversazione_film con una risposta breve e amichevole ('Ciao! Come va? Ti va di parlare di film o cerchi qualcosa da guardare?'), SENZA proporre film non richiesti.",
    "Un buon amico ascolta e aiuta quando gli viene chiesto: non spingere consigli a ogni messaggio.",
    "PARLA COME UNA PERSONA: reagisci a quello che l'utente dice (una battuta, un cambio di programma, una frustrazione) prima di passare al da farsi. Spiega in breve cosa stai facendo e perche'. Mai risposte telegrafiche o da modulo.",
    "Nelle risposte conversazionali (conversazione_film, domanda_memoria, fuori_dominio, chiarimento) scrivi 2-4 frasi piene: commenta quello che ha detto, aggiungi un pensiero tuo sul cinema quando ci sta, e solo poi la domanda o la conclusione. Una riga secca va bene solo per confermare una preferenza.",
    "Se il messaggio e' AMBIGUO e puo' significare due cose opposte (es. una negazione poco chiara come 'no voglio X', che puo' voler dire 'voglio X' oppure 'non voglio X'), NON tirare a indovinare: scegli 'chiarimento' e chiedi cosa intendeva.",
    "Analizza il messaggio dell'utente con lo storico e la memoria, e decidi cosa fare.",
    "Usa la CONVERSAZIONE_RECENTE per mantenere il filo del discorso: ricorda cosa e' gia' stato detto, non ripetere domande gia' fatte e collega la risposta a quello che l'utente ha raccontato.",
    "Rispondi SOLO con JSON valido, senza markdown, con questi campi:",
    '  azione: "raccomandazione" | "chiarimento" | "conversazione_film" | "domanda_memoria" | "aggiornamento_memoria" | "fuori_dominio"',
    '  tipo_richiesta: "NUOVA" | "CONTINUAZIONE" | "RAFFINAZIONE"',
    "  segna_visti_recenti: bool",
    "  aggiorna_memoria: bool",
    "  risposta_diretta: string",
    "  film_scelto: string (titolo ESATTO se l'utente dice di aver scelto o trovato uno dei film consigliati, altrimenti stringa vuota)",
    '  ricerca: {"tipo": "genere|tema|simili|titolo|generica", "generi_richiesti": [], "generi_esclusi": [], "voto_min": null, "voto_max": null, "anno_min": null, "anno_max": null, "film_base": "", "query_tema": ""}',
    "",
    "## AZIONI",
    "- raccomandazione: l'utente vuole consigli di film e la richiesta ha abbastanza dettagli (genere, tema, film di riferimento o vincoli). Compila 'ricerca'. In risposta_diretta scrivi UNA breve frase introduttiva amichevole e personale (es. collegata a cio' che ha chiesto): fara' da apertura alla lista dei film.",
    "- chiarimento: l'utente vuole consigli ma la richiesta e' troppo generica o ambigua per scegliere bene. In risposta_diretta scrivi TU la domanda per capire cosa preferisce. Compila comunque 'ricerca' con quello che sai gia'.",
    "- COME FARE LE DOMANDE: UNA sola domanda per turno, breve e naturale, come farebbe un amico in chat. Parti da cio' che sai gia' (memoria, conversazione, richiesta parziale) e chiedi la cosa PIU' IMPORTANTE che manca (di solito: genere o un film di riferimento). Non chiedere mai preferenze su attori, attrici o cast.",
    "  Le domande sono un'ECCEZIONE, non la regola: chiedile SOLO se senza quella risposta sceglieresti male. Se l'utente delega ('non so', 'scegli tu', 'qualcosa di popolare'), se ha gia' risposto a una domanda, o se la richiesta in corso e' gia' definita: NIENTE domande, procedi con i film.",
    "  Mai ripetere una domanda gia' fatta o chiedere cio' che si puo' dedurre. Mai piu' domande insieme, mai elenchi.",
    "- ACCUMULA le informazioni: se l'utente sta rispondendo a una tua domanda o aggiunge un dettaglio, NON fare un'altra domanda su cio' che sai gia' (guarda RICHIESTA_PRECEDENTE e CONVERSAZIONE_RECENTE). Quando hai genere/tema E almeno un altro criterio, passa a 'raccomandazione' con tipo_richiesta RAFFINAZIONE.",
    "- conversazione_film: chiacchiere, opinioni, spiegazioni su film, attori, registi, cinema ('cosa ne pensi di Nolan?', 'com'e' finito quel film?', 'come stai?'). In risposta_diretta scrivi una risposta conversazionale, utile e nel dominio del cinema.",
    "- Le domande SULLA CONVERSAZIONE ('quali film mi hai consigliato?', 'perche' li hai scelti?', 'com'era il secondo?') sono conversazione_film: rispondi usando ESCLUSIVAMENTE FILM_GIA_CONSIGLIATI_IN_QUESTA_SESSIONE (titoli e motivi esatti). MAI inventare o citare titoli che non sono in quella lista.",
    "- Se l'utente chiede QUALI film gli hai consigliato, la risposta DEVE contenere subito l'elenco esplicito dei titoli ('Ti ho consigliato: X, Y e Z'). Mai frasi come 'ecco i film che ti ho consigliato' senza scrivere i titoli.",
    "- La frase introduttiva della raccomandazione e' un'AFFERMAZIONE che presenta i film ('Perfetto, ho trovato dei drammi intensi che fanno per te'), MAI una domanda: le domande stanno solo nell'azione 'chiarimento'.",
    "- domanda_memoria: l'utente chiede cosa ricordi di lui (nome, gusti, preferenze). Rispondi in risposta_diretta usando la MEMORIA_UTENTE, in modo naturale.",
    "- aggiornamento_memoria: l'utente comunica SOLO una preferenza o un fatto personale ('adoro la fantascienza', 'il fantasy mi annoia', 'mi chiamo Sara'), senza chiedere film. Metti aggiorna_memoria=true e in risposta_diretta una breve conferma.",
    "- CONGEDO: se l'utente CONCLUDE (dice che ha trovato/scelto il film, ringrazia e chiude), azione conversazione_film con un saluto breve e caldo che augura buona visione: NIENTE domande, NIENTE altri film proposti. Se dice QUALE film ha scelto, riporta il titolo esatto in film_scelto.",
    "- fuori_dominio: la richiesta non riguarda il cinema (viaggi, mail, ricette, meteo, matematica). In risposta_diretta spiega gentilmente che aiuti solo con i film.",
    "- Se l'utente INSISTE su un argomento non cinematografico anche dopo una tua domanda ('pianifica il viaggio', 'dammi suggerimenti' riferito al viaggio), resta fuori_dominio: NON offrirti di aiutare con quello e NON cercare film. Puoi al massimo proporre un aggancio cinematografico ('non posso pianificare viaggi, pero' se vuoi ti consiglio film ambientati a Roma').",
    "- Se il messaggio e' incomprensibile o senza senso compiuto (lettere casuali, sigle, testo vuoto di significato), NON raccomandare film e NON inventare generi: usa azione 'chiarimento' e in risposta_diretta chiedi gentilmente cosa intendeva.",
    "- Non compilare 'ricerca' con generi che l'utente non ha chiesto ne' implicato chiaramente: nel dubbio, meglio 'chiarimento' che una ricerca inventata.",
    "- NON dedurre mai una raccomandazione dalla sola MEMORIA_UTENTE: serve una richiesta di film nel messaggio attuale. Un messaggio non cinematografico (viaggi, cibo, codice, mail) resta fuori_dominio anche se la memoria e' ricca di gusti.",
    "- La MEMORIA_UTENTE descrive i gusti PASSATI, e i gusti possono cambiare ogni giorno: quando fai una domanda di chiarimento puoi usarla per personalizzarla ('l'altra volta ti piacevano i thriller: ti va ancora, o oggi preferisci altro?'), mai per saltare la domanda.",
    "- Un messaggio che comunica solo un fatto personale ('mi chiamo X') NON e' una richiesta di film: azione aggiornamento_memoria.",
    "- Se l'utente delega la scelta ('non so', 'scegli tu', 'dimmi tu') dopo una tua domanda: tipo 'generica' basata sui gusti salvati, e dillo nella frase introduttiva.",
    "",
    "## tipo_richiesta",
    "- NUOVA: prima richiesta o tipo di film diverso da prima.",
    "- CONTINUAZIONE: vuole ALTRI film dello stesso tipo ('dammi altri', 'ancora', 'un altro').",
    "- RAFFINAZIONE: aggiunge solo vincoli alla richiesta precedente ('solo quelli con voto minimo 8', 'solo quelli nuovi', 'usciti dopo il 2015', 'meglio se italiani') senza cambiare tipo di film.",
    "- Usa RICHIESTA_PRECEDENTE e CONVERSAZIONE per decidere.",
    "",
    "## segna_visti_recenti",
    "true quando l'utente dice di aver GIA' VISTO i film appena consigliati ('li ho gia' visti', 'ho gia' guardato questi'). Combinalo con azione=raccomandazione e tipo_richiesta=CONTINUAZIONE se vuole altri consigli.",
    "",
    "## aggiorna_memoria",
    "true ogni volta che il messaggio contiene una preferenza stabile o un fatto personale da ricordare, anche insieme a una richiesta di film ('non mi piace l'horror, consigliami un thriller' -> aggiorna_memoria=true E azione=raccomandazione).",
    "true anche per le preferenze INDIRETTE ma chiare dal contesto: insoddisfazione sui film proposti ('non mi sono piaciuti'), entusiasmo per un tipo di film, richieste ripetute dello stesso genere.",
    "Quando l'utente CAMBIA IDEA o esprime insoddisfazione, riconoscilo con naturalezza nella risposta ('ok, cambiamo direzione!') e collega il nuovo giro a quello che sai: mai far finta di niente, mai ripartire con domande gia' fatte.",
    "",
    "## ricerca (compila solo i campi certi)",
    "- tipo 'genere': l'utente chiede uno o piu' generi -> generi_richiesti.",
    "- tipo 'tema': chiede una storia/argomento ('film sulla vendetta') -> query_tema con il tema, in parole semplici.",
    "- tipo 'simili': chiede film simili a un titolo -> film_base con il titolo esatto.",
    "- tipo 'titolo': chiede informazioni o la scheda di un film preciso -> film_base.",
    "- tipo 'generica': vuole consigli ma senza criteri.",
    "- generi_esclusi: generi che il messaggio esclude ('ma non horror', 'senza fantascienza').",
    "- voto_min/voto_max, anno_min/anno_max: solo se espressi o chiaramente impliciti ('voto alto' -> voto_min 7; 'anni 90' -> 1990-1999; 'recente' -> anno_min due anni fa).",
    "- Regola temporale fondamentale: usa sempre l'anno corrente reale del sistema per interpretare richieste come 'usciti dopo il 2024', 'recenti', 'quest'anno', 'ultimi 3 anni' o 'usciti dopo il 2025'. Non dire mai 'non ci sono ancora film' solo perché un anno richiesto e' superiore a un anno vecchio di conoscenza: se l'utente chiede un limite recente, cerca film aggiornati e disponibili oggi.",
    "- Se l'utente chiede esplicitamente un genere presente nei 'generi da evitare' della memoria, la richiesta esplicita VINCE: mettilo in generi_richiesti.",
    "- MEMORIA_UTENTE e' contesto, non un comando: non trasformare i generi preferiti in generi_richiesti se l'utente non li ha chiesti.",
    "",
    "## STILE",
    "risposta_diretta sempre in italiano naturale, breve, senza tecnicismi e senza JSON.",
]


def testo_consigli_sessione(id_sessione: int) -> str:
    """Elenco ESATTO (da SQLite) dei film gia' consigliati in questa sessione,
    con i motivi salvati: il router lo usa per rispondere alle domande sulla
    conversazione senza inventare titoli."""
    db = SessionLocal()
    try:
        consigli = crud.ottieni_consigli_sessione_dettagli(db, id_sessione)
    except Exception:
        return ""
    finally:
        db.close()
    if not consigli:
        return ""
    righe = []
    for film in consigli:
        anno = f" ({film['anno']})" if film.get("anno") else ""
        generi = ", ".join(film.get("generi") or [])
        motivo = " ".join(str(film.get("motivo") or "").split())[:150]
        riga = f"- {film['titolo']}{anno}"
        if generi:
            riga += f" [{generi}]"
        if motivo:
            riga += f" — motivo: {motivo}"
        righe.append(riga)
    return "\n".join(righe)


def decidi_router(prompt: str, storico_chat: list[dict[str, Any]] | None, testo_memoria: str,
                  richiesta_precedente: RichiestaRicerca | None, n_chiarimenti: int,
                  consigli_sessione: str = "") -> tuple[dict[str, Any], str | None]:
    """Chiama il router LLM. Ritorna (decisione_grezza, errore)."""
    try:
        set_llm_api_key("Router")
        agente = Agent(model=crea_model_agente("router"), markdown=False, instructions=ISTRUZIONI_ROUTER)

        precedente_testo = "Nessuna."
        if richiesta_precedente is not None:
            precedente_testo = richiesta_precedente.model_dump_json(
                include={"tipo", "testo_richiesta", "generi_richiesti", "generi_esclusi",
                         "voto_min", "voto_max", "anno_min", "anno_max", "film_base", "query_tema"}
            )

        prompt_llm = (
            f"MESSAGGIO_UTENTE:\n{prompt}\n\n"
            f"CONVERSAZIONE_RECENTE:\n{estrai_ultimi_scambi(storico_chat)}\n\n"
            f"MEMORIA_UTENTE:\n{testo_memoria or 'Nessuna memoria.'}\n\n"
            f"RICHIESTA_PRECEDENTE (stato di sessione):\n{precedente_testo}\n\n"
            f"FILM_GIA_CONSIGLIATI_IN_QUESTA_SESSIONE (dati esatti dal database, con i motivi):\n{consigli_sessione or 'Nessuno.'}\n\n"
            f"DOMANDE_DI_CHIARIMENTO_GIA_FATTE: {n_chiarimenti} (massimo {MASSIMO_CHIARIMENTI}: se il limite e' raggiunto non scegliere 'chiarimento')"
        )
        risposta = agente.run(prompt_llm)
        contenuto = risposta.content if isinstance(risposta.content, str) else str(risposta.content)
        try:
            return estrai_json_da_testo(contenuto), None
        except json.JSONDecodeError:
            # Il modello ha ignorato il formato e ha risposto in testo libero:
            # quel testo E' la sua risposta conversazionale, non un errore.
            # MAI mostrare all'utente testo che sembra JSON/codice.
            testo = contenuto.strip()
            sembra_codice = testo.startswith("{") or '"azione"' in testo or "azione:" in testo
            # se il messaggio dell'utente contiene generi o preferenze
            # estraibili, meglio agire (memoria/ricerca) che chiacchierare
            grezza_offline = decisione_fallback_offline(prompt)
            if grezza_offline.get("azione") in ("raccomandazione", "aggiornamento_memoria"):
                print("[Router]     risposta senza JSON ma il messaggio ha criteri chiari: uso gli estrattori")
                return grezza_offline, None
            if testo and len(testo) <= 1500 and not sembra_codice:
                print("[Router]     risposta senza JSON: la uso come risposta conversazionale")
                return {
                    "azione": "conversazione_film",
                    "tipo_richiesta": "NUOVA",
                    "risposta_diretta": testo,
                    "ricerca": {},
                }, None
            raise
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


def decisione_fallback_offline(prompt: str) -> dict[str, Any]:
    """Decisione minima senza LLM (solo quando il modello non risponde):
    usa gli estrattori deterministici condivisi, nessuna lista di frasi."""
    generi_pos, generi_neg = estrai_generi_per_sentimento(prompt)
    # generi citati senza marcatore di sentiment ("consigliami una commedia"):
    # in una richiesta contano come generi richiesti
    negati = {chiave_genere(g) for g in generi_neg}
    for genere in estrai_generi_da_testo(prompt):
        if chiave_genere(genere) not in negati and genere not in generi_pos:
            generi_pos.append(genere)
    voto_min = estrai_voto_min(prompt)
    voto_max = estrai_voto_max(prompt)
    anno_min, anno_max = estrai_anni(prompt)

    if generi_pos or voto_min is not None or anno_min is not None or anno_max is not None:
        return {
            "azione": "raccomandazione",
            "tipo_richiesta": "NUOVA",
            "aggiorna_memoria": bool(generi_pos or generi_neg),
            "risposta_diretta": "",
            "ricerca": {
                "tipo": "genere" if generi_pos else "generica",
                "generi_richiesti": generi_pos,
                "generi_esclusi": generi_neg,
                "voto_min": voto_min,
                "voto_max": voto_max,
                "anno_min": anno_min,
                "anno_max": anno_max,
            },
        }
    if generi_neg:
        return {
            "azione": "aggiornamento_memoria",
            "tipo_richiesta": "NUOVA",
            "aggiorna_memoria": True,
            "risposta_diretta": "Va bene, tengo conto di questa preferenza nei prossimi consigli.",
            "ricerca": {},
        }
    # nessun criterio estraibile: NON e' un errore da mostrare all'utente.
    # Si chiede cosa sta cercando (la domanda la genera genera_domanda_chiarimento,
    # che ha a sua volta un fallback deterministico se l'LLM resta giu'):
    # cosi' anche col modello indisponibile la conversazione resta in piedi
    # e il turno successivo puo' essere risolto dagli estrattori offline.
    return {
        "azione": "chiarimento",
        "tipo_richiesta": "NUOVA",
        "aggiorna_memoria": False,
        "risposta_diretta": "",
        "ricerca": {},
    }


def _come_float(valore: Any) -> float | None:
    if valore in (None, "", "null", "None"):
        return None
    try:
        return float(str(valore).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _come_int(valore: Any) -> int | None:
    if valore in (None, "", "null", "None"):
        return None
    try:
        return int(valore)
    except (TypeError, ValueError):
        return None


def valida_decisione(grezza: dict[str, Any], prompt: str) -> DecisioneRouter:
    """Valida l'output del router: azioni note, generi TMDB reali, numeri
    plausibili. Backstop deterministico sui vincoli espliciti del testo."""
    grezza = grezza or {}

    azione = str(grezza.get("azione") or "").strip()
    if azione not in AZIONI_VALIDE:
        risposta_llm = str(grezza.get("risposta_diretta") or "").strip()
        if risposta_llm and not risposta_llm.startswith("{"):
            # l'azione e' invalida ma il modello ha scritto una risposta
            # leggibile: meglio conversare che mostrare un errore
            azione = "conversazione_film"
        else:
            azione = "errore"

    tipo_richiesta = str(grezza.get("tipo_richiesta") or "NUOVA").strip().upper()
    if tipo_richiesta not in TIPI_RICHIESTA_VALIDI:
        tipo_richiesta = "NUOVA"

    ricerca_grezza = grezza.get("ricerca") or {}
    if not isinstance(ricerca_grezza, dict):
        ricerca_grezza = {}

    tipo_ricerca = str(ricerca_grezza.get("tipo") or "generica").strip().lower()
    if tipo_ricerca not in TIPI_RICERCA_VALIDI:
        tipo_ricerca = "generica"

    generi_richiesti = filtra_generi_validi(ricerca_grezza.get("generi_richiesti") or [])
    generi_esclusi = filtra_generi_validi(ricerca_grezza.get("generi_esclusi") or [])

    # backstop deterministico: i vincoli espliciti del testo non si perdono
    generi_pos_det, generi_neg_det = estrai_generi_per_sentimento(prompt)
    for genere in generi_neg_det:
        if chiave_genere(genere) not in {chiave_genere(g) for g in generi_esclusi}:
            generi_esclusi.append(genere)

    voto_min = estrai_voto_min(prompt)
    if voto_min is None:
        voto_min = _come_float(ricerca_grezza.get("voto_min"))
    voto_max = estrai_voto_max(prompt)
    if voto_max is None:
        voto_max = _come_float(ricerca_grezza.get("voto_max"))
    anno_min_det, anno_max_det = estrai_anni(prompt)
    anno_min = anno_min_det if anno_min_det is not None else _come_int(ricerca_grezza.get("anno_min"))
    anno_max = anno_max_det if anno_max_det is not None else _come_int(ricerca_grezza.get("anno_max"))

    # se l'LLM non ha compilato i generi ma il testo li contiene, recuperali
    # (generi citati nel messaggio, esclusi quelli negati)
    if not generi_richiesti:
        negati_det = {chiave_genere(g) for g in generi_neg_det}
        generi_richiesti = [
            g for g in estrai_generi_da_testo(prompt)
            if chiave_genere(g) not in negati_det
        ]

    # un genere escluso non puo' essere anche richiesto (la richiesta esplicita
    # dell'utente, se presente nel prompt, vince sull'esclusione)
    richiesti_espliciti = {chiave_genere(g) for g in generi_pos_det}
    generi_esclusi = [g for g in generi_esclusi if chiave_genere(g) not in richiesti_espliciti]
    esclusi_chiavi = {chiave_genere(g) for g in generi_esclusi}
    generi_richiesti = [g for g in generi_richiesti if chiave_genere(g) not in esclusi_chiavi]

    film_base = str(ricerca_grezza.get("film_base") or "").strip()
    query_tema = str(ricerca_grezza.get("query_tema") or "").strip()

    # backstop: 'simile a X' nel testo -> ricerca per similarita', anche se
    # il router non ha compilato film_base
    if not film_base:
        riferimento = estrai_film_riferimento(prompt)
        if riferimento:
            film_base = riferimento
            if tipo_ricerca not in ("simili", "titolo"):
                tipo_ricerca = "simili"

    # coerenza del tipo con i dati disponibili
    if tipo_ricerca == "simili" and not film_base:
        tipo_ricerca = "genere" if generi_richiesti else "generica"
    if tipo_ricerca == "titolo" and not film_base:
        tipo_ricerca = "generica"
    if tipo_ricerca == "tema" and not query_tema:
        tipo_ricerca = "genere" if generi_richiesti else "generica"
    if tipo_ricerca in ("generica", "genere") and not generi_richiesti and query_tema:
        tipo_ricerca = "tema"
    if tipo_ricerca == "generica" and generi_richiesti:
        tipo_ricerca = "genere"
    if tipo_ricerca == "genere" and not generi_richiesti:
        tipo_ricerca = "generica"

    ricerca = RichiestaRicerca(
        tipo=tipo_ricerca,
        testo_richiesta=prompt.strip(),
        generi_richiesti=generi_richiesti,
        generi_esclusi=generi_esclusi,
        voto_min=voto_min,
        voto_max=voto_max,
        anno_min=anno_min,
        anno_max=anno_max,
        film_base=film_base,
        query_tema=query_tema,
    )

    return DecisioneRouter(
        azione=azione,
        tipo_richiesta=tipo_richiesta,
        segna_visti_recenti=bool(grezza.get("segna_visti_recenti")),
        aggiorna_memoria=bool(grezza.get("aggiorna_memoria")),
        risposta_diretta=str(grezza.get("risposta_diretta") or "").strip(),
        film_scelto=str(grezza.get("film_scelto") or "").strip(),
        ricerca=ricerca,
    )


def _aggiunge_informazioni(ricerca: RichiestaRicerca, precedente: RichiestaRicerca,
                           generi_nel_testo: list[str], prompt: str) -> bool:
    """Vero se il messaggio porta qualcosa di NUOVO rispetto alla richiesta in
    corso. Una 'raffinazione' che non aggiunge nulla (vincoli solo copiati
    dall'LLM) di solito e' un messaggio che non riguarda la ricerca."""
    if generi_nel_testo:
        return True
    if anni_negati(prompt):
        return True  # chiede di RIMUOVERE il vincolo anno: e' un'informazione
    if ricerca.film_base and ricerca.film_base != precedente.film_base:
        return True
    if ricerca.query_tema and ricerca.query_tema != precedente.query_tema:
        return True
    for campo in ("voto_min", "voto_max", "anno_min", "anno_max"):
        valore = getattr(ricerca, campo)
        if valore is not None and valore != getattr(precedente, campo):
            return True
    esclusi_prec = {chiave_genere(g) for g in precedente.generi_esclusi}
    if any(chiave_genere(g) not in esclusi_prec for g in ricerca.generi_esclusi):
        return True
    return False


def ultimo_messaggio_assistente(storico_chat: list[dict[str, Any]] | None) -> str:
    for msg in reversed(storico_chat or []):
        ruolo = (msg.get("role") or msg.get("ruolo") or "").lower()
        if ruolo in ("assistant", "assistente"):
            return str(msg.get("content") or msg.get("contenuto") or "")
    return ""


def correggi_decisione(decisione: DecisioneRouter, prompt: str,
                       richiesta_precedente: RichiestaRicerca | None,
                       n_chiarimenti: int = 0,
                       msg_assistente: str = "") -> DecisioneRouter:
    """Correzioni deterministiche generali sulla decisione del router.

    1. Un messaggio che aggiunge SOLO vincoli (voto/anno/esclusioni) mentre
       esiste una richiesta precedente e' una RAFFINAZIONE, non una NUOVA
       ricerca: altrimenti i vincoli si applicherebbero a una ricerca vuota.
    2. Un messaggio che comunica solo il proprio nome (nessun genere, tema,
       film o vincolo nel testo) non e' una richiesta di film: aggiorna la
       memoria e risponde, senza cercare.
    """
    # preferenze ESPLICITE nel messaggio ('mi piacciono X, odio Y'): la memoria
    # va aggiornata sempre, qualunque cosa abbia deciso il router
    generi_pos_det, generi_neg_det = estrai_generi_per_sentimento(prompt)
    if (generi_pos_det or generi_neg_det) and not decisione.aggiorna_memoria:
        decisione.aggiorna_memoria = True
        print("[Conduttore] preferenze esplicite nel messaggio -> aggiorno la memoria (il router non l'aveva chiesto)")

    # il messaggio comunica SOLO una preferenza da evitare ("per il futuro non
    # mi piace l'horror"): nessun genere voluto, nessun vincolo, nessun film
    # di riferimento, e non e' ne' la risposta a una mia domanda ne' un
    # "dammi altri". E' informazione per la memoria, non una richiesta.
    if (
        n_chiarimenti == 0
        and generi_neg_det and not generi_pos_det
        and decisione.azione in ("raccomandazione", "chiarimento")
        and decisione.tipo_richiesta != "CONTINUAZIONE"
        and not decisione.segna_visti_recenti
    ):
        negati = {chiave_genere(g) for g in generi_neg_det}
        altri_generi = [g for g in estrai_generi_da_testo(prompt) if chiave_genere(g) not in negati]
        vincoli_nel_messaggio = (
            estrai_voto_min(prompt) is not None
            or estrai_voto_max(prompt) is not None
            or estrai_anni(prompt) != (None, None)
        )
        if not altri_generi and not vincoli_nel_messaggio and not estrai_film_riferimento(prompt):
            decisione.azione = "aggiornamento_memoria"
            decisione.aggiorna_memoria = True
            if not decisione.risposta_diretta or "?" in decisione.risposta_diretta:
                decisione.risposta_diretta = ""
            print("[Conduttore] il messaggio comunica solo una preferenza da evitare -> aggiorno la memoria, nessuna ricerca")
            return decisione

    if decisione.ricerca is None or decisione.azione not in ("raccomandazione", "chiarimento"):
        return decisione
    ricerca = decisione.ricerca

    generi_nel_testo = estrai_generi_da_testo(prompt)
    argomento_nel_testo = bool(generi_nel_testo or ricerca.film_base or ricerca.query_tema)
    vincoli_numerici = (
        ricerca.voto_min is not None or ricerca.voto_max is not None
        or ricerca.anno_min is not None or ricerca.anno_max is not None
    )

    # L'utente ha gia' una richiesta in corso e il messaggio porta informazioni
    # NUOVE rispetto allo stato (confronto deterministico: i campi copiati
    # dall'LLM dal profilo/stato non contano): si raccoglie e si CERCA,
    # invece di fare un'altra domanda.
    if decisione.azione == "chiarimento" and richiesta_precedente is not None:
        if _aggiunge_informazioni(ricerca, richiesta_precedente, generi_nel_testo, prompt):
            decisione.azione = "raccomandazione"
            decisione.tipo_richiesta = "RAFFINAZIONE"
            print("[Conduttore] correzione: la risposta porta nuove informazioni -> uso quelle raccolte e cerco (niente altra domanda)")
            return decisione
        if richiesta_precedente.ha_vincoli_specifici():
            # la ricerca in corso e' gia' definita: un'altra domanda e' inutile,
            # si continua con i criteri raccolti
            decisione.azione = "raccomandazione"
            decisione.tipo_richiesta = "CONTINUAZIONE"
            print("[Conduttore] correzione: richiesta già definita -> continuo con i criteri raccolti invece di fare altre domande")
            return decisione

    if decisione.azione != "raccomandazione":
        return decisione

    # una RAFFINAZIONE che non aggiunge NULLA di nuovo non e' una raffinazione:
    # quasi sempre il messaggio non riguarda la ricerca (es. e' fuori tema).
    # Meglio chiedere cosa intende che rifare la stessa ricerca.
    if (
        decisione.tipo_richiesta == "RAFFINAZIONE"
        and richiesta_precedente is not None
        and not _aggiunge_informazioni(ricerca, richiesta_precedente, generi_nel_testo, prompt)
    ):
        decisione.azione = "chiarimento"
        decisione.risposta_diretta = ""  # la domanda la genera l'LLM sul messaggio reale
        print("[Conduttore] correzione: 'raffinazione' senza nuove informazioni -> chiedo cosa intende")
        return decisione

    # cambio di argomento: il testo introduce un genere NUOVO, diverso da
    # quelli della richiesta in corso -> e' una richiesta NUOVA, non una
    # raffinazione (i vincoli vecchi non devono trascinarsi dietro)
    cambio_argomento = False
    if richiesta_precedente is not None and generi_nel_testo:
        vecchi = {chiave_genere(g) for g in richiesta_precedente.generi_richiesti}
        # i generi NEGATI nel messaggio ("senza horror") sono esclusioni, non
        # un nuovo argomento: non contano come cambio. Se pero' il router li
        # ha letti come RICHIESTI ("anzi no, un horror"), l'ambiguita' della
        # negazione e' gia' stata sciolta a favore della richiesta e contano.
        richiesti_router = {chiave_genere(g) for g in ricerca.generi_richiesti}
        solo_esclusi = {chiave_genere(g) for g in generi_neg_det} - richiesti_router
        nuovi = {chiave_genere(g) for g in generi_nel_testo} - solo_esclusi
        cambio_argomento = bool(vecchi) and bool(nuovi) and not nuovi.intersection(vecchi)
        # generi ADDITIVI: o il linguaggio e' esplicitamente additivo
        # ("mi piace ANCHE il dramma") o il genere era stato PROPOSTO dalla
        # mia ultima domanda ("ti va anche il romance?" -> "si va bene").
        # In entrambi i casi si AGGIUNGE alla richiesta in corso.
        if cambio_argomento:
            additivi = {chiave_genere(g) for g in generi_additivi(prompt)}
            offerti = {chiave_genere(g) for g in estrai_generi_da_testo(msg_assistente)} if msg_assistente else set()
            if nuovi and nuovi.issubset(additivi | offerti):
                cambio_argomento = False
                ricerca.generi_richiesti = filtra_generi_validi(
                    list(richiesta_precedente.generi_richiesti) + list(ricerca.generi_richiesti or generi_nel_testo)
                )
                if decisione.tipo_richiesta == "NUOVA":
                    decisione.tipo_richiesta = "RAFFINAZIONE"
                print("[Conduttore] correzione: genere aggiunto ('anche X' / proposto dalla mia domanda) -> si somma alla richiesta in corso, non cambio argomento")

    if decisione.tipo_richiesta in ("RAFFINAZIONE", "CONTINUAZIONE") and cambio_argomento:
        decisione.tipo_richiesta = "NUOVA"
        print("[Conduttore] correzione: genere diverso nel messaggio -> richiesta NUOVA (cambio argomento)")

    # la risposta a una domanda di chiarimento raffina la richiesta in corso,
    # non riparte da zero (a meno di un vero cambio di argomento)
    if (
        decisione.tipo_richiesta == "NUOVA"
        and n_chiarimenti >= 1
        and richiesta_precedente is not None
        and not cambio_argomento
        and not decisione.ricerca.film_base
    ):
        decisione.tipo_richiesta = "RAFFINAZIONE"
        print("[Conduttore] correzione: risposta alla domanda -> raffina la richiesta in corso, non riparte da zero")

    if (
        decisione.tipo_richiesta == "NUOVA"
        and richiesta_precedente is not None
        and not argomento_nel_testo
        and (vincoli_numerici or ricerca.generi_esclusi)
    ):
        decisione.tipo_richiesta = "RAFFINAZIONE"
        print("[Conduttore] correzione: il messaggio aggiunge solo vincoli -> RAFFINAZIONE della richiesta precedente")
        return decisione

    # generi proposti in blocco (>=2) senza alcuna traccia nel messaggio e
    # senza altri vincoli: e' il profilo copiato, non una richiesta dell'utente.
    # La ricerca torna generica, cosi' scatta il chiarimento invece di una
    # lista non richiesta. (Una richiesta implicita reale, tipo 'voglio
    # ridere' -> Commedia, produce UN genere e non viene toccata.)
    if (
        decisione.tipo_richiesta == "NUOVA"
        and len(ricerca.generi_richiesti) >= 2
        and not generi_nel_testo
        and not ricerca.film_base
        and not ricerca.query_tema
        and not vincoli_numerici
    ):
        ricerca.generi_richiesti = []
        ricerca.tipo = "generica"
        print("[Conduttore] correzione: generi non chiesti dall'utente (copiati dal profilo) -> richiesta generica, serve chiarimento")

    nome = cerca_nome_in_testo(prompt)
    if nome and not argomento_nel_testo and not vincoli_numerici and decisione.tipo_richiesta == "NUOVA":
        decisione.azione = "aggiornamento_memoria"
        decisione.aggiorna_memoria = True
        if not decisione.risposta_diretta:
            decisione.risposta_diretta = f"Piacere, {nome}! Me lo ricorderò."
        print("[Conduttore] correzione: solo un fatto personale (nome) -> aggiornamento_memoria, nessuna ricerca")
    return decisione


def fondi_con_precedente(nuova: RichiestaRicerca, precedente: RichiestaRicerca | None,
                         tipo_richiesta: str) -> RichiestaRicerca:
    """Stato di sessione esplicito: CONTINUAZIONE = stessa richiesta, pagina
    successiva; RAFFINAZIONE = stessa richiesta + vincoli aggiunti."""
    if precedente is None or tipo_richiesta == "NUOVA":
        nuova.pagina = 1
        return nuova

    base = precedente.model_copy(deep=True)
    # il testo si ACCUMULA: il rerank deve vedere tutta la richiesta raccolta
    # ("horror con suspense | dammi altri"), non solo l'ultimo messaggio
    if nuova.testo_richiesta and nuova.testo_richiesta not in base.testo_richiesta:
        base.testo_richiesta = f"{base.testo_richiesta} | {nuova.testo_richiesta}"[-500:].lstrip("| ")

    # vincoli nuovi si sommano/sovrascrivono quelli della richiesta base
    if nuova.voto_min is not None:
        base.voto_min = nuova.voto_min
    if nuova.voto_max is not None:
        base.voto_max = nuova.voto_max
    if nuova.anno_min is not None:
        base.anno_min = nuova.anno_min
    if nuova.anno_max is not None:
        base.anno_max = nuova.anno_max

    esclusi_chiavi = {chiave_genere(g) for g in base.generi_esclusi}
    for genere in nuova.generi_esclusi:
        if chiave_genere(genere) not in esclusi_chiavi:
            base.generi_esclusi.append(genere)

    if nuova.generi_richiesti:
        base.generi_richiesti = nuova.generi_richiesti
        if base.tipo == "generica":
            base.tipo = "genere"
    if nuova.film_base:
        base.film_base = nuova.film_base
        base.tipo = "simili"
    if nuova.query_tema:
        base.query_tema = nuova.query_tema
        if base.tipo in ("generica", "genere") and not base.generi_richiesti:
            base.tipo = "tema"

    base.pagina = base.pagina + 1 if tipo_richiesta == "CONTINUAZIONE" else 1
    return base


def applica_profilo(richiesta: RichiestaRicerca, memoria_output: MemoryAgentOutput | None) -> RichiestaRicerca:
    """Profilo utente sulla richiesta:
    - generi_da_evitare = vincolo forte, MA sospeso per i generi chiesti
      esplicitamente ora; mai applicato alle richieste 'simili';
    - generi preferiti e segnali = preferenze morbide, solo per il rerank."""
    if not memoria_output or not memoria_output.success:
        return richiesta

    richiesta.profilo_generi_preferiti = list(memoria_output.generi_preferiti or [])
    richiesta.profilo_segnali = {
        str(k): float(v) for k, v in (memoria_output.preferenze_json or {}).items()
        if isinstance(v, (int, float))
    }

    if richiesta.tipo != "simili":
        richiesti = {chiave_genere(g) for g in richiesta.generi_richiesti}
        esclusi = {chiave_genere(g) for g in richiesta.generi_esclusi}
        for genere in memoria_output.generi_da_evitare or []:
            chiave = chiave_genere(genere)
            if chiave and chiave not in richiesti and chiave not in esclusi:
                richiesta.generi_esclusi.append(genere)

    return richiesta


def applica_gusti_feedback(richiesta: RichiestaRicerca, id_utente: int) -> RichiestaRicerca:
    """Feedback thumbs da SQLite (fonte precisa): film piaciuti/non piaciuti
    e i loro generi diventano un criterio importante della SCELTA (punteggio
    deterministico + rerank LLM). L'esclusione dura dei film gia' visti
    viaggia gia' come tmdb_id."""
    db = SessionLocal()
    try:
        gusti = crud.ottieni_gusti_film_utente(db, id_utente)
    except Exception:
        return richiesta
    finally:
        db.close()

    richiesta.film_piaciuti = [f["titolo"] for f in gusti["piaciuti"]][:10]
    richiesta.film_non_piaciuti = [f["titolo"] for f in gusti["non_piaciuti"]][:10]

    generi_graditi: list[str] = []
    generi_sgraditi: list[str] = []
    for film in gusti["piaciuti"]:
        generi_graditi.extend(film.get("generi") or [])
    for film in gusti["non_piaciuti"]:
        generi_sgraditi.extend(film.get("generi") or [])
    richiesta.generi_graditi_feedback = filtra_generi_validi(generi_graditi)
    richiesta.generi_sgraditi_feedback = filtra_generi_validi(generi_sgraditi)
    return richiesta


def genera_domanda_chiarimento(prompt: str, testo_memoria: str) -> str:
    """Domanda di chiarimento generata dall'LLM (mai hardcoded, se il modello
    risponde). Fallback statico solo se l'LLM non e' raggiungibile."""
    try:
        set_llm_api_key("Chiarimento")
        agente = Agent(
            model=crea_model_agente("chiarimento"),
            markdown=False,
            instructions=[
                "Sei un amico esperto di cinema. Il messaggio dell'utente non basta per scegliere bene un film.",
                "Se il messaggio riguarda i film: scrivi UNA sola domanda breve e naturale, come in una chat tra amici, sulla cosa PIU' IMPORTANTE che manca (di solito genere o un film di riferimento; il resto si chiedera' nei turni successivi).",
                "Parti da quello che sai gia' dell'utente: mai chiedere cio' che ha gia' detto. La domanda deve restringere davvero la ricerca.",
                "I gusti salvati sono il passato: se rilevanti, chiedi conferma che valgano ancora oggi.",
                "Se il messaggio NON sembra riguardare i film (matematica, viaggi, altro): dillo con gentilezza (ti occupi solo di cinema) e chiedi cosa gli va di guardare.",
                "Rispondi solo con quella frase, nient'altro: niente elenchi.",
            ],
        )
        risposta = agente.run(
            f"Richiesta dell'utente: {prompt}\n\nQuello che so gia' di lui:\n{testo_memoria or 'Niente.'}"
        )
        if isinstance(risposta.content, str) and risposta.content.strip():
            return risposta.content.strip()
    except Exception:
        pass
    return "Che tipo di film ti andrebbe? Puoi dirmi un genere, un tono o un film che ti è piaciuto."


def statistiche_pool(pool: list[FilmCandidato]) -> str:
    """Descrive in modo compatto come si distribuiscono i candidati REALI
    (anni, generi, voti): serve al generatore di domande per scegliere un
    angolo che divide davvero il pool, non una dimensione su cui i film
    sono tutti uguali."""
    if not pool:
        return "nessun dato"
    righe: list[str] = [f"totale: {len(pool)} film"]

    anni = sorted(int(str(f.anno)[:4]) for f in pool if str(f.anno)[:4].isdigit())
    if anni:
        meta = anni[len(anni) // 2]
        righe.append(f"anni: dal {anni[0]} al {anni[-1]} (meta' prima e meta' dopo il {meta})")

    conteggio_generi: dict[str, int] = {}
    for film in pool:
        for genere in film.generi:
            conteggio_generi[genere] = conteggio_generi.get(genere, 0) + 1
    if conteggio_generi:
        top = sorted(conteggio_generi.items(), key=lambda kv: kv[1], reverse=True)[:6]
        righe.append("generi presenti: " + ", ".join(f"{g} ({n})" for g, n in top))

    voti = sorted(float(f.voto_medio) for f in pool if isinstance(f.voto_medio, (int, float)))
    if voti:
        sopra_75 = sum(1 for v in voti if v >= 7.5)
        righe.append(f"voti: da {voti[0]:.1f} a {voti[-1]:.1f} ({sopra_75} film con voto >= 7.5)")

    return "\n".join(righe)


def angoli_liberi_richiesta(richiesta: RichiestaRicerca, pool: list[FilmCandidato]) -> list[str]:
    """Dimensioni su cui una domanda puo' ANCORA restringere: quelle gia'
    decise dall'utente (periodo, voto, generi) non vanno mai richieste.
    Se non resta nessun angolo libero, non c'e' niente da chiedere."""
    angoli: list[str] = []
    if richiesta.anno_min is None and richiesta.anno_max is None:
        angoli.append("periodo o anno di uscita")
    if richiesta.voto_min is None and richiesta.voto_max is None:
        angoli.append("voto minimo")
    decisi = {chiave_genere(g) for g in richiesta.generi_richiesti + richiesta.generi_esclusi}
    conteggio: dict[str, int] = {}
    for film in pool:
        for genere in film.generi:
            if chiave_genere(genere) not in decisi:
                conteggio[genere] = conteggio.get(genere, 0) + 1
    secondari = [g for g, n in sorted(conteggio.items(), key=lambda kv: kv[1], reverse=True)
                 if n >= 3][:4]
    if secondari:
        angoli.append("genere secondario presente nel pool: " + ", ".join(secondari))
    return angoli


def genera_domanda_raffinamento(prompt: str, richiesta: RichiestaRicerca,
                                pool: list[FilmCandidato], esempi_titoli: str,
                                angoli_liberi: list[str] | None = None,
                                scambi_recenti: str = "",
                                pool_precedente: int = 0,
                                n_raffinamenti: int = 0) -> str:
    """Troppi candidati validi: l'LLM formula UNA domanda per restringere,
    scegliendo l'angolo sulle STATISTICHE reali del pool, e la cui risposta
    si traduca in un filtro applicabile. Nessuna domanda hardcoded: se l'LLM
    non risponde si mostrano direttamente i risultati."""
    try:
        set_llm_api_key("Raffinamento")
        agente = Agent(
            model=crea_model_agente("raffinamento"),
            markdown=False,
            instructions=[
                "Sei un amico esperto di cinema. La ricerca ha trovato TROPPI film adatti: serve UNA domanda per restringere la scelta.",
                "Ti passo le STATISTICHE del pool (anni, generi, voti): scegli l'angolo che DIVIDE DAVVERO questi film. Se sono tutti dello stesso periodo non chiedere il periodo; se hanno generi misti, il genere e' un ottimo angolo.",
                "Un angolo spesso ottimo e' il GENERE SECONDARIO: se molti film del pool hanno anche un altro genere oltre a quello chiesto (es. ha chiesto horror e meta' sono anche romance), chiedi se quel genere secondario gli va o se preferisce toglierlo.",
                "La risposta dell'utente deve potersi tradurre in un FILTRO CONCRETO: un periodo o un anno, uno dei generi presenti nel pool, un voto minimo, o un tema preciso. Offri 2-3 alternative concrete prese dalle statistiche (es. 'piu' romance o piu' dramma?', 'usciti dopo il 2020 o anche i classici?').",
                "MAI domande vaghe le cui risposte non filtrano nulla (es. 'epico o coinvolgente?', 'classico o innovativo?' senza legame con anni o generi reali del pool).",
                "Scegli l'angolo SOLO tra gli ANGOLI LIBERI che ti passo: le dimensioni gia' decise dall'utente (periodo, voto, generi nei vincoli attuali) NON vanno MAI richieste.",
                "Guarda la CONVERSAZIONE RECENTE: una domanda gia' fatta (anche con parole diverse) non si ripete MAI. Se l'utente ha gia' risposto a un angolo, quell'angolo e' chiuso.",
                "Una domanda per turno, come in una chat tra amici: se servira' altro, lo chiederai al turno successivo.",
                "I gusti salvati descrivono il passato e possono cambiare: se sono rilevanti, chiedi conferma che valgano ancora oggi invece di darli per scontati.",
                "APERTURA: naturale e SEMPRE DIVERSA dalle aperture gia' usate nella conversazione recente. Alla PRIMA domanda di' quanti film adatti hai trovato e che li tieni da parte. Nei turni successivi NON ripetere 'ho trovato X film': commenta il progresso guardando i numeri che ti passo (es. 'bene, siamo scesi da 34 a 15', 'ottimo, la rosa si sta stringendo'). POI fai la domanda.",
                "Se gli angoli utili sono quasi esauriti, invece della domanda su un filtro puoi chiedere apertamente se c'e' qualcos'altro che vuole aggiungere prima dei consigli.",
                "Rispondi solo con quella frase, nient'altro: niente elenchi di film e niente JSON.",
            ],
        )
        gusti_parti = []
        if richiesta.profilo_generi_preferiti:
            gusti_parti.append("generi preferiti in passato: " + ", ".join(richiesta.profilo_generi_preferiti))
        if richiesta.film_piaciuti:
            gusti_parti.append("film piaciuti: " + ", ".join(richiesta.film_piaciuti[:5]))
        if richiesta.film_non_piaciuti:
            gusti_parti.append("film non piaciuti: " + ", ".join(richiesta.film_non_piaciuti[:5]))
        risposta = agente.run(
            f"Richiesta dell'utente: {prompt}\n"
            f"Vincoli attuali: {richiesta.model_dump_json(include={'tipo', 'generi_richiesti', 'generi_esclusi', 'voto_min', 'voto_max', 'anno_min', 'anno_max', 'query_tema'})}\n"
            f"Gusti salvati (passato, da confermare): {'; '.join(gusti_parti) or 'nessuno'}\n"
            f"STATISTICHE DEL POOL:\n{statistiche_pool(pool)}\n"
            f"ANGOLI LIBERI (scegline UNO): {'; '.join(angoli_liberi or []) or 'nessuno'}\n"
            f"PROGRESSO: {'prima ' + str(pool_precedente) + ' candidati, ora ' + str(len(pool)) if pool_precedente else 'prima domanda della sequenza: ' + str(len(pool)) + ' candidati'} | domande di filtro gia' fatte: {n_raffinamenti}\n"
            f"CONVERSAZIONE RECENTE (domande gia' fatte: non ripeterle):\n{scambi_recenti or 'nessuna'}\n"
            f"Alcuni esempi: {esempi_titoli}"
        )
        if isinstance(risposta.content, str) and risposta.content.strip():
            return risposta.content.strip()
    except Exception as e:
        print(f"[Raffinamento] errore LLM: {type(e).__name__}: {str(e)[:80]}")
    return ""



def segna_film_scelto(id_utente: int, id_sessione: int, titolo_scelto: str) -> str | None:
    """L'utente dice di aver SCELTO un film tra quelli consigliati ("ho trovato
    il film giusto, X"): si salva in SQLite come visto e gradito (dato preciso:
    non verra' piu' riproposto e contera' nei gusti)."""
    chiave = normalizza_titolo(titolo_scelto)
    if not chiave:
        return None
    db = SessionLocal()
    try:
        for film in crud.ottieni_consigli_sessione_dettagli(db, id_sessione):
            titolo = str(film.get("titolo") or "")
            norm = normalizza_titolo(titolo)
            if norm and (norm == chiave or chiave in norm or norm in chiave):
                if film.get("id_tmdb"):
                    crud.segna_film_visto(
                        db, id_utente=id_utente, id_tmdb=int(film["id_tmdb"]),
                        titolo=titolo, fonte="dichiarato", gradito=True,
                    )
                    print(f"[Visti]      film scelto dall'utente: '{titolo}' -> salvato in SQLite come visto e gradito OK")
                return titolo
        print(f"[Visti]      film scelto '{titolo_scelto}' non trovato tra i consigli della sessione: nessun salvataggio")
        return None
    except Exception as e:
        print(f"[Visti]      errore salvataggio film scelto: {type(e).__name__}: {e}")
        return None
    finally:
        db.close()


def segna_ultimi_consigli_come_visti(id_utente: int, id_sessione: int) -> list[int]:
    """"Ho gia' visto questi": marca in SQLite i film dell'ultimo consiglio
    come visti (fonte precisa) e ritorna i loro tmdb_id da escludere subito."""
    db = SessionLocal()
    try:
        ultimi = crud.ottieni_ultimi_film_consigliati_sessione(db, id_sessione)
        tmdb_ids: list[int] = []
        for film in ultimi:
            id_tmdb = film.get("id_tmdb")
            if not id_tmdb:
                continue
            crud.segna_film_visto(
                db, id_utente=id_utente, id_tmdb=int(id_tmdb),
                titolo=film.get("titolo"), fonte="dichiarato",
            )
            tmdb_ids.append(int(id_tmdb))
        if tmdb_ids:
            print(f"[Visti]      segnati {len(tmdb_ids)} film come gia' visti (dichiarato)")
        return tmdb_ids
    except Exception as e:
        print(f"[Visti]      errore salvataggio visti: {type(e).__name__}: {e}")
        return []
    finally:
        db.close()


def ConduttoreDiAgents(id_utente: int, id_sessione: int, prompt: str = "",
                       storico_chat: list[dict[str, Any]] | None = None,
                       tmdb_id_da_evitare: list[int] | None = None,
                       richiesta_precedente: dict[str, Any] | RichiestaRicerca | None = None,
                       n_chiarimenti: int = 0, n_raffinamenti: int = 0) -> ConduttoreOutput:
    start = time.perf_counter()
    prompt_pulito = (prompt or "").strip()
    titolo_chat = genera_titolo_chat(prompt_pulito) if prompt_pulito else None
    print(f"\n{'=' * 60}")
    print(f'[Conduttore] "{prompt_pulito[:80]}" (utente #{id_utente}, sessione #{id_sessione}) | modello LLM: {get_model_id()}')

    log_notes: dict[str, Any] = {
        "agent": "ConduttoreDiAgents",
        "id_utente": id_utente,
        "id_sessione": id_sessione,
        "richiesta": prompt_pulito,
        "n_chiarimenti": n_chiarimenti,
        "n_raffinamenti": n_raffinamenti,
    }

    try:
        # --- guardrail minimi di sicurezza ---
        if not id_utente or id_utente <= 0:
            return crea_errore_conduttore_output("id_utente non valido o mancante.", id_utente=id_utente, id_sessione=id_sessione, prompt=prompt_pulito)
        if not id_sessione or id_sessione <= 0:
            return crea_errore_conduttore_output("id_sessione non valido o mancante.", id_utente=id_utente, id_sessione=id_sessione, prompt=prompt_pulito)
        if not prompt_pulito:
            return crea_errore_conduttore_output("Input vuoto.", id_utente=id_utente, id_sessione=id_sessione, prompt="")

        if isinstance(richiesta_precedente, dict):
            try:
                richiesta_precedente = RichiestaRicerca.model_validate(richiesta_precedente)
            except Exception:
                richiesta_precedente = None

        # --- memoria (contesto per il router) ---
        memoria_output = leggi_contesto_memoria(id_utente, prompt_pulito, num_memoria=5)
        testo_memoria = costruisci_testo_contesto_memoria(memoria_output)
        log_notes["memoria_letta"] = memoria_output.success

        # --- router LLM: decisore primario ---
        consigli_sessione = testo_consigli_sessione(id_sessione)
        decisione_grezza, errore_router = decidi_router(
            prompt_pulito, storico_chat, testo_memoria, richiesta_precedente, n_chiarimenti,
            consigli_sessione=consigli_sessione,
        )
        if errore_router:
            log_notes["errore_router"] = errore_router
            print(f"[Router]     errore ({errore_router[:120]}) - uso fallback offline")
            decisione_grezza = decisione_fallback_offline(prompt_pulito)

        decisione = valida_decisione(decisione_grezza, prompt_pulito)
        decisione = correggi_decisione(decisione, prompt_pulito, richiesta_precedente, n_chiarimenti,
                                       msg_assistente=ultimo_messaggio_assistente(storico_chat))
        log_notes["decisione"] = decisione.model_dump()
        print(f"[Conduttore] {decisione.tipo_richiesta} | azione: {decisione.azione}")

        # guardrail: limite chiarimenti raggiunto -> si cerca con quel che c'e',
        # ma SOLO se c'e' davvero qualcosa da cercare (mai film a caso)
        if decisione.azione == "chiarimento" and n_chiarimenti >= MASSIMO_CHIARIMENTI:
            criteri_disponibili = (
                (decisione.ricerca is not None and decisione.ricerca.ha_vincoli_specifici())
                or (richiesta_precedente is not None and richiesta_precedente.ha_vincoli_specifici())
            )
            decisione.azione = "raccomandazione"
            if criteri_disponibili:
                print(f"[Conduttore] limite domande raggiunto ({MASSIMO_CHIARIMENTI}): cerco con le informazioni raccolte")
            else:
                # nessun criterio: ricerca generica basata sul profilo,
                # dichiarata come tale nella risposta (tipo 'generica')
                print(f"[Conduttore] limite domande raggiunto: propongo film popolari basati sul profilo invece di richiedere")

        # guardrail: raccomandazione NUOVA senza alcun criterio -> chiarimento
        if (
            decisione.azione == "raccomandazione"
            and decisione.tipo_richiesta == "NUOVA"
            and decisione.ricerca is not None
            and not decisione.ricerca.ha_vincoli_specifici()
            and n_chiarimenti < MASSIMO_CHIARIMENTI
        ):
            decisione.azione = "chiarimento"
            print("[Conduttore] raccomandazione senza criteri: meglio una domanda che film a caso")

        # --- memoria: aggiornamento se richiesto (anche insieme alla ricerca) ---
        memoria_aggiornata = False
        if decisione.aggiorna_memoria or decisione.azione == "aggiornamento_memoria":
            esito_memoria = aggiorna_memoria_da_testo(
                id_utente, prompt_pulito,
                contesto=estrai_ultimi_scambi(storico_chat, n_scambi=3),
            )
            memoria_aggiornata = esito_memoria.success
            log_notes["memory_update"] = esito_memoria.model_dump()
            if memoria_aggiornata:
                memoria_output = leggi_contesto_memoria(id_utente, prompt_pulito, num_memoria=5)

        # --- film dichiarati visti: persistenza precisa in SQLite ---
        tmdb_esclusi = [int(t) for t in (tmdb_id_da_evitare or []) if t is not None]
        if decisione.segna_visti_recenti:
            appena_visti = segna_ultimi_consigli_come_visti(id_utente, id_sessione)
            tmdb_esclusi = sorted(set(tmdb_esclusi + appena_visti))
            log_notes["visti_dichiarati"] = appena_visti

        azione = decisione.azione
        risposta_diretta = decisione.risposta_diretta

        # --- azioni senza ricerca ---
        if azione in ("conversazione_film", "domanda_memoria", "fuori_dominio", "aggiornamento_memoria"):
            if decisione.film_scelto:
                scelto = segna_film_scelto(id_utente, id_sessione, decisione.film_scelto)
                if scelto:
                    log_notes["film_scelto"] = scelto
            if not risposta_diretta:
                risposta_diretta = (
                    "Va bene, tengo conto di questa preferenza nei prossimi consigli."
                    if memoria_aggiornata
                    else "Posso aiutarti con film, preferenze cinematografiche e consigli personalizzati."
                )
            return ConduttoreOutput(
                success=True,
                azione=azione,
                risposta=risposta_diretta,
                film_consigliati=[],
                memoria_aggiornata=memoria_aggiornata,
                titolo_chat=titolo_chat,
                richiesta_eseguita=richiesta_precedente,  # lo stato di sessione resta invariato
                log_notes=log_notes,
            )

        if azione == "errore" and not errore_router:
            # L'LLM ha risposto ma la decisione non e' utilizzabile (JSON
            # parziale, azione mancante). Degradazione in due passi, mai un
            # errore secco se si puo' fare di meglio:
            # 1) gli estrattori deterministici trovano una richiesta chiara -> si cerca
            offline = valida_decisione(decisione_fallback_offline(prompt_pulito), prompt_pulito)
            if offline.azione == "raccomandazione":
                print("[Conduttore] decisione del router non valida: uso i criteri estratti dal testo e cerco")
                decisione = offline
                azione = decisione.azione
                risposta_diretta = ""
            else:
                # 2) altrimenti meglio una domanda amichevole che un errore
                domanda = genera_domanda_chiarimento(prompt_pulito, testo_memoria)
                if domanda:
                    print("[Conduttore] decisione del router non valida: chiedo un chiarimento invece di mostrare un errore")
                    return ConduttoreOutput(
                        success=True,
                        azione="chiarimento",
                        risposta=domanda,
                        film_consigliati=[],
                        memoria_aggiornata=memoria_aggiornata,
                        titolo_chat=titolo_chat,
                        richiesta_eseguita=richiesta_precedente,
                        log_notes=log_notes,
                    )

        if azione == "errore":
            errore = risposta_diretta or errore_router or "Decisione del conduttore non valida."
            if errore_router:
                log_notes["debug_errors"] = [errore_router]
            return crea_errore_conduttore_output(
                errore, risposta=risposta_diretta or "Mi dispiace, non sono riuscito a elaborare la richiesta. Riprova tra poco.",
                id_utente=id_utente, id_sessione=id_sessione, prompt=prompt_pulito, log_notes=log_notes,
            )

        # --- richiesta di ricerca: fusione con lo stato di sessione + profilo ---
        # un chiarimento accumula SEMPRE sopra lo stato esistente: le
        # informazioni gia' raccolte non si perdono e non si richiedono
        tipo_fusione = decisione.tipo_richiesta
        if decisione.azione == "chiarimento" and richiesta_precedente is not None:
            tipo_fusione = "RAFFINAZIONE"
        richiesta = fondi_con_precedente(decisione.ricerca or RichiestaRicerca(testo_richiesta=prompt_pulito),
                                         richiesta_precedente, tipo_fusione)
        # vincoli espressi dall'utente (o ereditati dalla sessione), PRIMA che il
        # profilo aggiunga i suoi generi da evitare: decide se serve raffinare
        vincoli_utente = richiesta.numero_vincoli()
        if anni_negati(prompt_pulito) and (richiesta.anno_min is not None or richiesta.anno_max is not None):
            richiesta.anno_min = None
            richiesta.anno_max = None
            print("[Conduttore] vincolo anno rimosso su richiesta dell'utente ('non solo quelli di quell'anno')")
        richiesta = applica_profilo(richiesta, memoria_output)
        richiesta = applica_gusti_feedback(richiesta, id_utente)
        richiesta.tmdb_id_da_evitare = sorted(set(richiesta.tmdb_id_da_evitare + tmdb_esclusi))
        richiesta.contesto_semantico = estrai_memorie_semantiche(memoria_output)
        log_notes["richiesta_ricerca"] = richiesta.model_dump(exclude={"candidati_salvati"})

        # --- chiarimento: domanda generata dall'LLM, stato parziale salvato ---
        if azione == "chiarimento":
            # se il router non ha scritto una vera domanda, la genera l'LLM
            domanda = risposta_diretta if "?" in risposta_diretta else ""
            domanda = domanda or genera_domanda_chiarimento(prompt_pulito, testo_memoria)
            return ConduttoreOutput(
                success=True,
                azione="chiarimento",
                risposta=domanda,
                film_consigliati=[],
                memoria_aggiornata=memoria_aggiornata,
                titolo_chat=titolo_chat,
                richiesta_eseguita=richiesta,  # richiesta parziale: la risposta dell'utente la completera'
                log_notes=log_notes,
            )

        # --- ricerca ---
        search_output = RicercaAgent(richiesta)
        # i candidati salvati sono stati consumati: lo stato di sessione non
        # li trascina oltre (verranno ri-salvati solo con una nuova domanda)
        richiesta.candidati_salvati = []
        log_notes["search_output"] = search_output.log_notes

        if not search_output.success or not search_output.film_candidati:
            return ConduttoreOutput(
                success=True,
                azione=azione,
                risposta=(
                    "🎬 Non ho trovato film reali che rispettino tutti i vincoli della richiesta. "
                    "Prova ad allargare un vincolo (voto, periodo, genere escluso) e riprovo."
                ),
                film_consigliati=[],
                memoria_aggiornata=memoria_aggiornata,
                titolo_chat=titolo_chat,
                richiesta_eseguita=richiesta,
                errors=search_output.errors,
                log_notes=log_notes,
            )

        # pool grande con pochi vincoli: meglio UNA domanda (dell'LLM) che una
        # scelta poco informata. Se l'LLM non genera la domanda, si mostrano i film.
        candidati_raccolti = int(search_output.log_notes.get("candidati_raccolti") or 0)
        senza_direzione_positiva = not (
            richiesta.generi_richiesti or richiesta.query_tema or richiesta.film_base
        )
        print(f"[Conduttore] candidati: {candidati_raccolti} | vincoli utente: {vincoli_utente}"
              f"{' (solo esclusioni/limiti, nessuna direzione positiva)' if senza_direzione_positiva else ''}"
              f" | domande fatte: {n_chiarimenti}/{MASSIMO_CHIARIMENTI} | raffinamenti: {n_raffinamenti}/{MASSIMO_RAFFINAMENTI}")
        pool_valido = int(search_output.log_notes.get("pool_valido") or 0)
        pool_precedente = int(search_output.log_notes.get("pool_precedente") or 0)
        vincoli_allentati = bool(search_output.log_notes.get("vincoli_allentati"))
        # si continua a chiedere finche' i candidati validi sono piu' di 6,
        # MA solo se l'ultima risposta ha fatto progressi: se il pool non si
        # e' ridotto, un'altra domanda non aiuterebbe (si mostrano i migliori)
        nessun_progresso = pool_precedente > 0 and pool_valido >= pool_precedente
        if nessun_progresso:
            print(f"[Conduttore] la risposta non ha ridotto il pool ({pool_precedente} -> {pool_valido}): mostro i migliori senza altre domande")
        angoli_liberi = angoli_liberi_richiesta(richiesta, search_output.pool_candidati)
        if pool_valido > 6 and not angoli_liberi:
            print("[Conduttore] nessun angolo libero su cui fare domande (periodo, voto e generi gia' decisi): mostro i migliori")
        if (
            decisione.tipo_richiesta in ("NUOVA", "RAFFINAZIONE")
            and pool_valido > 6
            and angoli_liberi
            and not nessun_progresso
            and not vincoli_allentati
            and n_raffinamenti < MASSIMO_RAFFINAMENTI
        ):
            print(f"[Conduttore] {pool_valido} candidati validi (> 6): salvo tutto e chiedo all'utente come restringere ({n_raffinamenti + 1}/{MASSIMO_RAFFINAMENTI})")
            esempi = ", ".join(f"{f.titolo} ({f.anno})" for f in search_output.film_candidati[:5])
            domanda = genera_domanda_raffinamento(
                prompt_pulito, richiesta, search_output.pool_candidati, esempi,
                angoli_liberi=angoli_liberi,
                scambi_recenti=estrai_ultimi_scambi(storico_chat, n_scambi=3),
                pool_precedente=pool_precedente,
                n_raffinamenti=n_raffinamenti,
            )
            if not domanda:
                print("[Raffinamento] l'LLM non ha generato la domanda: mostro direttamente i risultati")
            if domanda:
                # i candidati validi restano salvati nello stato di sessione:
                # la risposta dell'utente filtrera' QUESTI, senza ricominciare
                richiesta.candidati_salvati = search_output.pool_candidati
                print(f"[Conduttore] {len(richiesta.candidati_salvati)} candidati salvati: la prossima risposta li filtrera'")
                return ConduttoreOutput(
                    success=True,
                    azione="chiarimento",
                    risposta=domanda,
                    film_consigliati=[],
                    memoria_aggiornata=memoria_aggiornata,
                    titolo_chat=titolo_chat,
                    richiesta_eseguita=richiesta,
                    log_notes={**log_notes, "raffinamento_post_ricerca": True},
                )

        risposta_finale = formatta_risposta_markdown_locale(search_output.film_candidati)
        # apertura conversazionale: preferisce il commento del reranker (che ha
        # visto messaggio, contesto e film scelti); in mancanza, l'intro del
        # router se e' un'affermazione (una DOMANDA sopra una lista sarebbe
        # incoerente)
        commento = str(search_output.log_notes.get("commento_conversazionale") or "").strip()
        intro = commento or (risposta_diretta if (azione == "raccomandazione" and risposta_diretta and "?" not in risposta_diretta) else "")
        intro_usata = False
        if intro:
            risposta_finale = intro + "\n\n" + risposta_finale
            intro_usata = True
        if vincoli_allentati:
            risposta_finale = (
                "Con l'ultimo filtro non restava nessun film: ti mostro i piu' "
                "vicini alla tua richiesta.\n\n" + risposta_finale
            )
        if richiesta.tipo == "generica":
            risposta_finale = (
                "La richiesta era generica, quindi ti propongo alcuni film popolari "
                "adatti alle tue preferenze:\n\n" + risposta_finale
            )
        if memoria_aggiornata and not intro_usata:
            # solo se non c'e' gia' un'apertura naturale del router
            risposta_finale = "Preferenze segnate!\n\n" + risposta_finale

        durata = time.perf_counter() - start
        print(f"[Conduttore] risposta pronta OK ({durata:.1f}s)")
        print("=" * 60)
        return ConduttoreOutput(
            success=True,
            azione=azione,
            risposta=risposta_finale,
            film_consigliati=search_output.film_candidati,
            memoria_aggiornata=memoria_aggiornata,
            titolo_chat=titolo_chat,
            richiesta_eseguita=richiesta,
            log_notes=log_notes,
        )

    except Exception as e:
        errore = f"{type(e).__name__}: {e}"
        print(f"[Conduttore] errore: {errore[:100]}")
        return crea_errore_conduttore_output(
            normalizza_errore_llm(e),
            id_utente=id_utente,
            id_sessione=id_sessione,
            prompt=prompt_pulito,
            log_notes={"debug_errors": [errore]},
        )
