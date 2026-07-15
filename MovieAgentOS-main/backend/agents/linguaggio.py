"""Modulo linguistico condiviso di MovieAgentOS.

Unica fonte per: normalizzazione di testi/generi, tabella alias dei generi,
estrazione del sentiment sui generi (con gestione della negazione locale come
"ma non horror"), estrazione dei vincoli numerici (voto, anni) e lettura di
campi da oggetti film. Memoria, orchestrator e ricerca importano solo da qui:
niente copie divergenti della stessa logica.
"""

import re
from datetime import date
from typing import Any


GENERI_CANONICI: dict[str, str] = {
    "azione": "Azione",
    "avventura": "Avventura",
    "animazione": "Animazione",
    "commedia": "Commedia",
    "crime": "Crime",
    "documentario": "Documentario",
    "dramma": "Dramma",
    "famiglia": "Famiglia",
    "fantasy": "Fantasy",
    "storia": "Storia",
    "horror": "Horror",
    "musica": "Musica",
    "mistero": "Mistero",
    "romance": "Romance",
    "fantascienza": "Fantascienza",
    "thriller": "Thriller",
    "guerra": "Guerra",
    "western": "Western",
}

# alias -> chiave canonica (inglese, sinonimi, plurali, typo ricorrenti)
ALIAS_GENERI: dict[str, str] = {
    "action": "azione",
    "azioni": "azione",
    "adventure": "avventura",
    "animation": "animazione",
    "cartoni animati": "animazione",
    "comedy": "commedia",
    "commedie": "commedia",
    "comico": "commedia",
    "satirico": "commedia",
    "satira": "commedia",
    "crimine": "crime",
    "poliziesco": "crime",
    "documentary": "documentario",
    "documentari": "documentario",
    "drama": "dramma",
    "drammatico": "dramma",
    "family": "famiglia",
    "history": "storia",
    "storico": "storia",
    "orrore": "horror",
    "music": "musica",
    "musical": "musica",
    "mystery": "mistero",
    "romantico": "romance",
    "romantici": "romance",
    "sentimentale": "romance",
    "science fiction": "fantascienza",
    "sci fi": "fantascienza",
    "scifi": "fantascienza",
    "fantacienza": "fantascienza",
    "fantascenza": "fantascienza",
    "fantasceinza": "fantascienza",
    "war": "guerra",
}

# Tutte le forme riconosciute (canoniche + alias), gia' normalizzate.
_FORME_GENERE: dict[str, str] = {
    **{chiave: chiave for chiave in GENERI_CANONICI},
    **ALIAS_GENERI,
}

GENERI_NOTI: list[str] = list(GENERI_CANONICI.keys())

def normalizza_testo(testo: Any) -> str:
    """Minuscole, niente punteggiatura, spazi singoli."""
    testo = str(testo or "").lower()
    testo = re.sub(r"[^\w\s]", " ", testo)
    testo = testo.replace("_", " ")
    return " ".join(testo.split())


def normalizza_titolo(titolo: Any) -> str:
    """Come normalizza_testo, ma toglie anche l'anno tra parentesi."""
    testo = re.sub(r"\(\d{4}\)", "", str(titolo or ""))
    return normalizza_testo(testo)


def chiave_genere(nome: Any) -> str | None:
    """Restituisce la chiave canonica del genere, o None se non riconosciuto."""
    testo = normalizza_testo(nome)
    if not testo:
        return None
    if testo in _FORME_GENERE:
        return _FORME_GENERE[testo]
    # forma con articolo residuo: "il thriller", "l horror"
    parole = testo.split()
    if len(parole) >= 2 and parole[0] in {"il", "lo", "la", "le", "gli", "i", "l", "un", "una", "uno"}:
        resto = " ".join(parole[1:])
        if resto in _FORME_GENERE:
            return _FORME_GENERE[resto]
    return None


def normalizza_genere(nome: Any) -> str | None:
    """Nome canonico visualizzabile del genere ("Fantascienza"), o None."""
    chiave = chiave_genere(nome)
    return GENERI_CANONICI.get(chiave) if chiave else None


def filtra_generi_validi(generi: list[Any] | None) -> list[str]:
    """Tiene solo generi riconosciuti, in forma canonica, senza duplicati."""
    risultato: list[str] = []
    visti: set[str] = set()
    for genere in generi or []:
        chiave = chiave_genere(genere)
        if chiave and chiave not in visti:
            visti.add(chiave)
            risultato.append(GENERI_CANONICI[chiave])
    return risultato


def _occorrenze_generi(tokens: list[str]) -> list[tuple[int, str]]:
    """Trova i generi nel testo tokenizzato: lista di (indice_token, chiave)."""
    occorrenze: list[tuple[int, str]] = []
    i = 0
    while i < len(tokens):
        # bigrammi prima ("science fiction", "cartoni animati")
        if i + 1 < len(tokens):
            bigramma = f"{tokens[i]} {tokens[i + 1]}"
            if bigramma in _FORME_GENERE:
                occorrenze.append((i, _FORME_GENERE[bigramma]))
                i += 2
                continue
        token = tokens[i]
        if token in _FORME_GENERE:
            occorrenze.append((i, _FORME_GENERE[token]))
        elif token.startswith("storic"):
            occorrenze.append((i, "storia"))
        i += 1
    return occorrenze


def _storia_ha_contesto(tokens: list[str], indice: int) -> bool:
    """'storia' e' un genere solo con contesto stretto ("film di storia",
    "genere storia", "storia vera"): altrimenti e' la parola comune
    ("storia di vendetta", "film sulla storia di...")."""
    if tokens[indice].startswith("storic"):
        return True
    contesto = {"film", "genere", "generi", "cinema", "pellicola", "pellicole"}
    precedente = tokens[indice - 1] if indice >= 1 else ""
    due_prima = tokens[indice - 2] if indice >= 2 else ""
    if precedente in contesto:
        return True
    if precedente in {"di", "della", "sulla"} and due_prima in contesto and precedente == "di":
        return True
    dopo = tokens[indice + 1:indice + 2]
    return bool(dopo) and dopo[0] in {"vera", "antica", "medievale", "moderna"}


def estrai_generi_da_testo(testo: Any) -> list[str]:
    """Generi citati nel testo, in forma canonica, senza sentiment."""
    tokens = normalizza_testo(testo).split()
    risultato: list[str] = []
    visti: set[str] = set()
    for indice, chiave in _occorrenze_generi(tokens):
        if chiave == "storia" and not _storia_ha_contesto(tokens, indice):
            continue
        if chiave not in visti:
            visti.add(chiave)
            risultato.append(GENERI_CANONICI[chiave])
    return risultato


# frasi segnale, gia' normalizzate; l'ordine non conta, si usa la piu' vicina.
_SEGNALI_NEGATIVI = (
    "non mi piacciono", "non mi piace", "non mi pace", "non amo",
    "non voglio", "non sopporto", "odio", "detesto",
    "evita", "evitare", "escludi", "escludere", "escludendo",
    "da evitare", "mi annoia", "mi annoiano",
)
_SEGNALI_POSITIVI = (
    "mi piacciono", "mi piace", "mi piacce", "preferisco",
    "adoro", "amo", "apprezzo", "preferiti sono", "generi preferiti",
    "mi interessano", "mi interessa",
)

# parole che negano localmente il genere subito successivo
_NEGAZIONI_LOCALI = {"non", "no", "senza", "tranne", "eccetto", "escluso", "esclusi"}
# "solo/soltanto commedia" e' una RESTRIZIONE AL genere, non un suo rifiuto:
# se compare tra la negazione e il genere ("no, solo commedia"), la negazione
# riguarda le alternative, non il genere stesso
_RESTRIZIONI = {"solo", "soltanto", "solamente"}


def _eventi_sentiment(testo_norm: str) -> list[tuple[int, str]]:
    """Posizioni (indice carattere, sentimento) dei segnali nel testo normalizzato."""
    eventi: list[tuple[int, str]] = []
    for segnale in _SEGNALI_NEGATIVI:
        for match in re.finditer(rf"\b{re.escape(segnale)}\b", testo_norm):
            eventi.append((match.start(), "negativo"))
    for segnale in _SEGNALI_POSITIVI:
        for match in re.finditer(rf"\b{re.escape(segnale)}\b", testo_norm):
            prima = testo_norm[max(0, match.start() - 5):match.start()].strip()
            if prima.endswith("non"):
                continue  # gia' coperto dal segnale negativo composto
            eventi.append((match.start(), "positivo"))
    eventi.sort()
    return eventi


def estrai_generi_per_sentimento(testo: Any) -> tuple[list[str], list[str]]:
    """Separa i generi graditi da quelli rifiutati.

    Regole, in ordine di priorita' per ogni occorrenza di genere:
    1. negazione locale nei 2 token precedenti ("ma NON horror", "SENZA horror",
       "tranne horror") -> negativo, anche dentro una frase positiva;
    2. altrimenti sentimento del segnale piu' vicino che precede il genere;
    3. se un genere risulta sia positivo sia negativo, prevale il rifiuto.
    """
    testo_norm = normalizza_testo(testo)
    tokens = testo_norm.split()
    if not tokens:
        return [], []

    eventi = _eventi_sentiment(testo_norm)

    # posizione carattere di inizio di ogni token
    posizioni: list[int] = []
    cursore = 0
    for token in tokens:
        inizio = testo_norm.find(token, cursore)
        posizioni.append(inizio)
        cursore = inizio + len(token)

    positivi: dict[str, None] = {}
    negativi: dict[str, None] = {}

    for indice, chiave in _occorrenze_generi(tokens):
        if chiave == "storia" and not _storia_ha_contesto(tokens, indice):
            continue

        finestra = tokens[max(0, indice - 2):indice]
        negazione_locale = any(token in _NEGAZIONI_LOCALI for token in finestra)
        if negazione_locale and any(token in _RESTRIZIONI for token in finestra):
            negazione_locale = False  # "no, solo commedia": restrizione, non rifiuto
        if negazione_locale:
            negativi[chiave] = None
            continue

        sentimento = None
        pos_genere = posizioni[indice]
        for pos_evento, tipo in eventi:
            if pos_evento < pos_genere:
                sentimento = tipo
            else:
                break
        if sentimento == "positivo":
            positivi[chiave] = None
        elif sentimento == "negativo":
            negativi[chiave] = None

    generi_positivi = [GENERI_CANONICI[c] for c in positivi if c not in negativi]
    generi_negativi = [GENERI_CANONICI[c] for c in negativi]
    return generi_positivi, generi_negativi


_ADDITIVI = {"anche", "pure", "oltre", "inoltre"}


def generi_additivi(testo: Any) -> list[str]:
    """Generi introdotti in modo ADDITIVO ("mi piace anche il dramma",
    "pure un thriller"): vanno AGGIUNTI alla richiesta in corso, non sono
    un cambio di argomento."""
    tokens = normalizza_testo(testo).split()
    trovati: list[str] = []
    for indice, chiave in _occorrenze_generi(tokens):
        finestra = tokens[max(0, indice - 3):indice]
        if any(token in _ADDITIVI for token in finestra):
            nome = GENERI_CANONICI[chiave]
            if nome not in trovati:
                trovati.append(nome)
    return trovati


def estrai_generi_positivi(testo: Any) -> list[str]:
    return estrai_generi_per_sentimento(testo)[0]


def estrai_generi_esclusi(testo: Any) -> list[str]:
    return estrai_generi_per_sentimento(testo)[1]

def _numero_da_match(match: re.Match[str]) -> float | None:
    valore = next((gruppo for gruppo in match.groups() if gruppo), None)
    if valore is None:
        return None
    try:
        return float(str(valore).replace(",", "."))
    except ValueError:
        return None


def estrai_voto_min(testo: Any) -> float | None:
    pattern = re.compile(
        r"(?:voto|rating|valutazione)[^\d]{0,40}(?:>=|>|uguale o superiore a|superiore a|sopra|oltre|minimo|almeno|maggiore di)\s*(\d+(?:[\.,]\d+)?)"
        r"|(?:voto\s+)?(?:superiore a|sopra|oltre|minimo|almeno|maggiore di)\s*(\d+(?:[\.,]\d+)?)",
        re.IGNORECASE,
    )
    match = pattern.search(str(testo or ""))
    valore = _numero_da_match(match) if match else None
    if valore is not None and 0 <= valore <= 10:
        return valore
    # vincolo qualitativo ma reale: "voto alto", "ben valutato", "ottimo voto"
    if re.search(r"\b(?:vot[oi]\s+alt[oi]|voto\s+elevato|ottimo\s+voto|ben\s+valutat)", str(testo or ""), re.IGNORECASE):
        return 7.0
    return None


def estrai_voto_max(testo: Any) -> float | None:
    pattern = re.compile(
        r"(?:voto|rating|valutazione)[^\d]{0,40}(?:<=|<|uguale o inferiore a|inferiore a|sotto|massimo|minore di)\s*(\d+(?:[\.,]\d+)?)"
        r"|(?:voto\s+)?(?:inferiore a|massimo|minore di|sotto)\s*(\d+(?:[\.,]\d+)?)",
        re.IGNORECASE,
    )
    match = pattern.search(str(testo or ""))
    valore = _numero_da_match(match) if match else None
    return valore if valore is not None and 0 <= valore <= 10 else None


def anni_negati(testo: Any) -> bool:
    """Vero se il vincolo temporale e' NEGATO ("non dammi solo quelli del 2026",
    "non solo film recenti"): in quel caso il vincolo anno va rimosso,
    non applicato."""
    return bool(re.search(
        r"\bnon\b[^,.;!?]{0,50}(?:\b(?:19|20)\d{2}\b|\brecent[ei]\b|\bnuov[oiae]\b|\bmodern[oiae]\b|\bclassic[oiae]\b)",
        str(testo or ""), re.IGNORECASE,
    ))


def estrai_anni(testo: Any) -> tuple[int | None, int | None]:
    testo = str(testo or "")
    if anni_negati(testo):
        return None, None

    match = re.search(r"anni\s*(\d{2})\b", testo, re.IGNORECASE)
    if match:
        decade = int(match.group(1))
        base = 1900 + decade if decade > 30 else 2000 + decade
        return base, base + 9

    match = re.search(r"(?:tra|fra)\s*(?:il\s*)?(\d{4})\s*(?:e|-|al|a)\s*(?:il\s*)?(\d{4})", testo, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"(?:dal|da)\s*(?:il\s*)?(\d{4})\s*(?:al|a|-)\s*(?:il\s*)?(\d{4})", testo, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(
        r"(?:film|pellicole|titoli|opere)\s+(?:uscit[ei]|pubblicat[ei]|rilasciat[ei])?\s*(?:nel|del|in|dell'anno)\s*(\d{4})"
        r"|(?:uscit[oiae]|pubblicat[oiae]|rilasciat[oiae])\s+(?:nel|del|in)\s*(\d{4})",
        testo,
        re.IGNORECASE,
    )
    if match:
        anno = int(next(gruppo for gruppo in match.groups() if gruppo))
        return anno, anno

    # "quest'anno" / "di questo anno (2026)": l'anno corrente o quello citato
    match = re.search(r"quest[o']? ?anno(?:\s*(\d{4}))?", testo, re.IGNORECASE)
    if match:
        anno = int(match.group(1)) if match.group(1) else date.today().year
        return anno, anno

    anno_min = None
    anno_max = None
    match = re.search(r"(?:dopo|dal|successivi a)\s*(?:il\s*)?(\d{4})", testo, re.IGNORECASE)
    if match:
        anno_min = int(match.group(1))
    match = re.search(r"(?:prima|fino al|entro)\s*(?:del\s*|il\s*)?(\d{4})", testo, re.IGNORECASE)
    if match:
        anno_max = int(match.group(1))

    # periodi impliciti ma reali, dal piu' stretto al piu' largo:
    # "recenti/nuovi" = ultimi 3 anni; "moderno/contemporaneo" = ultimi 10;
    # "classico/d'epoca/vecchio" = almeno 20 anni fa
    if anno_min is None and anno_max is None:
        if re.search(r"\b(?:recent[ei]|nuov[oiae]|appena uscit[oiae]|ultime uscite)\b", testo, re.IGNORECASE):
            anno_min = date.today().year - 3
        elif re.search(r"\b(?:modern[oiae]|contemporane[oiae]|attual[ei])\b", testo, re.IGNORECASE):
            anno_min = date.today().year - 10
        elif re.search(r"\b(?:classic[oiae]|d'epoca|vecchi[oae]?)\b", testo, re.IGNORECASE):
            anno_max = date.today().year - 20

    return anno_min, anno_max


def estrai_film_riferimento(testo: Any) -> str:
    """Titolo di riferimento in costrutti come 'simile a X', 'qualcosa come X',
    'sullo stile di X'. Restituisce "" se non c'e' o se X e' un genere
    (in quel caso 'come una commedia' e' una richiesta di genere, non di titolo)."""
    match = re.search(
        r"\b(?:simil[ei]\s+a|qualcosa\s+come|sullo\s+stile\s+di|proprio\s+come|un\s+film\s+come)\s+(.{2,60})",
        str(testo or ""), re.IGNORECASE,
    )
    if not match:
        return ""
    titolo = match.group(1)
    # taglia alla prima virgola/punto o congiunzione che apre un'altra clausola
    titolo = re.split(r"[,.;!?]|\b(?:ma|pero|però|con voto|senza|uscito)\b", titolo, maxsplit=1)[0]
    titolo = titolo.strip(" '\"")
    if not titolo or chiave_genere(titolo) is not None:
        return ""
    if normalizza_testo(titolo) in {"questo", "questa", "questi", "quelli", "quelle", "prima", "sempre"}:
        return ""
    return titolo


def limita_score(score: float) -> float:
    if score > 1:
        return 1.0
    if score < -1:
        return -1.0
    return round(score, 3)


def estrai_valore_film(film: Any, chiave: str, default: Any = None) -> Any:
    if film is None:
        return default
    if isinstance(film, dict):
        return film.get(chiave, default)
    return getattr(film, chiave, default)


def estrai_generi_film(film:Any) -> list[str]:
    generi = estrai_valore_film(film, "generi", [])
    if generi is None:
        return []
    if isinstance(generi, str):
        generi = generi.split(",")
    return filtra_generi_validi([str(g) for g in generi if str(g).strip()])

