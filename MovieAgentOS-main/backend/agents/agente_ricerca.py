"""SearchAgent di MovieAgentOS.

Riceve una RichiestaRicerca strutturata dal conduttore (nessun testo da
ri-parsare), raccoglie un pool ampio di film reali da TMDB che rispettano i
vincoli duri, poi sceglie i migliori con un rerank in due passi:
punteggio deterministico (vincoli + preferenze morbide del profilo) e rerank
LLM del top-10 (profilo + memoria semantica Chroma), con motivazioni.

Regole:
- i vincoli si applicano in UN solo punto (film_rispetta_vincoli);
- l'LLM sceglie solo tra candidati reali (validazione per tmdb_id);
- niente fallback popolari per richieste specifiche (genere, tema, simili,
  titolo): meglio zero risultati dichiarati che film sbagliati;
- per "simili a X": X non compare mai nei risultati.
"""

import re
import time
from datetime import date
from typing import Any

import requests
from agno.agent import Agent

from backend.agents.linguaggio import (
    GENERI_CANONICI,
    chiave_genere,
    normalizza_testo,
    normalizza_titolo,
)
from backend.agents.llm_config import crea_model_agente, set_llm_api_key
from backend.agents.schemas import FilmCandidato, RichiestaRicerca, SearchAgentOutput
from backend.agents.text_utils import estrai_json_da_testo
from backend.tools.tmdb_tools import (
    cerca_film,
    cerca_film_per_genere,
    get_TMDB_API_KEY,
    get_film_popolari as get_film_popolari_base,
    get_film_simili,
    get_generi_film,
)

POOL_MASSIMO = 40
CANDIDATI_RERANK = 12
# il rerank LLM propone tutti i candidati davvero adatti fino a un massimo
# di RISULTATI_MASSIMI: se gli adatti sono di piu', prende i PIU' VICINI alla
# richiesta e il conduttore fara' una domanda per restringere; senza LLM si
# ricade sui migliori RISULTATI_FALLBACK deterministici
RISULTATI_MASSIMI = 6
RISULTATI_FALLBACK = 5

def contiene_errore_tool(risultato: Any) -> bool:
    if isinstance(risultato, dict):
        return "errore" in risultato
    if isinstance(risultato, list) and risultato:
        return isinstance(risultato[0], dict) and "errore" in risultato[0]
    return False


def estrai_errore_tool(risultato: Any) -> str:
    if isinstance(risultato, dict):
        return str(risultato.get("errore", "Errore sconosciuto dal tool TMDB."))
    if isinstance(risultato, list) and risultato and isinstance(risultato[0], dict):
        return str(risultato[0].get("errore", "Errore sconosciuto dal tool TMDB."))
    return "Errore sconosciuto dal tool TMDB."


def chiama_tool(nome: str, funzione, *args, **kwargs):
    nome_corto = nome.replace("tool_", "").replace("_", " ")
    try:
        risultato = funzione(*args, **kwargs)
        errore = estrai_errore_tool(risultato) if contiene_errore_tool(risultato) else None
        quantita = len(risultato) if isinstance(risultato, list) else 1
        if errore:
            print(f"[Tool]       {nome_corto} -> errore: {str(errore)[:60]}")
        else:
            print(f"[Tool]       {nome_corto} -> {quantita} risultati")
        return risultato
    except Exception as e:
        print(f"[Tool]       {nome_corto} -> eccezione: {type(e).__name__}")
        raise


_GENERI_CACHE: list[dict[str, Any]] | None = None


def get_generi_film_cached() -> list[dict[str, Any]]:
    global _GENERI_CACHE
    if _GENERI_CACHE is None:
        generi = chiama_tool("tool_get_generi_film", get_generi_film)
        if not contiene_errore_tool(generi):
            _GENERI_CACHE = generi
        else:
            return generi
    return _GENERI_CACHE


#tmdb
def _film_pulito(film: dict[str, Any]) -> dict[str, Any]:
    data_uscita = film.get("release_date") or ""
    return {
        "tmdb_id": film.get("id"),
        "titolo": film.get("title"),
        "titolo_originale": film.get("original_title"),
        "anno": data_uscita[:4] if data_uscita else "Sconosciuto",
        "descrizione": film.get("overview"),
        "voto_medio": film.get("vote_average"),
        "numero_voti": film.get("vote_count"),
        "popolarita": film.get("popularity"),
        "poster_path": film.get("poster_path"),
        "lingua_originale": film.get("original_language"),
        "genre_ids": film.get("genre_ids", []),
    }


def get_film_popolari(page: int = 1, voto_min: float | None = None, voto_max: float | None = None,
                      anno_min: int | None = None, anno_max: int | None = None,
                      generi_esclusi_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Popolari TMDB; con vincoli usa discover per applicarli a livello API."""
    if voto_min is None and voto_max is None and anno_min is None and anno_max is None and not generi_esclusi_ids:
        return get_film_popolari_base(page=page)

    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    param: dict[str, Any] = {
        "api_key": api_tmdb,
        "language": "it-IT",
        "include_adult": False,
        "include_video": False,
        "sort_by": "popularity.desc",
        "vote_count.gte": 20,
        "page": page if page and page > 0 else 1,
        "region": "IT",
    }
    if voto_min is not None:
        param["vote_average.gte"] = voto_min
    if voto_max is not None:
        param["vote_average.lte"] = voto_max
    if anno_min:
        param["primary_release_date.gte"] = f"{anno_min}-01-01"
    if anno_max:
        param["primary_release_date.lte"] = f"{anno_max}-12-31"
    if generi_esclusi_ids:
        param["without_genres"] = ",".join(str(gid) for gid in generi_esclusi_ids)

    try:
        response = requests.get("https://api.themoviedb.org/3/discover/movie", params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la richiesta a TMDB: {e}"}]

    return [_film_pulito(f) for f in response.json().get("results", []) if f.get("overview")][:10]


def cerca_keyword_tmdb(query: str) -> list[dict[str, Any]]:
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]
    if not query or not query.strip():
        return []
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/keyword",
            params={"api_key": api_tmdb, "query": query.strip(), "page": 1},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la ricerca keyword TMDB: {e}"}]
    return [
        {"id": item.get("id"), "name": item.get("name")}
        for item in response.json().get("results", [])[:5]
        if item.get("id")
    ]


def cerca_film_per_keyword(keyword_ids: list[int], page: int = 1,
                           generi_inclusi_ids: list[int] | None = None,
                           generi_esclusi_ids: list[int] | None = None,
                           voto_min: float | None = None, voto_max: float | None = None,
                           anno_min: int | None = None, anno_max: int | None = None) -> list[dict[str, Any]]:
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    ids = [str(k) for k in keyword_ids if k]
    if not ids:
        return []

    param: dict[str, Any] = {
        "api_key": api_tmdb,
        "language": "it-IT",
        "include_adult": False,
        "include_video": False,
        "sort_by": "popularity.desc",
        "vote_count.gte": 25,
        "with_keywords": "|".join(ids[:8]),
        "page": page if page and page > 0 else 1,
    }
    if generi_inclusi_ids:
        param["with_genres"] = ",".join(str(gid) for gid in generi_inclusi_ids)
    if generi_esclusi_ids:
        param["without_genres"] = ",".join(str(gid) for gid in generi_esclusi_ids)
    if voto_min is not None:
        param["vote_average.gte"] = voto_min
    if voto_max is not None:
        param["vote_average.lte"] = voto_max
    if anno_min:
        param["primary_release_date.gte"] = f"{anno_min}-01-01"
    if anno_max:
        param["primary_release_date.lte"] = f"{anno_max}-12-31"

    try:
        response = requests.get("https://api.themoviedb.org/3/discover/movie", params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la ricerca per tema TMDB: {e}"}]

    return [_film_pulito(f) for f in response.json().get("results", []) if f.get("overview")][:10]


# Traduzioni tema IT -> keyword EN (le keyword TMDB sono in inglese).
ALIAS_TEMA = {
    "vendetta": ["revenge"],
    "riscatto": ["redemption"],
    "redenzione": ["redemption"],
    "amicizia": ["friendship"],
    "amore": ["love", "love story"],
    "tradimento": ["betrayal"],
    "sopravvivenza": ["survival"],
    "viaggio nel tempo": ["time travel"],
    "rapina": ["heist"],
    "mafia": ["mafia", "organized crime"],
    "serial killer": ["serial killer"],
    "indagine": ["investigation", "detective"],
    "investigazione": ["investigation", "detective"],
    "distopia": ["dystopia"],
    "apocalisse": ["apocalypse", "post-apocalyptic"],
    "intelligenza artificiale": ["artificial intelligence"],
    "robot": ["robot"],
    "spazio": ["space"],
    "alieni": ["alien"],
    "storia vera": ["based on true story"],
    "crescita": ["coming of age"],
    "formazione": ["coming of age"],
    "solitudine": ["loneliness"],
    "scuola": ["school"],
    "sport": ["sports"],
    "cucina": ["cooking", "chef"],
    "guerra": ["war"],
}


def varianti_query_tema(query: str) -> list[str]:
    query_norm = normalizza_testo(query)
    varianti: list[str] = []
    if query_norm:
        varianti.append(query_norm)
    for chiave, alias in ALIAS_TEMA.items():
        if chiave in query_norm:
            varianti.extend(alias)
    varianti.extend([p for p in query_norm.split() if len(p) > 2][:4])
    return list(dict.fromkeys(v.strip() for v in varianti if v.strip()))[:8]

def anno_come_int(anno: Any) -> int | None:
    if anno is None:
        return None
    match = re.search(r"\d{4}", str(anno))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def mappa_generi_da_ids(film: dict[str, Any], generi_tmdb: list[dict[str, Any]] | None) -> list[str]:
    if film.get("generi"):
        return [str(g) for g in film.get("generi") or [] if str(g).strip()]
    if not generi_tmdb:
        return []
    mappa = {
        g.get("id"): str(g.get("nome") or g.get("name") or "")
        for g in generi_tmdb
        if g.get("id") and (g.get("nome") or g.get("name"))
    }
    return [mappa[gid] for gid in film.get("genre_ids", []) if gid in mappa]


def normalizza_film_candidato(film: dict[str, Any], motivo_ricerca: str = "",
                              generi_tmdb: list[dict[str, Any]] | None = None) -> FilmCandidato | None:
    if not isinstance(film, dict) or film.get("errore"):
        return None
    tmdb_id = film.get("tmdb_id") or film.get("id")
    titolo = film.get("titolo") or film.get("title") or film.get("titolo_originale") or film.get("original_title") or ""
    if not tmdb_id and not titolo:
        return None
    data_uscita = film.get("data_uscita") or film.get("release_date") or ""
    anno = film.get("anno") or (data_uscita[:4] if isinstance(data_uscita, str) and len(data_uscita) >= 4 else None)
    return FilmCandidato(
        tmdb_id=tmdb_id,
        titolo=titolo,
        anno=anno,
        descrizione=film.get("descrizione") or film.get("overview"),
        generi=mappa_generi_da_ids(film, generi_tmdb),
        voto_medio=film.get("voto_medio") or film.get("vote_average"),
        numero_voti=film.get("numero_voti") or film.get("vote_count"),
        poster_path=film.get("poster_path"),
        motivo_ricerca=motivo_ricerca or None,
    )


def normalizza_lista_film(film_raw: list[dict[str, Any]], motivo_ricerca: str,
                          generi_tmdb: list[dict[str, Any]] | None = None,
                          limite: int = 20) -> list[FilmCandidato]:
    candidati: list[FilmCandidato] = []
    ids_visti: set[int] = set()
    titoli_visti: set[str] = set()
    for film in film_raw:
        candidato = normalizza_film_candidato(film, motivo_ricerca=motivo_ricerca, generi_tmdb=generi_tmdb)
        if not candidato:
            continue
        titolo_norm = normalizza_titolo(candidato.titolo)
        if candidato.tmdb_id and candidato.tmdb_id in ids_visti:
            continue
        if titolo_norm and titolo_norm in titoli_visti:
            continue
        if candidato.tmdb_id:
            ids_visti.add(candidato.tmdb_id)
        if titolo_norm:
            titoli_visti.add(titolo_norm)
        candidati.append(candidato)
        if len(candidati) >= limite:
            break
    return candidati


def trova_genere_tmdb(nome_genere: str, generi_tmdb: list[dict[str, Any]]) -> dict[str, Any] | None:
    chiave = chiave_genere(nome_genere)
    target = normalizza_testo(GENERI_CANONICI.get(chiave, nome_genere)) if chiave else normalizza_testo(nome_genere)
    if not target:
        return None
    for genere in generi_tmdb or []:
        nome = normalizza_testo(str(genere.get("nome") or genere.get("name") or ""))
        if nome == target or target in nome or nome in target:
            return genere
    return None


def ids_generi_tmdb(nomi_generi: list[str], generi_tmdb: list[dict[str, Any]] | None) -> list[int]:
    ids: list[int] = []
    for nome in nomi_generi or []:
        genere = trova_genere_tmdb(nome, generi_tmdb or [])
        if genere and genere.get("id") is not None:
            ids.append(int(genere["id"]))
    return list(dict.fromkeys(ids))


def film_rispetta_vincoli(film: FilmCandidato, richiesta: RichiestaRicerca) -> bool:
    """Unico filtro sui vincoli duri della richiesta."""
    if film.tmdb_id is not None and film.tmdb_id in set(richiesta.tmdb_id_da_evitare or []):
        return False

    titolo_norm = normalizza_titolo(film.titolo)
    for titolo in richiesta.titoli_da_evitare or []:
        if titolo_norm and titolo_norm == normalizza_titolo(titolo):
            return False

    voto = film.voto_medio
    if voto is not None:
        try:
            voto_float = float(voto)
        except (TypeError, ValueError):
            voto_float = None
        if voto_float is not None:
            if richiesta.voto_min is not None and voto_float < richiesta.voto_min:
                return False
            if richiesta.voto_max is not None and voto_float > richiesta.voto_max:
                return False

    anno = anno_come_int(film.anno)
    if anno is not None:
        if richiesta.anno_min is not None and anno < richiesta.anno_min:
            return False
        if richiesta.anno_max is not None and anno > richiesta.anno_max:
            return False
        if anno > date.today().year + 1:
            return False

    generi_film = {chiave_genere(g) for g in film.generi} - {None}
    for genere in richiesta.generi_esclusi or []:
        if chiave_genere(genere) in generi_film:
            return False

    # per "simili" il legame col film base conta piu' del match esatto di genere
    if richiesta.tipo != "simili":
        richiesti = {chiave_genere(g) for g in richiesta.generi_richiesti or []} - {None}
        if richiesti and not richiesti.intersection(generi_film):
            return False

    return True


def punteggio_film(film: FilmCandidato, richiesta: RichiestaRicerca) -> float:
    """Punteggio deterministico: pertinenza + preferenze morbide del profilo."""
    score = 0.0
    generi_film = {chiave_genere(g) for g in film.generi} - {None}

    richiesti = {chiave_genere(g) for g in richiesta.generi_richiesti or []} - {None}
    score += len(richiesti.intersection(generi_film)) * 5.0

    # con una direzione esplicita (generi/tema chiesti ORA) il profilo passato
    # pesa la meta': la richiesta attuale comanda, il profilo personalizza
    peso_profilo = 0.5 if (richiesti or richiesta.query_tema) else 1.0

    preferiti = {chiave_genere(g) for g in richiesta.profilo_generi_preferiti or []} - {None}
    score += len(preferiti.intersection(generi_film)) * 2.0 * peso_profilo

    for chiave, valore in (richiesta.profilo_segnali or {}).items():
        if chiave_genere(chiave) in generi_film:
            try:
                score += float(valore) * 2.0 * peso_profilo
            except (TypeError, ValueError):
                pass

    # feedback thumbs: criterio importante nella scelta tra candidati validi
    graditi = {chiave_genere(g) for g in richiesta.generi_graditi_feedback or []} - {None}
    sgraditi = {chiave_genere(g) for g in richiesta.generi_sgraditi_feedback or []} - {None}
    score += len(graditi.intersection(generi_film)) * 1.5 * peso_profilo
    score -= len(sgraditi.intersection(generi_film)) * 2.5 * peso_profilo

    testo_richiesta = normalizza_testo(f"{richiesta.query_tema} {richiesta.testo_richiesta}")
    testo_film = normalizza_testo(f"{film.titolo} {film.descrizione or ''}")
    for parola in set(testo_richiesta.split()):
        if len(parola) > 3 and parola in testo_film:
            score += 1.0

    try:
        score += min(float(film.voto_medio or 0), 10) * 0.5
    except (TypeError, ValueError):
        pass
    try:
        score += min(float(film.numero_voti or 0), 2000) / 2000
    except (TypeError, ValueError):
        pass
    return score


def deduplica_film(candidati: list[FilmCandidato]) -> list[FilmCandidato]:
    risultato: list[FilmCandidato] = []
    ids: set[int] = set()
    titoli: set[str] = set()
    for film in candidati:
        titolo_norm = normalizza_titolo(film.titolo)
        if film.tmdb_id and film.tmdb_id in ids:
            continue
        if titolo_norm and titolo_norm in titoli:
            continue
        if film.tmdb_id:
            ids.add(film.tmdb_id)
        if titolo_norm:
            titoli.add(titolo_norm)
        risultato.append(film)
    return risultato


def filtra_e_ordina(candidati: list[FilmCandidato], richiesta: RichiestaRicerca,
                    limite: int | None = None) -> list[FilmCandidato]:
    filtrati = [f for f in deduplica_film(candidati) if film_rispetta_vincoli(f, richiesta)]
    ordinati = sorted(filtrati, key=lambda f: punteggio_film(f, richiesta), reverse=True)
    return ordinati if limite is None else ordinati[:limite]


def piu_vicini_alla_richiesta(salvati: list[FilmCandidato], richiesta: RichiestaRicerca) -> list[FilmCandidato]:
    """Quando l'ultimo filtro azzera i candidati: i film salvati rispettavano
    tutti i vincoli PRECEDENTI, quindi sono i piu' vicini alla richiesta.
    Restano escluse solo le cose non negoziabili (film gia' visti/consigliati),
    e si ordina per affinita' con la richiesta completa."""
    evitare_ids = set(richiesta.tmdb_id_da_evitare or [])
    evitare_titoli = {normalizza_titolo(t) for t in (richiesta.titoli_da_evitare or []) if t}
    validi = [
        f for f in deduplica_film(salvati)
        if f.tmdb_id not in evitare_ids
        and normalizza_titolo(f.titolo or "") not in evitare_titoli
    ]
    return sorted(validi, key=lambda f: punteggio_film(f, richiesta), reverse=True)


def _profondita_ricerca(richiesta: RichiestaRicerca) -> tuple[int, int]:
    """Piu' film da escludere -> si scava piu' a fondo (continuazioni)."""
    n_escl = len(richiesta.titoli_da_evitare or []) + len(richiesta.tmdb_id_da_evitare or [])
    max_pagine = min(15, 4 + n_escl // 3)
    soglia = min(120, POOL_MASSIMO + 6 * n_escl)
    return max_pagine, soglia



def _raccogli_per_genere(richiesta: RichiestaRicerca, generi_tmdb: list[dict[str, Any]],
                         tool_usati: list[str], warnings: list[str]) -> list[FilmCandidato]:
    nome_genere = (richiesta.generi_richiesti or [""])[0]
    genere = trova_genere_tmdb(nome_genere, generi_tmdb)
    if not genere:
        warnings.append(f"Genere TMDB non trovato per: {nome_genere}.")
        return []
    risultati: list[dict[str, Any]] = []
    max_pagine, soglia = _profondita_ricerca(richiesta)
    pagina_iniziale = max(1, richiesta.pagina)
    for page in range(pagina_iniziale, pagina_iniziale + max_pagine):
        pagina = chiama_tool(
            "tool_cerca_film_per_genere", cerca_film_per_genere, genere.get("id"),
            page=page,
            anno_min=richiesta.anno_min, anno_max=richiesta.anno_max,
            voto_min=richiesta.voto_min, voto_max=richiesta.voto_max,
        )
        tool_usati.append(f"cerca_film_per_genere_page_{page}")
        if contiene_errore_tool(pagina):
            warnings.append(estrai_errore_tool(pagina))
            break
        if not pagina:
            break
        risultati.extend(pagina)
        if len(risultati) >= soglia:
            break
    return normalizza_lista_film(risultati, "", generi_tmdb=generi_tmdb, limite=soglia)


def _raccogli_per_tema(richiesta: RichiestaRicerca, generi_tmdb: list[dict[str, Any]],
                       tool_usati: list[str], warnings: list[str]) -> list[FilmCandidato]:
    varianti = varianti_query_tema(richiesta.query_tema or richiesta.testo_richiesta)
    keyword_ids: list[int] = []
    for variante in varianti[:5]:
        keywords = chiama_tool("tool_cerca_keyword_tmdb", cerca_keyword_tmdb, variante)
        tool_usati.append("cerca_keyword_tmdb")
        if contiene_errore_tool(keywords):
            warnings.append(estrai_errore_tool(keywords))
            continue
        for keyword in keywords:
            kid = keyword.get("id")
            if kid and int(kid) not in keyword_ids:
                keyword_ids.append(int(kid))
        if len(keyword_ids) >= 6:
            break

    risultati: list[dict[str, Any]] = []
    if keyword_ids:
        inclusi = ids_generi_tmdb(richiesta.generi_richiesti or [], generi_tmdb)
        esclusi = ids_generi_tmdb(richiesta.generi_esclusi or [], generi_tmdb)
        pagina_iniziale = max(1, richiesta.pagina)
        for page in range(pagina_iniziale, pagina_iniziale + 3):
            pagina = chiama_tool(
                "tool_cerca_film_per_keyword", cerca_film_per_keyword, keyword_ids,
                page=page,
                generi_inclusi_ids=inclusi, generi_esclusi_ids=esclusi,
                voto_min=richiesta.voto_min, voto_max=richiesta.voto_max,
                anno_min=richiesta.anno_min, anno_max=richiesta.anno_max,
            )
            tool_usati.append(f"cerca_film_per_keyword_page_{page}")
            if contiene_errore_tool(pagina):
                warnings.append(estrai_errore_tool(pagina))
                break
            risultati.extend(pagina)
            if len(risultati) >= POOL_MASSIMO:
                break

    if not risultati:
        # ricerca testuale come secondo tentativo tematico (sempre film reali)
        for variante in varianti[:3]:
            ricerca = chiama_tool("tool_cerca_film_tema_testuale", cerca_film, variante)
            tool_usati.append("cerca_film_tema_testuale")
            if contiene_errore_tool(ricerca):
                continue
            risultati.extend(ricerca[:5])
            if len(risultati) >= 15:
                break

    return normalizza_lista_film(risultati, "", generi_tmdb=generi_tmdb, limite=POOL_MASSIMO)


def _raccogli_simili(richiesta: RichiestaRicerca, generi_tmdb: list[dict[str, Any]],
                     tool_usati: list[str], warnings: list[str]) -> list[FilmCandidato]:
    if not richiesta.film_base.strip():
        warnings.append("Richiesta 'simili' senza film base.")
        return []

    risultati_base = chiama_tool("tool_cerca_film", cerca_film, richiesta.film_base.strip())
    tool_usati.append("cerca_film")
    if contiene_errore_tool(risultati_base) or not risultati_base:
        warnings.append("Film base non trovato su TMDB.")
        return []

    base = risultati_base[0]
    base_id = base.get("tmdb_id") or base.get("id")
    generi_base = {chiave_genere(g) for g in mappa_generi_da_ids(base, generi_tmdb)} - {None}
    if not generi_base:
        ids_base = set(base.get("genre_ids") or [])
        generi_base = {
            chiave_genere(str(g.get("nome") or g.get("name") or ""))
            for g in (generi_tmdb or [])
            if g.get("id") in ids_base
        } - {None}

    # il film base non deve MAI comparire nei risultati
    if base_id and int(base_id) not in richiesta.tmdb_id_da_evitare:
        richiesta.tmdb_id_da_evitare.append(int(base_id))
    if richiesta.film_base not in richiesta.titoli_da_evitare:
        richiesta.titoli_da_evitare.append(richiesta.film_base)

    risultati = chiama_tool("tool_get_film_simili", get_film_simili, base_id)
    tool_usati.append("get_film_simili")
    if contiene_errore_tool(risultati):
        warnings.append(estrai_errore_tool(risultati))
        return []

    candidati = normalizza_lista_film(risultati, "", generi_tmdb=generi_tmdb, limite=POOL_MASSIMO)
    if generi_base:
        candidati = [
            f for f in candidati
            if generi_base.intersection({chiave_genere(g) for g in f.generi})
        ]
    return candidati


def _raccogli_per_titolo(richiesta: RichiestaRicerca, generi_tmdb: list[dict[str, Any]],
                         tool_usati: list[str], warnings: list[str]) -> list[FilmCandidato]:
    query = richiesta.film_base.strip() or richiesta.testo_richiesta.strip()
    risultati = chiama_tool("tool_cerca_film", cerca_film, query)
    tool_usati.append("cerca_film")
    if contiene_errore_tool(risultati):
        warnings.append(estrai_errore_tool(risultati))
        return []
    return normalizza_lista_film(risultati, "", generi_tmdb=generi_tmdb, limite=10)


def _raccogli_popolari(richiesta: RichiestaRicerca, generi_tmdb: list[dict[str, Any]],
                       tool_usati: list[str], warnings: list[str]) -> list[FilmCandidato]:
    esclusi_ids = ids_generi_tmdb(richiesta.generi_esclusi or [], generi_tmdb)
    risultati: list[dict[str, Any]] = []
    max_pagine, soglia = _profondita_ricerca(richiesta)
    pagina_iniziale = max(1, richiesta.pagina)
    for page in range(pagina_iniziale, pagina_iniziale + max_pagine):
        pagina = chiama_tool(
            "tool_get_film_popolari", get_film_popolari,
            page=page,
            voto_min=richiesta.voto_min, voto_max=richiesta.voto_max,
            anno_min=richiesta.anno_min, anno_max=richiesta.anno_max,
            generi_esclusi_ids=esclusi_ids,
        )
        tool_usati.append(f"get_film_popolari_page_{page}")
        if contiene_errore_tool(pagina):
            warnings.append(estrai_errore_tool(pagina))
            break
        if not pagina:
            break
        risultati.extend(pagina)
        if len(risultati) >= soglia:
            break
    return normalizza_lista_film(risultati, "", generi_tmdb=generi_tmdb, limite=soglia)


def raccogli_candidati(richiesta: RichiestaRicerca, generi_tmdb: list[dict[str, Any]],
                       tool_usati: list[str], warnings: list[str]) -> list[FilmCandidato]:
    if richiesta.tipo == "simili":
        return _raccogli_simili(richiesta, generi_tmdb, tool_usati, warnings)
    if richiesta.tipo == "tema":
        return _raccogli_per_tema(richiesta, generi_tmdb, tool_usati, warnings)
    if richiesta.tipo == "titolo":
        return _raccogli_per_titolo(richiesta, generi_tmdb, tool_usati, warnings)
    if richiesta.tipo == "genere" and richiesta.generi_richiesti:
        return _raccogli_per_genere(richiesta, generi_tmdb, tool_usati, warnings)
    # generica: popolari con i vincoli espliciti (dichiarato dal conduttore)
    return _raccogli_popolari(richiesta, generi_tmdb, tool_usati, warnings)



ISTRUZIONI_RERANK = [
    "Sei il reranker di MovieAgentOS.",
    "Ricevi la richiesta dell'utente, il suo profilo, le sue memorie e una lista di film candidati REALI.",
    "Scegli TUTTI i film davvero adatti PER QUESTO UTENTE tra i candidati, fino a un massimo di 6: se gli adatti sono piu' di 6, prendi i 6 PIU' VICINI alla richiesta e ai gusti; se solo 2 lo sono, proponine 2. Non inventare mai film fuori lista.",
    "Le MEMORIE SEMANTICHE e i FILM PIACIUTI / NON PIACIUTI sono un criterio MOLTO IMPORTANTE della scelta:",
    "  - penalizza fortemente i candidati molto simili (storia, saga, tono) ai film NON piaciuti;",
    "  - favorisci i candidati affini ai film piaciuti e alle memorie positive;",
    "  - la richiesta attuale resta il vincolo primario: mai proporre film fuori dai suoi vincoli espliciti.",
    "  - la personalizzazione non deve cambiare il TONO richiesto: se l'utente chiede suspense o un genere serio, non favorire commedie solo perche' in passato gli sono piaciute.",
    "Per ogni film scelto scrivi un motivo breve in italiano semplice, riferito alla richiesta e ai gusti dell'utente.",
    "Il motivo deve essere leggibile dall'utente finale: niente termini tecnici.",
    "Scrivi anche 'commento': 2-4 frasi CONVERSAZIONALI che apriranno la risposta, come un amico che parla davvero. Reagisci a quello che l'utente ha appena detto (se ha cambiato idea commentalo con naturalezza, se ha chiesto altri film digli cosa hai pescato di nuovo), spiega in breve la logica della scelta e, se hai selezionato molti film, dillo ('te ne ho messi parecchi, cosi' scegliete insieme'). Chiudi con un aggancio naturale, non con una formula. Mai frasi robotiche tipo 'ecco alcune opzioni'.",
    "Rispondi SOLO con JSON valido: {\"commento\": \"...\", \"scelte\": [{\"tmdb_id\": <id>, \"motivo\": \"...\"}]}",
    "La qualita' e' il filtro, non un numero: includi un candidato solo se e' davvero adatto, ma NON scartare film adatti solo per restare in un numero fisso.",
]


def _testo_candidati(candidati: list[FilmCandidato]) -> str:
    righe = []
    for film in candidati:
        descrizione = " ".join(str(film.descrizione or "").split())[:180]
        righe.append(
            f"- tmdb_id={film.tmdb_id} | {film.titolo} ({film.anno}) | "
            f"generi: {', '.join(film.generi) or 'n/d'} | voto: {film.voto_medio} | {descrizione}"
        )
    return "\n".join(righe)


def rerank_llm(candidati: list[FilmCandidato], richiesta: RichiestaRicerca) -> tuple[list[FilmCandidato], bool, str]:
    """Rerank LLM del top-N. Ritorna (film scelti, llm_usato, commento
    conversazionale). In caso di errore o output non valido: top deterministico."""
    fallback = candidati[:RISULTATI_FALLBACK]
    if len(candidati) <= 1:
        return fallback, False, ""

    try:
        set_llm_api_key("Rerank")
        agente = Agent(model=crea_model_agente("rerank"), markdown=False, instructions=ISTRUZIONI_RERANK)

        profilo_parti = []
        if richiesta.profilo_generi_preferiti:
            profilo_parti.append("Generi preferiti: " + ", ".join(richiesta.profilo_generi_preferiti))
        if richiesta.profilo_segnali:
            segnali = [f"{k}={v}" for k, v in list(richiesta.profilo_segnali.items())[:10]]
            profilo_parti.append("Segnali: " + ", ".join(segnali))
        contesto = "\n".join(f"- {m}" for m in richiesta.contesto_semantico[:4])

        prompt = (
            f"RICHIESTA UTENTE:\n{richiesta.testo_richiesta or richiesta.query_tema or richiesta.film_base or 'richiesta generica'}\n\n"
            f"PROFILO UTENTE:\n{chr(10).join(profilo_parti) or 'Nessun profilo.'}\n\n"
            f"FILM PIACIUTI (thumbs up): {', '.join(richiesta.film_piaciuti[:10]) or 'nessuno'}\n"
            f"FILM NON PIACIUTI (thumbs down): {', '.join(richiesta.film_non_piaciuti[:10]) or 'nessuno'}\n\n"
            f"MEMORIE SEMANTICHE (criterio importante per la scelta):\n{contesto or 'Nessuna.'}\n\n"
            f"CANDIDATI:\n{_testo_candidati(candidati)}"
        )
        risposta = agente.run(prompt)
        contenuto = risposta.content if isinstance(risposta.content, str) else str(risposta.content)
        dati = estrai_json_da_testo(contenuto)
        scelte = dati.get("scelte") or []
        commento = str(dati.get("commento") or "").strip()[:600]

        per_id = {f.tmdb_id: f for f in candidati if f.tmdb_id is not None}
        scelti: list[FilmCandidato] = []
        for scelta in scelte:
            if not isinstance(scelta, dict):
                continue
            try:
                tmdb_id = int(scelta.get("tmdb_id"))
            except (TypeError, ValueError):
                continue
            film = per_id.get(tmdb_id)
            if film is None or film in scelti:
                continue  # l'LLM sceglie SOLO tra candidati reali
            motivo = str(scelta.get("motivo") or "").strip()
            if motivo:
                film.motivo_ricerca = motivo[:300]
            scelti.append(film)
            if len(scelti) >= RISULTATI_MASSIMI:
                break

        if scelti:
            return scelti, True, commento
        return fallback, False, ""
    except Exception as e:
        print(f"[Rerank]     errore LLM: {type(e).__name__}: {str(e)[:80]} - uso ordinamento deterministico")
        return fallback, False, ""


def _motivo_default(film: FilmCandidato, richiesta: RichiestaRicerca) -> str:
    parti = []
    generi = ", ".join(film.generi[:2])
    if generi:
        parti.append(f"è un film di {generi}")
    if isinstance(film.voto_medio, (int, float)) and film.voto_medio >= 7:
        parti.append(f"ha un ottimo voto TMDB ({film.voto_medio:.1f}/10)")
    if richiesta.film_base:
        parti.append(f"è affine a {richiesta.film_base}")
    return (", ".join(parti) + ".") if parti else "corrisponde alla tua richiesta."


def RicercaAgent(richiesta: RichiestaRicerca) -> SearchAgentOutput:
    start = time.perf_counter()
    tool_usati: list[str] = []
    warnings: list[str] = []
    print(f'[Search]     avvio -> tipo={richiesta.tipo} | "{(richiesta.testo_richiesta or "")[:80]}"')

    try:
        # candidati salvati al turno precedente (domanda di raffinamento):
        # la risposta dell'utente filtra QUESTI film, senza rifare la raccolta
        candidati: list[FilmCandidato] = []
        pool_completo: list[FilmCandidato] = []
        salvati: list[FilmCandidato] = []
        vincoli_allentati = False
        if richiesta.candidati_salvati and richiesta.pagina <= 1:
            salvati = [f.model_copy(deep=True) for f in richiesta.candidati_salvati]
            pool_completo = filtra_e_ordina(salvati, richiesta)
            if pool_completo:
                candidati = salvati
                print(f"[Search]     riparto dai {len(salvati)} candidati salvati al turno precedente: {len(pool_completo)} rispettano i nuovi vincoli (nessuna nuova ricerca TMDB)")
            else:
                # in FILTRAZIONE non si rifa' la ricerca: se l'ultimo filtro
                # azzera i salvati, si mostrano i piu' vicini alla richiesta
                # (rispettavano tutti i vincoli precedenti), dichiarandolo
                pool_completo = piu_vicini_alla_richiesta(salvati, richiesta)
                if pool_completo:
                    candidati = salvati
                    vincoli_allentati = True
                    print(f"[Search]     0 candidati salvati con TUTTI i filtri: niente nuova ricerca, mostro i {min(len(pool_completo), CANDIDATI_RERANK)} piu' vicini (lo dichiaro all'utente)")

        if not pool_completo:
            generi_tmdb = get_generi_film_cached()
            if contiene_errore_tool(generi_tmdb):
                generi_tmdb = []
                warnings.append("Non sono riuscito a leggere i generi TMDB; continuo con i dati disponibili.")
            candidati = raccogli_candidati(richiesta, generi_tmdb, tool_usati, warnings)
            pool_completo = filtra_e_ordina(candidati, richiesta)
        pool = pool_completo[:CANDIDATI_RERANK]

        if not pool:
            durata = time.perf_counter() - start
            print(f"[Search]     0 film trovati - ({durata:.1f}s)")
            return SearchAgentOutput(
                success=False,
                film_candidati=[],
                errors=["Nessun film reale rispetta tutti i vincoli espliciti della richiesta."],
                log_notes={
                    "richiesta": richiesta.model_dump(exclude={"candidati_salvati"}),
                    "tool_usati": tool_usati,
                    "warnings": warnings,
                    "candidati_raccolti": len(candidati),
                },
            )

        film_finali, rerank_usato, commento = rerank_llm(pool, richiesta)
        for film in film_finali:
            if not (film.motivo_ricerca or "").strip():
                film.motivo_ricerca = _motivo_default(film, richiesta)

        durata = time.perf_counter() - start
        print(f"[Search]     {len(film_finali)} film scelti su {len(pool)} candidati (rerank LLM: {'sì' if rerank_usato else 'no'}) ({durata:.1f}s)")
        return SearchAgentOutput(
            success=True,
            film_candidati=film_finali,
            pool_candidati=pool_completo,
            errors=[],
            log_notes={
                "richiesta": richiesta.model_dump(exclude={"candidati_salvati"}),
                "tool_usati": tool_usati,
                "warnings": warnings,
                "candidati_raccolti": len(candidati),
                "pool_valido": len(pool_completo),
                "pool_precedente": len(salvati),
                "vincoli_allentati": vincoli_allentati,
                "pool_rerank": len(pool),
                "rerank_llm_usato": rerank_usato,
                "ricerca_generica": richiesta.tipo == "generica",
                "commento_conversazionale": commento,
            },
        )
    except Exception as e:
        durata = time.perf_counter() - start
        print(f"[Search] errore: {type(e).__name__}: {str(e)[:100]} ({durata:.1f}s)")
        return SearchAgentOutput(
            success=False,
            film_candidati=[],
            errors=[f"Errore durante esecuzione SearchAgent: {type(e).__name__}: {e}"],
            log_notes={"richiesta": richiesta.model_dump(), "tool_usati": tool_usati, "warnings": warnings},
        )
