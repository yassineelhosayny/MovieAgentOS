import streamlit as s
import requests


def get_TMDB_API_KEY()->str:
    """
    Restiuce la chiave API di TMDB locale, salvata nel file .toml
    """
    try:
        api_key = s.secrets["TMDB_API_KEY"]
    except Exception as e:
        raise ValueError(f"Chiave API TMDB mancante: {e}")

    if not api_key:
        raise ValueError("Chiave API TMDB vuota.")
    return api_key



def cerca_film(nomeFilm : str) -> list[dict]:
    """
    tool per cercare un film su TMDB usando il nome del film o una query in genere
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]
    
    if not nomeFilm or not nomeFilm.strip():
        return [{"errore": "Query mancante"}]
    
    url="https://api.themoviedb.org/3/search/movie"
    param = {
        "api_key": api_tmdb,
        "query": nomeFilm,
        "language": "it-IT",
        "include_adult": False,
        "page": 1
    }
    #faccio una richiesta a TMDB
    try:
        response = requests.get(url,params=param,timeout=10) #serve per fare chiamate HTTP.
        response.raise_for_status() # invia la chiamata a except se gli informazione sono errati
    except requests.RequestException as e:
        return [{"errore":f"Errore durante la richiesta a TMDB: \n{e}"}]   
     
    #se tutto andava bene, faccio un filtr al resultati
    result = response.json()
    Films_Resultati = result.get("results",[])

    film_puliti =[]
    #prendo solo i primi 10
    for f in Films_Resultati[:10]:
        data_uscita = f.get("release_date") or ""
        anno = data_uscita[:4] if data_uscita else "Sconosciuto"

        film_puliti.append({
            "tmdb_id": f.get("id"),
            "titolo": f.get("title"),
            "titolo_originale": f.get("original_title"),
            "anno": anno,
            "descrizione": f.get("overview"),
            "voto_medio": f.get("vote_average"),
            "numero_voti": f.get("vote_count"),
            "popolarita": f.get("popularity"),
            "poster_path": f.get("poster_path"),
            "lingua_originale": f.get("original_language"),
            "genre_ids": f.get("genre_ids", [])
        })
    return film_puliti



def cerca_film_per_genere(
    genre_id: int,
    page: int = 1,
    anno_min: int | None = None,
    anno_max: int | None = None,
    voto_min: float | None = None,
    voto_max: float | None = None,
) -> list[dict]:
    """
    Tool per cercare film su TMDB usando l'id di un genere.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    if genre_id is None:
        return [{"errore": "Genre id mancante."}]

    url = "https://api.themoviedb.org/3/discover/movie"

    param = {
        "api_key": api_tmdb,
        "with_genres": genre_id,
        "language": "it-IT",
        "include_adult": False,
        "include_video": False,
        "sort_by": "popularity.desc",
        "vote_count.gte": 50,
        "page": page if page and page > 0 else 1
    }

    if anno_min:
        param["primary_release_date.gte"] = f"{anno_min}-01-01"

    if anno_max:
        param["primary_release_date.lte"] = f"{anno_max}-12-31"

    if voto_min is not None:
        param["vote_average.gte"] = voto_min

    if voto_max is not None:
        param["vote_average.lte"] = voto_max

    try:
        response = requests.get(url, params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la richiesta a TMDB: {e}"}]

    result = response.json()
    risultati_film = result.get("results", [])

    film_puliti = []

    for f in risultati_film:
        if not f.get("overview"):
            continue

        data_uscita = f.get("release_date") or ""
        anno = data_uscita[:4] if data_uscita else "Sconosciuto"

        film_puliti.append({
            "tmdb_id": f.get("id"),
            "titolo": f.get("title"),
            "titolo_originale": f.get("original_title"),
            "anno": anno,
            "descrizione": f.get("overview"),
            "voto_medio": f.get("vote_average"),
            "numero_voti": f.get("vote_count"),
            "popolarita": f.get("popularity"),
            "poster_path": f.get("poster_path"),
            "lingua_originale": f.get("original_language"),
            "genre_ids": f.get("genre_ids", [])
        })

        if len(film_puliti) == 10:
            break

    return film_puliti


def cerca_film_avanzato(
    query: str = "",
    generi_inclusi: list[int] | None = None,
    generi_esclusi: list[int] | None = None,
    voto_min: float | None = None,
    anno_min: int | None = None,
    anno_max: int | None = None,
    pagine: int = 2,
) -> list[dict]:
    """
    Tool avanzato che combina ricerca testuale e filtri discover in una sola chiamata.
    Usa 'without_genres' di TMDB per escludere generi a livello API — più preciso
    e veloce del filtraggio Python post-fetch.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    url = "https://api.themoviedb.org/3/discover/movie"
    param: dict = {
        "api_key": api_tmdb,
        "language": "it-IT",
        "include_adult": False,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 100,
    }

    if generi_inclusi:
        param["with_genres"] = ",".join(str(g) for g in generi_inclusi)

    if generi_esclusi:
        param["without_genres"] = ",".join(str(g) for g in generi_esclusi)

    if voto_min is not None:
        param["vote_average.gte"] = voto_min

    if anno_min:
        param["primary_release_date.gte"] = f"{anno_min}-01-01"

    if anno_max:
        param["primary_release_date.lte"] = f"{anno_max}-12-31"

    film_puliti: list[dict] = []

    for page in range(1, pagine + 1):
        param["page"] = page
        try:
            response = requests.get(url, params=param, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            if page == 1:
                return [{"errore": f"Errore TMDB discover: {e}"}]
            break

        for f in response.json().get("results", []):
            if not f.get("overview"):
                continue
            data_uscita = f.get("release_date") or ""
            anno = data_uscita[:4] if data_uscita else "Sconosciuto"
            film_puliti.append({
                "tmdb_id": f.get("id"),
                "titolo": f.get("title"),
                "titolo_originale": f.get("original_title"),
                "anno": anno,
                "descrizione": f.get("overview"),
                "voto_medio": f.get("vote_average"),
                "numero_voti": f.get("vote_count"),
                "popolarita": f.get("popularity"),
                "poster_path": f.get("poster_path"),
                "lingua_originale": f.get("original_language"),
                "genre_ids": f.get("genre_ids", []),
            })

        if len(film_puliti) >= 20:
            break

    return film_puliti


def get_movie_details(film_id: int) -> dict:
    """
    Recupera dettagli e informazioni completi di un film da TMDB,
    tipo generi con nome, durata, trama completa, data uscita, voto,
    lingua e poster.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return {"errore": str(e)}

    if film_id is None or film_id <= 0:
        return {"errore": "Film id mancante o non valido."}

    url = f"https://api.themoviedb.org/3/movie/{film_id}"

    param = {
        "api_key": api_tmdb,
        "language": "it-IT"
    }

    try:
        response = requests.get(url, params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"errore": f"Errore durante la richiesta a TMDB: {e}"}

    film = response.json()

    generi = []
    for g in film.get("genres", []):
        generi.append(g.get("name"))

    return {
        "tmdb_id": film.get("id"),
        "titolo": film.get("title"),
        "titolo_originale": film.get("original_title"),
        "descrizione": film.get("overview"),
        "generi": generi,
        "durata_minuti": film.get("runtime"),
        "data_uscita": film.get("release_date"),
        "voto_medio": film.get("vote_average"),
        "numero_voti": film.get("vote_count"),
        "popolarita": film.get("popularity"),
        "lingua_originale": film.get("original_language"),
        "poster_path": film.get("poster_path"),
        "tagline": film.get("tagline"),
        "stato": film.get("status"),
        "budget": film.get("budget"),
        "incasso": film.get("revenue")
    }


def get_generi_film() -> list[dict]:
    """
    Recupera la lista dei generi disponibili su TMDB.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    url = "https://api.themoviedb.org/3/genre/movie/list"

    param = {
        "api_key": api_tmdb,
        "language": "it-IT"
    }
    try:
        response = requests.get(url, params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la richiesta a TMDB: {e}"}]

    result = response.json()
    generi = result.get("genres", [])

    generi_puliti = []

    for genere in generi:
        generi_puliti.append({
            "id": genere.get("id"),
            "nome": genere.get("name")
        })

    return generi_puliti


def get_film_simili(film_id: int) -> list[dict]:
    """
    Recupera una lista di film simili a un film specifico usando film_id in TMDB.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    if film_id is None or film_id <= 0:
        return [{"errore": "Film id mancante o non valido."}]

    url = f"https://api.themoviedb.org/3/movie/{film_id}/recommendations"

    param = {
        "api_key": api_tmdb,
        "language": "it-IT",
        "page": 1
    }

    try:
        response = requests.get(url, params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la richiesta a TMDB: {e}"}]

    result = response.json()
    risultati_film = result.get("results", [])

    film_puliti = []

    for f in risultati_film:
        if not f.get("overview"):
            continue

        data_uscita = f.get("release_date") or ""
        anno = data_uscita[:4] if data_uscita else "Sconosciuto"

        film_puliti.append({
            "tmdb_id": f.get("id"),
            "titolo": f.get("title"),
            "titolo_originale": f.get("original_title"),
            "anno": anno,
            "descrizione": f.get("overview"),
            "voto_medio": f.get("vote_average"),
            "numero_voti": f.get("vote_count"),
            "popolarita": f.get("popularity"),
            "poster_path": f.get("poster_path"),
            "lingua_originale": f.get("original_language"),
            "genre_ids": f.get("genre_ids", [])
        })

        if len(film_puliti) == 10:
            break

    return film_puliti


#serve nel tipo di domanda tipo:- consigliami qualcosa - non so cosa guardare
def get_film_popolari(page: int = 1) -> list[dict]:
    """
    Recupera una lista di film popolari da TMDB.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    url = "https://api.themoviedb.org/3/movie/popular"

    param = {
        "api_key": api_tmdb,
        "language": "it-IT",
        "page": page if page and page > 0 else 1,
        "region": "IT"
    }

    try:
        response = requests.get(url, params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la richiesta a TMDB: {e}"}]

    result = response.json()
    risultati_film = result.get("results", [])

    film_puliti = []

    for f in risultati_film:
        if not f.get("overview"):
            continue

        data_uscita = f.get("release_date") or ""
        anno = data_uscita[:4] if data_uscita else "Sconosciuto"

        film_puliti.append({
            "tmdb_id": f.get("id"),
            "titolo": f.get("title"),
            "titolo_originale": f.get("original_title"),
            "anno": anno,
            "descrizione": f.get("overview"),
            "voto_medio": f.get("vote_average"),
            "numero_voti": f.get("vote_count"),
            "popolarita": f.get("popularity"),
            "poster_path": f.get("poster_path"),
            "lingua_originale": f.get("original_language"),
            "genre_ids": f.get("genre_ids", [])
        })

        if len(film_puliti) == 10:
            break

    return film_puliti

def get_dove_guardarlo(film_id: int) -> list[dict]:
    """
    Recupera le piattaforme dove guardare un film in Italia usando il suo id TMDB.
    """
    try:
        api_tmdb = get_TMDB_API_KEY()
    except ValueError as e:
        return [{"errore": str(e)}]

    if film_id is None or film_id <= 0:
        return [{"errore": "Film id mancante o non valido."}]

    url = f"https://api.themoviedb.org/3/movie/{film_id}/watch/providers"

    param = {
        "api_key": api_tmdb
    }

    try:
        response = requests.get(url, params=param, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return [{"errore": f"Errore durante la richiesta a TMDB: {e}"}]

    result = response.json()
    risultati = result.get("results", {})
    italia = risultati.get("IT", {})

    if not italia:
        return [{"messaggio": "Nessuna piattaforma disponibile per l'Italia."}]

    provider_puliti = []

    for tipo in ["flatrate", "rent", "buy"]:
        piattaforme = italia.get(tipo, [])

        for p in piattaforme:
            provider_puliti.append({
                "tipo": tipo,
                "nome": p.get("provider_name"),
                "provider_id": p.get("provider_id"),
                "logo_path": p.get("logo_path"),
                "display_priority": p.get("display_priority")
            })

    if not provider_puliti:
        return [{"messaggio": "Nessuna piattaforma streaming, noleggio o acquisto trovata per l'Italia."}]

    return provider_puliti


#riassunto
# get_TMDB_API_KEY() -> recupera la chiave API TMDB dal file secrets.toml
# cerca_film(nomeFilm) -> cerca film su TMDB usando un titolo o una query testuale
# cerca_film_per_genere(genre_id) -> cerca film appartenenti a un genere specifico usando l'id TMDB del genere
# get_movie_details(film_id) -> recupera informazioni complete di un singolo film, come generi, durata, descrizione, voto e poster
# get_generi_film() -> recupera la lista dei generi disponibili su TMDB con id e nome
# get_film_simili(film_id) -> recupera film simili a un film specifico usando il suo id TMDB
# get_film_popolari() -> recupera film popolari, utile quando l’utente fa una richiesta vaga
# get_dove_guardarlo(film_id) -> recupera le piattaforme dove guardare il film in Italia, se disponibili
# cerca_film_avanzato(query, generi_inclusi, generi_esclusi, voto_min, anno_min, anno_max)
# -> tool unificato con esclusione generi nativa TMDB (without_genres)

