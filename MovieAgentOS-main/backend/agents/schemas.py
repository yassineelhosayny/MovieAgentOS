from typing import Any, Optional

from pydantic import BaseModel, Field


class FilmCandidato(BaseModel):
    tmdb_id: int | None = None
    titolo: str = ""
    anno: Optional[int | str] = None
    descrizione: Optional[str] = None
    generi: list[str] = Field(default_factory=list)
    voto_medio: Optional[float] = None
    poster_path: Optional[str] = None
    motivo_ricerca: Optional[str] = None

    numero_voti: Optional[int] = None


class RichiestaRicerca(BaseModel):
    """Contratto strutturato tra orchestrator e SearchAgent.

    Nessun dato passa come testo da ri-parsare: il conduttore compila questo
    oggetto e il SearchAgent lo esegue. E' anche lo stato di sessione: una
    CONTINUAZIONE riparte da qui (stessa richiesta, pagina successiva,
    esclusioni aggiornate), una RAFFINAZIONE vi aggiunge vincoli.
    """

    tipo: str = "generica"  # genere | tema | simili | titolo | generica
    testo_richiesta: str = ""

    generi_richiesti: list[str] = Field(default_factory=list)
    generi_esclusi: list[str] = Field(default_factory=list)
    voto_min: float | None = None
    voto_max: float | None = None
    anno_min: int | None = None
    anno_max: int | None = None

    film_base: str = ""       # per tipo=simili / titolo
    query_tema: str = ""      # per tipo=tema

    titoli_da_evitare: list[str] = Field(default_factory=list)
    tmdb_id_da_evitare: list[int] = Field(default_factory=list)

    # preferenze morbide del profilo: influenzano il rerank, mai i vincoli
    profilo_generi_preferiti: list[str] = Field(default_factory=list)
    profilo_segnali: dict[str, float] = Field(default_factory=dict)
    # memorie semantiche ChromaDB + feedback thumbs (SQLite): criterio
    # molto importante nella SCELTA tra candidati validi (rerank); le
    # esclusioni dure restano per tmdb_id
    contesto_semantico: list[str] = Field(default_factory=list)
    film_piaciuti: list[str] = Field(default_factory=list)
    film_non_piaciuti: list[str] = Field(default_factory=list)
    generi_graditi_feedback: list[str] = Field(default_factory=list)
    generi_sgraditi_feedback: list[str] = Field(default_factory=list)

    pagina: int = 1

    # candidati salvati al turno precedente, quando la ricerca ha trovato
    # troppi film adatti e si e' preferito fare una domanda: la risposta
    # dell'utente filtra QUESTI film, senza ripartire da zero
    candidati_salvati: list[FilmCandidato] = Field(default_factory=list)

    def numero_vincoli(self) -> int:
        """Quanti vincoli espliciti ha la richiesta: serve a decidere se,
        con un pool grande, conviene chiedere all'utente piu' informazioni."""
        vincoli = 0
        if self.generi_richiesti:
            vincoli += 1
        if self.generi_esclusi:
            vincoli += 1
        if self.query_tema:
            vincoli += 1
        if self.film_base:
            vincoli += 1
        if self.voto_min is not None or self.voto_max is not None:
            vincoli += 1
        if self.anno_min is not None or self.anno_max is not None:
            vincoli += 1
        return vincoli

    def ha_vincoli_specifici(self) -> bool:
        return bool(
            self.generi_richiesti
            or self.query_tema
            or self.film_base
            or self.voto_min is not None
            or self.voto_max is not None
            or self.anno_min is not None
            or self.anno_max is not None
        )


class DecisioneRouter(BaseModel):
    """Output del router LLM del conduttore."""

    azione: str = "errore"
    # raccomandazione | chiarimento | conversazione_film | domanda_memoria |
    # aggiornamento_memoria | fuori_dominio | errore
    tipo_richiesta: str = "NUOVA"  # NUOVA | CONTINUAZIONE | RAFFINAZIONE
    segna_visti_recenti: bool = False
    aggiorna_memoria: bool = False
    risposta_diretta: str = ""
    # titolo del film che l'utente dice di aver SCELTO tra quelli consigliati
    # ("ho trovato il film giusto, X"): va salvato come visto e gradito
    film_scelto: str = ""
    ricerca: RichiestaRicerca | None = None


class ConduttoreOutput(BaseModel):
    success: bool = False
    azione: str = ""
    risposta: str = ""
    film_consigliati: list[FilmCandidato] = Field(default_factory=list)
    memoria_aggiornata: bool = False
    titolo_chat: str | None = None
    # stato di sessione: l'ultima ricerca eseguita (o parziale, se chiarimento)
    richiesta_eseguita: RichiestaRicerca | None = None
    errors: list[str] = Field(default_factory=list)
    log_notes: dict[str, Any] = Field(default_factory=dict)


class SearchAgentOutput(BaseModel):
    success: bool = False
    film_candidati: list[FilmCandidato] = Field(default_factory=list)
    # pool completo dei candidati validi (oltre ai film scelti): il conduttore
    # lo puo' salvare nello stato di sessione per farlo filtrare dall'utente
    pool_candidati: list[FilmCandidato] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    log_notes: dict[str, Any] = Field(default_factory=dict)


class MemoryAgentOutput(BaseModel):
    success: bool = False
    generi_preferiti: list[str] = Field(default_factory=list)
    generi_da_evitare: list[str] = Field(default_factory=list)
    summary_testuale: str = ""
    errors: list[str] = Field(default_factory=list)

    id_utente: int | None = None
    query_originale: str = ""
    preferenze_json: dict[str, Any] = Field(default_factory=dict)
    memorie_rilevanti: list[dict[str, Any]] = Field(default_factory=list)
    log_notes: dict[str, Any] = Field(default_factory=dict)


class PreferenzeEstratteMemoria(BaseModel):
    """informazione estratti dalla memoria."""

    is_memoria: bool = False
    tipo_memoria: str = "non_memoria"
    generi_preferiti: list[str] = Field(default_factory=list)
    generi_da_evitare: list[str] = Field(default_factory=list)
    preferenze_positive: list[str] = Field(default_factory=list)
    preferenze_negative: list[str] = Field(default_factory=list)
    vincoli: list[str] = Field(default_factory=list)
    preferenze_json: dict[str, float] = Field(default_factory=dict)
    summary_memoria: str = ""
    confidence: float = 0.0
