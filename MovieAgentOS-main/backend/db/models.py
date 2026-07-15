from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.db.database import Base
import json


class Utente(Base):
    __tablename__ = "utenti"
    id_utente = Column(Integer, primary_key=True, index=True)

    # Identificativo univoco fornito dal provider di autenticazione.
    # abbiamo usato Auth0 con Google,
    auth_sub = Column(String, unique=True, nullable=False, index=True)

    email = Column(String, unique=True, nullable=False, index=True)
    nome = Column(String, nullable=False)
    foto_url = Column(String, nullable=True)

    provider = Column(String, default="auth0")

    creato_il = Column(DateTime(timezone=True), server_default=func.now())
    ultimo_accesso = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    sessioni = relationship(
        "Sessione",
        back_populates="utente",
        cascade="all, delete-orphan"
    )

    feedback = relationship(
        "Feedback",
        back_populates="utente",
        cascade="all, delete-orphan"
    )

    profilo = relationship(
        "ProfiloUtente",
        back_populates="utente",
        uselist=False,
        cascade="all, delete-orphan"
    )


class Sessione(Base):
    __tablename__ = "sessioni"

    id_sessione = Column(Integer, primary_key=True, index=True)

    id_utente = Column(
        Integer,
        ForeignKey("utenti.id_utente", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    titolo = Column(String, default="Nuova sessione")

    creata_il = Column(DateTime(timezone=True), server_default=func.now())

    aggiornata_il = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    utente = relationship(
        "Utente",
        back_populates="sessioni"
    )

    messaggi = relationship(
        "Messaggio",
        back_populates="sessione",
        cascade="all, delete-orphan",
        order_by="Messaggio.inviato_il"
    )


class Messaggio(Base):
    __tablename__ = "messaggi"

    id_messaggio = Column(Integer, primary_key=True, index=True)
    id_sessione = Column(
        Integer,
        ForeignKey("sessioni.id_sessione", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Valori previsti lato CRUD:
    # "user", "assistant", "system", "tool"
    ruolo = Column(String, nullable=False)

    contenuto = Column(Text, nullable=False)

    inviato_il = Column(DateTime(timezone=True), server_default=func.now())

    sessione = relationship(
        "Sessione",
        back_populates="messaggi"
    )

    # Un messaggio assistant può consigliare più film.
    film_consigliati = relationship(
        "FilmConsigliato",
        back_populates="messaggio",
        cascade="all, delete-orphan"
    )


class FilmConsigliato(Base):
    __tablename__ = "film_consigliati"

    id_consiglio = Column(Integer, primary_key=True, index=True)

    id_messaggio = Column(
        Integer,
        ForeignKey("messaggi.id_messaggio", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    id_utente = Column(
        Integer,
        ForeignKey("utenti.id_utente", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    id_tmdb = Column(Integer, nullable=False, index=True)

    titolo = Column(String, nullable=False)
    titolo_originale = Column(String, nullable=True)

    anno = Column(Integer, nullable=True)

    descrizione = Column(Text, nullable=True)

    generi = Column(Text, default="[]")

    poster_path = Column(String, nullable=True)
    poster_url = Column(String, nullable=True)

    voto_medio = Column(String, nullable=True)
    numero_voti = Column(Integer, nullable=True)

    motivo_raccomandazione = Column(Text, nullable=True)

    link_streaming = Column(Text, default="{}")

    consigliato_il = Column(DateTime(timezone=True), server_default=func.now())

    messaggio = relationship(
        "Messaggio",
        back_populates="film_consigliati"
    )

    utente = relationship(
        "Utente"
    )

    feedback = relationship(
        "Feedback",
        back_populates="film_consigliato",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def get_generi(self):
        return json.loads(self.generi) if self.generi else []

    def set_generi(self, generi_list):
        self.generi = json.dumps(generi_list)

    def get_link_streaming(self):
        return json.loads(self.link_streaming) if self.link_streaming else {}

    def set_link_streaming(self, link_dict):
        self.link_streaming = json.dumps(link_dict)


class Feedback(Base):
    __tablename__ = "feedback"

    id_feedback = Column(Integer, primary_key=True, index=True)

    id_utente = Column(
        Integer,
        ForeignKey("utenti.id_utente", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    id_consiglio = Column(
        Integer,
        ForeignKey("film_consigliati.id_consiglio", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )

    # Scala 1-10:
    # >= 7 feedback positivo
    # < 7 feedback negativo
    voto = Column(Integer, nullable=False)

    commento = Column(Text, nullable=True)

    dato_il = Column(DateTime(timezone=True), server_default=func.now())

    utente = relationship(
        "Utente",
        back_populates="feedback"
    )

    film_consigliato = relationship(
        "FilmConsigliato",
        back_populates="feedback"
    )


class FilmVisto(Base):
    """Film che l'utente ha gia' visto: fonte precisa SQLite.

    Si popola da un feedback (thumbs up/down: qualsiasi feedback implica che
    il film e' stato visto) o da una dichiarazione esplicita in chat
    ("ho gia' visto questi"). Questi film non vanno mai riproposti.
    """

    __tablename__ = "film_visti"

    id_visto = Column(Integer, primary_key=True, index=True)

    id_utente = Column(
        Integer,
        ForeignKey("utenti.id_utente", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    id_tmdb = Column(Integer, nullable=False, index=True)

    titolo = Column(String, nullable=True)

    # "feedback" (thumbs) oppure "dichiarato" (detto in chat)
    fonte = Column(String, default="dichiarato")

    # True/False se noto dal feedback, None se solo dichiarato visto
    gradito = Column(Integer, nullable=True)  # 1 = piaciuto, 0 = non piaciuto

    visto_il = Column(DateTime(timezone=True), server_default=func.now())

    utente = relationship("Utente")


class ProfiloUtente(Base):
    __tablename__ = "profilo_utente"

    id_utente = Column(
        Integer,
        ForeignKey("utenti.id_utente", ondelete="CASCADE"),
        primary_key=True
    )

    generi_preferiti = Column(Text, default="[]")
    generi_da_evitare = Column(Text, default="[]")

    # Preferenze strutturate più flessibili.
    # Esempio:
    # {
    #   "sci_fi": 0.8,
    #   "horror": 0.1,
    #   "film_lenti": 0.7,
    #   "azione": 0.3
    # }
    preferenze_json = Column(Text, default="{}")

    # Riassunto naturale del profilo utente, aggiornabile dal MemoryAgent.
    summary_testuale = Column(Text, nullable=True)

    aggiornato_il = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    utente = relationship(
        "Utente",
        back_populates="profilo"
    )

    def get_generi_preferiti(self):
        return json.loads(self.generi_preferiti) if self.generi_preferiti else []

    def set_generi_preferiti(self, generi_list):
        self.generi_preferiti = json.dumps(generi_list)

    def get_generi_da_evitare(self):
        return json.loads(self.generi_da_evitare) if self.generi_da_evitare else []

    def set_generi_da_evitare(self, generi_list):
        self.generi_da_evitare = json.dumps(generi_list)

    def get_preferenze(self):
        return json.loads(self.preferenze_json) if self.preferenze_json else {}

    def set_preferenze(self, preferenze_dict):
        self.preferenze_json = json.dumps(preferenze_dict)
