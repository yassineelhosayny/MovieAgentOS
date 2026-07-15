# backend/db/crud.py

from datetime import datetime
from typing import List, Dict, Optional, Any

from sqlalchemy.orm import Session

from backend.db import models


# ===================== UTENTI =====================

def ottieniCrea_utente(db: Session,email: str,nome: str,auth_sub: str,foto_url: str | None = None,provider: str = "auth0") -> int:
    """
    Passi:
    1. Cerca nel DB un utente con la stessa email.
    2. Se esiste:
       - Aggiorna il campo 'ultimo_accesso' con la data/ora corrente.
       - Esegui commit.
       - Restituisci id_utente.
    3. Se non esiste:
       - Crea un nuovo record Utente con i dati forniti.
       - Aggiungi alla sessione e fai commit.
       - Crea un profilo vuoto per questo utente (ProfiloUtente).
       - Fai un secondo commit.
       - Restituisci id_utente.
    """
    db_user = db.query(models.Utente).filter(models.Utente.email == email).first()

    if db_user:
        db_user.ultimo_accesso = datetime.utcnow()
        db_user.nome = nome
        db_user.foto_url = foto_url
        db_user.auth_sub = auth_sub
        db_user.provider = provider
        db.commit()
        return db_user.id_utente

    nuovo_utente = models.Utente(
                                email=email,
                                nome=nome,
                                auth_sub=auth_sub,
                                foto_url=foto_url,
                                provider=provider,
                                creato_il=datetime.utcnow(),
                                ultimo_accesso=datetime.utcnow()
                            )

    db.add(nuovo_utente)
    db.commit()
    db.refresh(nuovo_utente)

    profilo = models.ProfiloUtente(
                                    id_utente=nuovo_utente.id_utente,
                                    generi_preferiti="[]",
                                    generi_da_evitare="[]",
                                    preferenze_json="{}",
                                    summary_testuale=None,
                                    aggiornato_il=datetime.utcnow()
                                )

    db.add(profilo)
    db.commit()

    return nuovo_utente.id_utente


# ===================== SESSIONI =====================

def crea_sessione(db: Session,id_utente: int, titolo: str = "Nuova sessione") -> int:
    """
    Passi:
    1. Crea un oggetto Sessione con id_utente e titolo.
    2. Aggiungilo alla sessione DB, fai commit e refresh.
    3. Restituisci id_sessione.
    """  
    nuova_sessione = models.Sessione(
            id_utente=id_utente,
            titolo=titolo,
            creata_il=datetime.utcnow(),
            aggiornata_il=datetime.utcnow()
    )

    db.add(nuova_sessione)
    db.commit()
    db.refresh(nuova_sessione)

    return nuova_sessione.id_sessione


def ottieni_sessioni_utente(db: Session, id_utente: int) -> List[Dict]:
    """
    Passi:
    1. Query per tutte le sessioni dell'utente, ordinate per creata_il decrescente.
    2. Per ogni sessione costruisci un dizionario con id_sessione, titolo, creata_il.
    3. Restituisci la lista.
    """
    sessioni = db.query(models.Sessione).filter(
        models.Sessione.id_utente == id_utente
    ).order_by(models.Sessione.aggiornata_il.desc()).all()

    return [
        {
            "id_sessione": s.id_sessione,
            "id_utente": s.id_utente,
            "titolo": s.titolo,
            "creata_il": s.creata_il,
            "aggiornata_il": s.aggiornata_il
        }
        for s in sessioni
    ]


def rinomina_sessione(db: Session, id_sessione: int, titolo: str) -> bool:
    """
    Passi:
    1. Cerca la sessione per id.
    2. Se esiste, cambia il campo titolo e fai commit.
    3. Restituisci True se modificata.
    """
    db_sessione = db.query(models.Sessione).filter(
        models.Sessione.id_sessione == id_sessione
    ).first()

    if not db_sessione:
        return False

    db_sessione.titolo = titolo
    db_sessione.aggiornata_il = datetime.utcnow()
    db.commit()

    return True


def elimina_sessione(db: Session, id_sessione: int) -> bool:
    """
    Passi:
    1. Cerca la sessione per id.
    2. Se esiste, cancellala e fai commit.
    3. Restituisci True/False.
    """
    db_sessione = db.query(models.Sessione).filter(
        models.Sessione.id_sessione == id_sessione
    ).first()

    if not db_sessione:
        return False

    db.delete(db_sessione)
    db.commit()

    return True


# ===================== MESSAGGI =====================

def salva_messaggio(db: Session,id_sessione: int,ruolo: str,contenuto: str) -> int:
    """
    Passi:
    1. Crea un nuovo Messaggio con i campi forniti.
    2. Aggiungi, commit, refresh.
    3. Restituisci id_messaggio.
    """
    ruoli_validi = {"user", "assistant", "system", "tool"}

    if ruolo not in ruoli_validi:
        raise ValueError(f"Ruolo messaggio non valido: {ruolo}")

    nuovo_messaggio = models.Messaggio(
        id_sessione=id_sessione,
        ruolo=ruolo,
        contenuto=contenuto,
        inviato_il=datetime.utcnow()
    )

    db.add(nuovo_messaggio)

    sessione = db.query(models.Sessione).filter(
        models.Sessione.id_sessione == id_sessione
    ).first()

    if sessione:
        sessione.aggiornata_il = datetime.utcnow()

    db.commit()
    db.refresh(nuovo_messaggio)

    return nuovo_messaggio.id_messaggio


def ottieni_messaggi_sessione(db: Session, id_sessione: int) -> List[Dict]:
    """
    Passi:
    1. Query su Messaggio filtrando per id_sessione, ordine cronologico.
    2. Per ogni messaggio costruisci dizionario con id, ruolo, contenuto, inviato_il.
    3. Restituisci lista.
    """    
    messaggi = db.query(models.Messaggio).filter(
        models.Messaggio.id_sessione == id_sessione
    ).order_by(models.Messaggio.inviato_il.asc()).all()

    return [
        {
            "id_messaggio": m.id_messaggio,
            "id_sessione": m.id_sessione,
            "ruolo": m.ruolo,
            "contenuto": m.contenuto,
            "inviato_il": m.inviato_il
        }
        for m in messaggi
    ]


def ottieni_profilo(db: Session, id_utente: int) -> Optional[Dict]:
    """
    Passi:
    1. Query su ProfiloUtente con quell'id_utente.
    2. Se non esiste, restituisci None.
    3. Altrimenti costruisci dizionario con:
       id_utente, generi_preferiti (lista Python), generi_da_evitare, aggiornato_il.
    """
    db_profilo = db.query(models.ProfiloUtente).filter(
        models.ProfiloUtente.id_utente == id_utente).first()

    if not db_profilo:
        return None

    return {
        "id_utente": db_profilo.id_utente,
        "generi_preferiti": db_profilo.get_generi_preferiti(),
        "generi_da_evitare": db_profilo.get_generi_da_evitare(),
        "preferenze": db_profilo.get_preferenze(),
        "summary_testuale": db_profilo.summary_testuale,
        "aggiornato_il": db_profilo.aggiornato_il
    }


def crea_profilo_se_non_esiste(db: Session, id_utente: int) -> None:
    profilo = db.query(models.ProfiloUtente).filter(
        models.ProfiloUtente.id_utente == id_utente
    ).first()

    if profilo:
        return

    nuovo_profilo = models.ProfiloUtente(
        id_utente=id_utente,
        generi_preferiti="[]",
        generi_da_evitare="[]",
        preferenze_json="{}",
        summary_testuale=None,
        aggiornato_il=datetime.utcnow()
    )

    db.add(nuovo_profilo)
    db.commit()


def aggiorna_profilo(db: Session,id_utente: int,generi_preferiti: List[str] | None = None,generi_da_evitare: List[str] | None = None,preferenze: Dict[str, Any] | None = None, summary_testuale: str | None = None) -> bool:
    """
    Passi:
    1. Cerca il profilo dell'utente.
    2. Se non esiste, crealo (in teoria dovrebbe sempre esistere, ma per sicurezza).
    3. Usa i metodi helper (set_generi_preferiti, set_generi_da_evitare) per salvare le liste come JSON.
    4. Aggiorna il timestamp aggiornato_il.
    5. Commit.
    6. Restituisci True.
    """
    db_profilo = db.query(models.ProfiloUtente).filter(
        models.ProfiloUtente.id_utente == id_utente
    ).first()

    if not db_profilo:
        db_profilo = models.ProfiloUtente(
            id_utente=id_utente,
            generi_preferiti="[]",
            generi_da_evitare="[]",
            preferenze_json="{}",
            summary_testuale=None
        )
        db.add(db_profilo)

    if generi_preferiti is not None:
        db_profilo.set_generi_preferiti(generi_preferiti)

    if generi_da_evitare is not None:
        db_profilo.set_generi_da_evitare(generi_da_evitare)

    if preferenze is not None:
        db_profilo.set_preferenze(preferenze)

    if summary_testuale is not None:
        db_profilo.summary_testuale = summary_testuale

    db_profilo.aggiornato_il = datetime.utcnow()

    db.commit()

    return True


def reimposta_profilo(db: Session, id_utente: int) -> bool:
    """
    Passi:
    1. Cerca il profilo dell'utente.
    2. Se esiste, imposta generi_preferiti = "[]" e generi_da_evitare = "[]".
    3. Aggiorna timestamp.
    4. Commit.
    5. Restituisci True se fatto, False se profilo non trovato.
    """
    db_profilo = db.query(models.ProfiloUtente).filter(
        models.ProfiloUtente.id_utente == id_utente
    ).first()

    if not db_profilo:
        return False

    db_profilo.set_generi_preferiti([])
    db_profilo.set_generi_da_evitare([])
    db_profilo.set_preferenze({})
    db_profilo.summary_testuale = None
    db_profilo.aggiornato_il = datetime.utcnow()

    db.commit()

    return True


# ===================== FILM CONSIGLIATI =====================

def salva_film_consigliato(db: Session,id_messaggio: int,id_utente: int,id_tmdb: int,titolo: str,
    titolo_originale: str | None = None, anno: int | None = None, descrizione: str | None = None,generi: List[str] | None = None,poster_path: str | None = None,
    poster_url: str | None = None,voto_medio: str | None = None,numero_voti: int | None = None,motivo_raccomandazione: str | None = None,
    link_streaming: Dict | None = None) -> int:

    """
    Passi:
    1. Crea un nuovo FilmConsigliato con id_messaggio, id_tmdb, titolo, anno, poster_url.
    2. Se generi non è vuoto, chiama film.set_generi(generi).
    3. Se link_streaming non è vuoto, chiama film.set_link_streaming(link_streaming).
    4. Aggiungi, commit, refresh.
    5. Restituisci id_consiglio.
    """
    nuovo_consiglio = models.FilmConsigliato(
        id_messaggio=id_messaggio,
        id_utente=id_utente,
        id_tmdb=id_tmdb,
        titolo=titolo,
        titolo_originale=titolo_originale,
        anno=anno,
        descrizione=descrizione,
        poster_path=poster_path,
        poster_url=poster_url,
        voto_medio=voto_medio,
        numero_voti=numero_voti,
        motivo_raccomandazione=motivo_raccomandazione,
        consigliato_il=datetime.utcnow()
    )

    if generi is not None:
        nuovo_consiglio.set_generi(generi)

    if link_streaming is not None:
        nuovo_consiglio.set_link_streaming(link_streaming)

    db.add(nuovo_consiglio)
    db.commit()
    db.refresh(nuovo_consiglio)

    return nuovo_consiglio.id_consiglio


def ottieni_film_consigliato(db: Session, id_consiglio: int) -> Optional[Dict]:
    """
    Passi:
    1. Query per id_consiglio.
    2. Se non trovato, restituisci None.
    3. Altrimenti costruisci dizionario con tutti i campi (convertendo generi e link da JSON usando i getter).
    """
    db_consiglio = db.query(models.FilmConsigliato).filter(
        models.FilmConsigliato.id_consiglio == id_consiglio
    ).first()

    if not db_consiglio:
        return None

    return {
        "id_consiglio": db_consiglio.id_consiglio,
        "id_messaggio": db_consiglio.id_messaggio,
        "id_utente": db_consiglio.id_utente,
        "id_tmdb": db_consiglio.id_tmdb,
        "titolo": db_consiglio.titolo,
        "titolo_originale": db_consiglio.titolo_originale,
        "anno": db_consiglio.anno,
        "descrizione": db_consiglio.descrizione,
        "generi": db_consiglio.get_generi(),
        "poster_path": db_consiglio.poster_path,
        "poster_url": db_consiglio.poster_url,
        "voto_medio": db_consiglio.voto_medio,
        "numero_voti": db_consiglio.numero_voti,
        "motivo_raccomandazione": db_consiglio.motivo_raccomandazione,
        "link_streaming": db_consiglio.get_link_streaming(),
        "consigliato_il": db_consiglio.consigliato_il
    }


def ottieni_film_per_messaggio(db: Session, id_messaggio: int) -> List[Dict]:
    """
    Passi:
    1. Fai una join tra FilmConsigliato, Messaggio, Sessione per filtrare solo i film appartenuti all'utente.
    2. Ordina per data consiglio decrescente.
    3. Per ogni risultato costruisci un dizionario (come in ottieni_film_consigliato).
    4. Restituisci lista.
    """
    film = db.query(models.FilmConsigliato).filter(
        models.FilmConsigliato.id_messaggio == id_messaggio
    ).order_by(models.FilmConsigliato.id_consiglio.asc()).all()

    return [
        ottieni_film_consigliato(db, f.id_consiglio)
        for f in film
    ]


def ottieni_film_per_utente(db: Session, id_utente: int) -> List[Dict]:
    film = db.query(models.FilmConsigliato).filter(
        models.FilmConsigliato.id_utente == id_utente
    ).order_by(models.FilmConsigliato.consigliato_il.desc()).all()

    return [
        ottieni_film_consigliato(db, f.id_consiglio)
        for f in film
    ]


def ottieni_tmdb_consigliati_sessione(db: Session, id_sessione: int) -> List[int]:
    """
    Restituisce gli id_tmdb di tutti i film gia' consigliati nella sessione
    indicata. Fonte precisa SQLite per evitare di riproporre nella stessa
    conversazione film gia' suggeriti.
    """
    righe = (
        db.query(models.FilmConsigliato.id_tmdb)
        .join(
            models.Messaggio,
            models.FilmConsigliato.id_messaggio == models.Messaggio.id_messaggio,
        )
        .filter(models.Messaggio.id_sessione == id_sessione)
        .all()
    )
    return [riga[0] for riga in righe if riga[0] is not None]


# ===================== FILM VISTI =====================

def segna_film_visto(db: Session,id_utente: int,id_tmdb: int,titolo: str | None = None,fonte: str = "dichiarato",gradito: bool | None = None) -> int:
    """
    Registra (o aggiorna) un film come gia' visto dall'utente.
    Fonte precisa SQLite: questi film non vanno mai riproposti.
    """
    if not id_tmdb:
        raise ValueError("id_tmdb mancante: impossibile segnare il film come visto.")

    gradito_int = None if gradito is None else (1 if gradito else 0)

    esistente = db.query(models.FilmVisto).filter(
        models.FilmVisto.id_utente == id_utente,
        models.FilmVisto.id_tmdb == id_tmdb,
    ).first()

    if esistente:
        if titolo:
            esistente.titolo = titolo
        esistente.fonte = fonte
        if gradito_int is not None:
            esistente.gradito = gradito_int
        esistente.visto_il = datetime.utcnow()
        db.commit()
        return esistente.id_visto

    nuovo = models.FilmVisto(
        id_utente=id_utente,
        id_tmdb=id_tmdb,
        titolo=titolo,
        fonte=fonte,
        gradito=gradito_int,
        visto_il=datetime.utcnow(),
    )
    db.add(nuovo)
    db.commit()
    db.refresh(nuovo)
    return nuovo.id_visto


def ottieni_tmdb_visti(db: Session, id_utente: int) -> List[int]:
    """
    Tutti gli id_tmdb dei film visti dall'utente (feedback o dichiarati),
    in qualsiasi sessione. Da escludere sempre dalle raccomandazioni.
    """
    righe = db.query(models.FilmVisto.id_tmdb).filter(
        models.FilmVisto.id_utente == id_utente
    ).all()
    return list({riga[0] for riga in righe if riga[0] is not None})


def ottieni_gusti_film_utente(db: Session, id_utente: int) -> Dict[str, List[Dict]]:
    """
    Film valutati con i thumbs (fonte precisa SQLite): due liste, piaciuti e
    non piaciuti, con generi recuperati dai consigli salvati. Usati come
    criterio importante nel rerank e mostrati nella pagina profilo.
    """
    visti = (
        db.query(models.FilmVisto)
        .filter(
            models.FilmVisto.id_utente == id_utente,
            models.FilmVisto.gradito.isnot(None),
        )
        .order_by(models.FilmVisto.visto_il.desc())
        .all()
    )
    piaciuti: List[Dict] = []
    non_piaciuti: List[Dict] = []
    for visto in visti:
        consiglio = (
            db.query(models.FilmConsigliato)
            .filter(
                models.FilmConsigliato.id_utente == id_utente,
                models.FilmConsigliato.id_tmdb == visto.id_tmdb,
            )
            .first()
        )
        voce = {
            "id_tmdb": visto.id_tmdb,
            "titolo": visto.titolo or (consiglio.titolo if consiglio else f"Film {visto.id_tmdb}"),
            "generi": consiglio.get_generi() if consiglio else [],
            "anno": consiglio.anno if consiglio else None,
            "visto_il": visto.visto_il,
        }
        (piaciuti if visto.gradito == 1 else non_piaciuti).append(voce)
    return {"piaciuti": piaciuti, "non_piaciuti": non_piaciuti}


def ottieni_consigli_sessione_dettagli(db: Session, id_sessione: int, limite: int = 12) -> List[Dict]:
    """
    Tutti i film consigliati in questa sessione, con generi e motivo salvato:
    fonte esatta SQLite per rispondere a domande sulla conversazione
    ("quali film mi hai consigliato?", "perche' li hai scelti?").
    """
    righe = (
        db.query(models.FilmConsigliato)
        .join(
            models.Messaggio,
            models.FilmConsigliato.id_messaggio == models.Messaggio.id_messaggio,
        )
        .filter(models.Messaggio.id_sessione == id_sessione)
        .order_by(models.FilmConsigliato.id_consiglio.desc())
        .limit(limite)
        .all()
    )
    return [
        {
            "titolo": f.titolo,
            "anno": f.anno,
            "id_tmdb": f.id_tmdb,
            "generi": f.get_generi(),
            "motivo": f.motivo_raccomandazione or "",
        }
        for f in reversed(righe)
    ]


def ottieni_ultimi_film_consigliati_sessione(db: Session, id_sessione: int) -> List[Dict]:
    """
    I film dell'ultimo messaggio assistant con consigli in questa sessione.
    Servono quando l'utente dice "ho gia' visto questi": "questi" sono
    esattamente questi film.
    """
    ultimo = (
        db.query(models.Messaggio)
        .join(
            models.FilmConsigliato,
            models.FilmConsigliato.id_messaggio == models.Messaggio.id_messaggio,
        )
        .filter(models.Messaggio.id_sessione == id_sessione)
        .order_by(models.Messaggio.inviato_il.desc())
        .first()
    )
    if not ultimo:
        return []
    return ottieni_film_per_messaggio(db, ultimo.id_messaggio)


# ===================== FEEDBACK =====================

def salva_feedback(db: Session, id_utente: int, id_consiglio: int, voto: int, commento: str | None = None) -> bool:
    """
    Passi:
    1. Controlla se esiste già un feedback per quella coppia (id_utente, id_consiglio).
    2. Se esiste, aggiorna il voto e la data.
    3. Se non esiste, crea un nuovo record Feedback.
    4. Commit.
    5. Restituisci True.
    """
    if voto < 1 or voto > 10:
        raise ValueError("Il voto deve essere compreso tra 1 e 10.")

    db_feedback = db.query(models.Feedback).filter(
        models.Feedback.id_utente == id_utente,
        models.Feedback.id_consiglio == id_consiglio
    ).first()

    if db_feedback:
        db_feedback.voto = voto
        db_feedback.commento = commento
        db_feedback.dato_il = datetime.utcnow()
    else:
        nuovo_feedback = models.Feedback(
            id_utente=id_utente,
            id_consiglio=id_consiglio,
            voto=voto,
            commento=commento,
            dato_il=datetime.utcnow()
        )
        db.add(nuovo_feedback)

    db.commit()

    return True


def ottieni_feedback_utente(db: Session, id_utente: int) -> List[Dict]:
    """
    Passi:
    1. Query su Feedback filtrando per id_utente, ordine decrescente per dato_il.
    2. Per ogni feedback restituisci dizionario con id_feedback, id_consiglio, voto, dato_il.
    """   
    feedbacks = db.query(models.Feedback).filter(
        models.Feedback.id_utente == id_utente
    ).order_by(models.Feedback.dato_il.desc()).all()

    return [
        {
            "id_feedback": f.id_feedback,
            "id_utente": f.id_utente,
            "id_consiglio": f.id_consiglio,
            "voto": f.voto,
            "commento": f.commento,
            "dato_il": f.dato_il
        }
        for f in feedbacks
    ]


def ottieni_feedback_film(db: Session,id_utente: int,id_consiglio: int) -> Optional[Dict]:
    """
    Passi:
    1. Query per feedback con quell'utente e quel film.
    2. Se trovato, restituisci dizionario (id_feedback, voto, dato_il).
    3. Altrimenti None.
    """    
    feedback = db.query(models.Feedback).filter(
        models.Feedback.id_utente == id_utente,
        models.Feedback.id_consiglio == id_consiglio
    ).first()

    if not feedback:
        return None

    return {
        "id_feedback": feedback.id_feedback,
        "id_utente": feedback.id_utente,
        "id_consiglio": feedback.id_consiglio,
        "voto": feedback.voto,
        "commento": feedback.commento,
        "dato_il": feedback.dato_il
    }


def ottieni_tmdb_da_evitare_storico(db: Session, id_utente: int, soglia_voto: int = 5) -> List[int]:
    """
    Restituisce gli id_tmdb dei film a cui l'utente ha dato un feedback
    negativo (voto < soglia_voto, default 5). Sono film gia' visti e non
    graditi: non vanno riproposti nelle raccomandazioni future.
    Fonte precisa SQLite, non memoria semantica.
    """
    righe = (
        db.query(models.FilmConsigliato.id_tmdb)
        .join(
            models.Feedback,
            models.Feedback.id_consiglio == models.FilmConsigliato.id_consiglio,
        )
        .filter(
            models.Feedback.id_utente == id_utente,
            models.Feedback.voto < soglia_voto,
        )
        .all()
    )
    return list({riga[0] for riga in righe if riga[0] is not None})
