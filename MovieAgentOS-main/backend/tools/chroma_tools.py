from datetime import datetime, timezone
from uuid import uuid4

import chromadb
import json
from typing import Any
from pathlib import Path

#path della chroma
CHROMA_PATH = Path(__file__).resolve().parents[1] / "db" / "chroma"


def get_chroma_client()-> chromadb.PersistentClient:
    """
    crea il client chroma, apre chromaDB nella cartella backend/db/chroma 
    """
    CHROMA_PATH.mkdir(parents=True,exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))

def get_utente_memory_collection(id_utente:int):
    """
    restiusce un collection/informazione completa dove viene salvata la memoria del utente
    """
    if id_utente is None or id_utente <= 0:
        raise ValueError("id_utente non valido o mancante!")

    client = get_chroma_client()

#ricupero solo la memoria del utente con id_utente
    nome_collection = f"memorie_utente_{id_utente}"
    
    info= client.get_or_create_collection(
        name=nome_collection,
        metadata={
            "descrizione": f"Memorie semantiche dell'utente {id_utente}",
            "id_utente": id_utente
        }
    )
    return info

def pulisci_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_puliti = {}
    for chiave, valore in metadata.items():
        if valore is None:
            continue

        if isinstance(valore, (str, int, float, bool)):
            metadata_puliti[chiave] = valore
        else:
            metadata_puliti[chiave] = json.dumps(valore, ensure_ascii=False)

    return metadata_puliti

def salva_memoria_utente(id_utente:int, testo:str, tipo:str, fonte:str, metadata_extra: dict[str, Any] | None = None)-> dict[str, Any]:
    """
    salva nuovi informazione/preferences del utente
    """
    if id_utente is None or id_utente<=0:
        return{
            "success":False,
            "errore":"id_utente non valido o mancante!"
        }
    if not testo or not testo.strip():
        return{
            "success":False,
            "errore":"Test della memoria è vuoto!!!"
        }       
    if not tipo or not tipo.strip():
        return {
            "success": False,
            "errore": "Tipo memoria mancante."
        }
    if not fonte or not fonte.strip():
        return {
            "success": False,
            "errore": "Fonte memoria mancante."
        }
    #recuperare la collection
    info= get_utente_memory_collection(id_utente)
    #memory id, uso uuid4 per evitare creare conflitti di ids tra utente, esmpio di un id : utente_1_a83f91c8cde74f8da29b82c71f0d2e91
    memoria_id =f"utente_{id_utente}_{uuid4().hex}"

    metadata={
        "id_utente": id_utente,
        "tipo": tipo.strip(),
        "fonte": fonte.strip(),
        "created_at": datetime.now(timezone.utc).isoformat()    
    }
    #controllo se ci metadata_extra e salvarla se esiste
    if metadata_extra :
        metadata.update(metadata_extra)

    metadata = pulisci_metadata(metadata)
    try:
        info.add(
            ids=[memoria_id],
            documents= [testo.strip()],
            metadatas=[metadata],
        )
        return{
            "success": True,
            "id_memoria": memoria_id,
            "testo": testo.strip(),
            "metadata": metadata

        }
    except Exception as e:
        return {
            "success": False,
            "errore": f"Errore durante il salvataggio memoria in ChromaDB: {e}"
        }


def cerca_memorie_utente(id_utente:int, query:str, num_results:int= 5)->list[dict[str,Any]]:
    """
    recuperare una lista di memorie rilevanti rispetto alla query attuale
    """
    if id_utente is None or id_utente <= 0:
            return []
    if not query or not query.strip():
            return []
    if num_results <= 0:
        num_results = 5

    info = get_utente_memory_collection(id_utente)
    try:
        res = info.query(
            query_texts=[query.strip()],
            n_results=num_results,
            where={ "id_utente":id_utente} #non serve ma lo uso per sicurezza
            )
        ids= res.get("ids",[[]])[0]
        doc= res.get("documents",[[]])[0]
        md = res.get("metadatas",[[]])[0]
        dis = res.get("distances", [[]])[0]

        memoria =[]
        for i in range(len(ids)):
            memoria.append({
                "id": ids[i],
                "testo": doc[i],
                "metadata": md[i],
                "distance": dis[i] if i < len(dis) else None                
            })
        return memoria
    except Exception as e:
        return [{
        "errore": f"Errore durante la ricerca memorie ChromaDB: {e}"
    }]


def salva_feedback_memoria(id_utente:int, titolo_film:str, positivo:bool, commento:str |None=None, generi:list[str]|None=None)->dict[str,Any]:
    """
    Trasforma un feedback thumbs up/down su un film in una memoria semantica.
    Questa memoria serve SOLO per personalizzare il rerank: l'esclusione del
    film gia' visto e' un dato preciso gestito in SQLite (film_visti), non qui.
    """
    if id_utente is None or id_utente <= 0:
        return {
            "success": False,
            "errore": "id_utente non valido o mancante."
        }
    if not titolo_film or not titolo_film.strip():
        return {
            "success": False,
            "errore": "titolo_film mancante."
        }
    titolo = titolo_film.strip()
    generi_testo = ", ".join(generi) if generi else "generi non specificati"
    commento_pulito = commento.strip() if commento and commento.strip() else None

    if positivo:
        tipo = "feedback_positivo"
        testo = (
            f"All'utente e' piaciuto il film '{titolo}' (generi: {generi_testo}). "
            f"E' un segnale di gradimento per questi generi e per atmosfere simili, "
            f"da usare come preferenza morbida tra risultati compatibili."
        )
    else:
        tipo = "feedback_negativo"
        testo = (
            f"All'utente NON e' piaciuto il film '{titolo}' (generi: {generi_testo}). "
            f"Film molto simili per storia, trama o atmosfera sono meno adatti. "
            f"Questo riguarda il film specifico e non vieta automaticamente i suoi generi."
        )

    if commento_pulito:
        testo += f" Commento dell'utente: {commento_pulito}."

    metadata_extra = {
        "titolo_film": titolo,
        "positivo": positivo,
        "generi": generi or [],
        "ha_commento": commento_pulito is not None
    }

    return salva_memoria_utente(
            id_utente=id_utente,
            testo=testo,
            tipo=tipo,
            fonte="feedback",
            metadata_extra=metadata_extra
       )

def cancella_memorie_utente(id_utente:int)-> dict[str, Any]: #ricorda di controllare se il return è errore nel uso
    """
    cancella tutta la memoria, per ora l'ho sto usando solo per Tests
    """
    if id_utente is None or id_utente <= 0:
        return {
            "success": False,
            "errore": "id_utente non valido o mancante."
        }
    info = get_utente_memory_collection(id_utente)
    try:
        res = info.get(
            where= {"id_utente": id_utente}
        )
        ids= res.get("ids",[])
        if not ids :
            return{
                "success":True, # è true perchè operazione è tecnicamente riuscita (non c'era niente da cancellare)
                "memorie_cancellate": 0
            }
        info.delete(ids=ids)
        return{
              "success": True,
              "memorie_cancellate": len(ids)# miglio restiusco numero di info che stato cancellato      
                }
    except Exception as e:
        return{
            "success": False,
            "errore": f"Errore durante la cancellazione memorie ChromaDB: {e}"           
        }


    
