# 🎬 MovieAgentOS

**Sistema multi-agente per la raccomandazione personalizzata di film**

MovieAgentOS è una web application intelligente che consiglia film all'utente tramite linguaggio naturale. Il sistema apprende progressivamente i gusti dell'utente, memorizza le preferenze nel tempo e genera raccomandazioni personalizzate attraverso un'architettura multi-agente.


## Idea del Progetto

L'obiettivo è costruire un assistente cinematografico che non si limiti a cercare film, ma che **impari** dall'utente nel tempo. A differenza di un semplice chatbot o recommender system, MovieAgentOS:

- Mantiene una **memoria persistente** dei gusti dell'utente (generi preferiti, generi da evitare, feedback sui film visti)
- Usa **RAG su ChromaDB** per recuperare memorie semanticamente rilevanti in base al contesto della conversazione
- Gestisce **sessioni multiple** per ogni utente, permettendo di riprendere conversazioni precedenti
- Prende **decisioni autonome** — capisce se l'utente vuole un film, vuole aggiornare le sue preferenze, o sta facendo una domanda informativa

Il sistema soddisfa i requisiti di un progetto universitario su agenti AI perché implementa un vero loop agentivo: **percezione → ragionamento → azione → aggiornamento memoria**.

---

## Architettura

```
Frontend Streamlit
        ↓
ConduttoreDiAgents  ←→  AgenteMemoria  ←→  ChromaDB (RAG)
        ↓                                   SQLite (persistenza)
   AgenteRicerca
        ↓
    TMDB API
```

---

## Tecnologie

| Componente | Tecnologia | Motivo |
|---|---|---|
| Framework agenti | [Agno](https://docs.agno.com) | Orchestrazione multi-agente in Python |
| LLM | OpenRouter (gpt-oss-20b) | Provider LLM flessibile, supporta modelli multipli |
| Memoria vettoriale | ChromaDB | RAG per preferenze semantiche utente |
| Database persistente | SQLite + SQLAlchemy | Utenti, sessioni, messaggi, feedback |
| API film | [TMDB API](https://developer.themoviedb.org) | Database film gratuito e completo |
| Frontend | Streamlit | Web app Python rapida da sviluppare |
| Autenticazione | Google OAuth (`st.login`) | Login sicuro senza gestire password |
| Linguaggio | Python 3.11+ | Stack uniforme backend e frontend |

---

## Installazione e Avvio

### Requisiti

- Python 3.11 o superiore
- Account OpenRouter (per la API key LLM)
- Account TMDB (per la API key film)
- Progetto Google Cloud con OAuth 2.0 configurato

### 1. Clona il repository

```bash
git clone https://github.com/yassineelhosayny/MovieAgentOS
cd movieagent-os
```

### 2. Crea un ambiente virtuale

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

## 4. API Key per la valutazione

Per facilitare la valutazione del progetto, è disponibile una chiave temporanea
già configurata valida fino al **[22 lug 2026]**:

> ⚠️ Questa chiave è fornita solo per la valutazione e scadrà.
> Per uso continuativo, crea una chiave gratuita su openrouter.ai


### 5. Avvia l'applicazione

```bash
cd MovieAgentOS-main
streamlit run .\frontEnd\app.py
```
L'app sarà disponibile su `http://localhost:8501`

---

## Configurazione

### modello LLM

Per migliorare le prestazioni, modifica `MODEL_ID_LLM` in `secrets.toml`:

```toml
# Più veloce e potente (richiede crediti OpenRouter)
MODEL_ID_LLM = "gpt-4.1-nano"   => usato per test (veloce)

# Gratuito ma più lento
MODEL_ID_LLM = "openai/gpt-oss-20b:free" 
```
