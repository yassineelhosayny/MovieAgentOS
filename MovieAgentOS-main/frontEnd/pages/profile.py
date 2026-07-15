import streamlit as st
from backend.db import crud
from backend.db.database import SessionLocal
from backend.tools import chroma_tools
from gui.mostra_pagina_loggato import mostra_pagina_loggato
from datetime import timezone, timedelta


def cancella_preferenze_utente(db, id_utente: int) -> tuple[bool, str]:
    """Cancella preferenze salvate senza passare dagli agent o dal LLM."""
    crud.crea_profilo_se_non_esiste(db, id_utente)
    profilo_resettato = crud.reimposta_profilo(db, id_utente)
    memoria_resettata = chroma_tools.cancella_memorie_utente(id_utente)

    if not profilo_resettato:
        return False, "Profilo utente non trovato."
    if not memoria_resettata.get("success"):
        return False, memoria_resettata.get("errore", "Errore durante la cancellazione della memoria.")

    memorie_cancellate = memoria_resettata.get("memorie_cancellate", 0)
    return True, f"Preferenze cancellate. Memorie semantiche rimosse: {memorie_cancellate}."


if "id_utente" not in st.session_state:
    st.session_state.id_utente = 1

id_utente = st.session_state.id_utente
db = SessionLocal()

st.markdown(
    """
    <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #c1121f;
            border-color: #c1121f;
            color: white;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #9f0f1a;
            border-color: #9f0f1a;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

titolo_col, azione_col = st.columns([3, 1], vertical_alignment="center")
with titolo_col:
    st.title("Il tuo Profilo")

with azione_col:
    if st.button("Cancella preferenze", type="primary", use_container_width=True):
        success, messaggio = cancella_preferenze_utente(db, id_utente)
        st.session_state["messaggio_cancella_preferenze"] = (success, messaggio)
        st.rerun()

messaggio_cancellazione = st.session_state.pop("messaggio_cancella_preferenze", None)
if messaggio_cancellazione:
    success, messaggio = messaggio_cancellazione
    if success:
        st.success(messaggio)
    else:
        st.error(messaggio)


st.divider()
st.subheader("I tuoi gusti")
profilo = crud.ottieni_profilo(db, id_utente)
generi_preferiti = profilo.get("generi_preferiti", []) if profilo else []
generi_da_evitare = profilo.get("generi_da_evitare", []) if profilo else []

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Generi preferiti**")
    if generi_preferiti:
        for g in generi_preferiti:
            st.markdown(f"- {g}")
    else:
        st.caption("Nessun genere ancora — inizia a chattare!")

with col2:
    st.markdown("**Generi da evitare**")
    if generi_da_evitare:
        for g in generi_da_evitare:
            st.markdown(f"- {g}")
    else:
        st.caption("Nessuno per ora")

st.divider()

#======== I tuoi film (dai feedback thumbs) ===========================================================================
# Questi film sono dati precisi (SQLite): non vengono mai riproposti e i loro
# generi/affinita' sono un criterio importante nella scelta dei prossimi consigli.

st.subheader("I tuoi film")

gusti = crud.ottieni_gusti_film_utente(db, id_utente)

col_up, col_down = st.columns(2)

with col_up:
    st.markdown("**👍 Ti sono piaciuti**")
    if gusti["piaciuti"]:
        for film in gusti["piaciuti"]:
            with st.container(border=True):
                anno = f" ({film['anno']})" if film.get("anno") else ""
                st.markdown(f"**{film['titolo']}**{anno}")
                if film.get("generi"):
                    st.caption(", ".join(film["generi"]))
    else:
        st.caption("Ancora nessuno: valuta i consigli in chat con 👍")

with col_down:
    st.markdown("**👎 Non ti sono piaciuti**")
    if gusti["non_piaciuti"]:
        for film in gusti["non_piaciuti"]:
            with st.container(border=True):
                anno = f" ({film['anno']})" if film.get("anno") else ""
                st.markdown(f"**{film['titolo']}**{anno}")
                if film.get("generi"):
                    st.caption(", ".join(film["generi"]))
    else:
        st.caption("Ancora nessuno: valuta i consigli in chat con 👎")

st.caption("Uso queste valutazioni per scegliere meglio i prossimi consigli: "
           "i film valutati non vengono mai riproposti.")

st.divider()

#======== Sessioni =====================================================================================================

st.subheader("Sessioni precedenti")

sessioni = crud.ottieni_sessioni_utente(db, id_utente)

orario_ita = timezone(timedelta(hours=2))

if sessioni:
    for s in sessioni:
        data = s['creata_il'].replace(tzinfo=timezone.utc).astimezone(orario_ita)
        st.markdown(f"- **{s['titolo']}** — {data.strftime('%d/%m/%Y %H:%M')}")
else:
    st.caption("Nessuna sessione, chatta per crearne una")

st.divider()

mostra_pagina_loggato()
