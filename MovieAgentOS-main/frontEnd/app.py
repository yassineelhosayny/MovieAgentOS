import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gui.auth import is_loggato
from gui.mostra_pagina_loggato import mostra_pagina_loggato
from backend.db.database import SessionLocal, init_db
from backend.db import crud


init_db()

#======== Login ========================================================================================================


if not is_loggato():
    #carica pagina per login
    pagina = st.navigation([st.Page("gui/mostra_pagina_iniziale.py")])
else:
    #recupea l'utente loggato o lo crea se nuovo
    if "id_utente" not in st.session_state:
        db = SessionLocal()
        try:
            st.session_state.id_utente = crud.ottieniCrea_utente(
                db = db,
                email = st.user.email,
                nome = st.user.name,
                auth_sub = st.user.sub,
                foto_url = st.user.picture
            )
        finally:
            db.close()


    #router per le pagine
    pagina = st.navigation([
        st.Page("pages/chat.py", title = "Chat"),
        st.Page("pages/profile.py", title="Profile"),
    ])

    #se non carica le route da comunque la possibilità di fare logout
    if not pagina:
        mostra_pagina_loggato()

pagina.run()
