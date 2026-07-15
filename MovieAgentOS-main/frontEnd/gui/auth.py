import streamlit as st

#funzione per controllare se l'utente è loggato
def is_loggato():
    return bool(st.user.get("is_logged_in", False))
