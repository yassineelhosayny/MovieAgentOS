import streamlit as st


def mostra_pagina_loggato():
    with st.sidebar:
        if st.user.picture:
            st.image(st.user.picture, width=69)

        st.markdown(f"**{st.user.name}**")

        if st.button("🚪 Logout", use_container_width=True):
            st.logout()

