import streamlit as st
from style.util import load_css

#mostra la prima pagina dove da informazioni e la possibilità di fare login

#carico il css
load_css("landing.css")

#titolo in alto
st.markdown('<div class="title"> 🎬 MovieAgentOS</div>', unsafe_allow_html=True)

#parte centrale con descrizione
st.markdown(
    '<div class="subtitle">La piattaforma intelligente per scoprire film su misura per te</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="features">🍿 Catalogo aggiornato<br>🤖 Consigli personalizzati<br>⚡ Esperienza veloce</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("🔐 Accesso richiesto")
#Bottone accesso
if st.button("🔑 Accedi con Google-Auth0", use_container_width=True):
    st.login("auth0")