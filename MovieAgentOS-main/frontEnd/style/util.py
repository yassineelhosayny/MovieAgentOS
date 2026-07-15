import streamlit as st
from pathlib import Path


def load_css(filename: str):
    base_dir = Path(__file__).resolve().parent
    css_path = base_dir / filename

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)