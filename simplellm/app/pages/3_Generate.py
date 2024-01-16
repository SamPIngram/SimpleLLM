import streamlit as st

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

st.markdown("# Generate Text")

st.sidebar.warning("No model loaded")
