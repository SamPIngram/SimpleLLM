import streamlit as st

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

st.markdown("# Training")

st.sidebar.error("Need configuration and dataset to train model.")