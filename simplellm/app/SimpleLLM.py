import streamlit as st

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

st.write("# Welcome to SimpleLLM GUI! 👋")

st.sidebar.success("Select a option above.")

st.markdown(
    """
    This web application is a GUI for the [SimpleLLM](https://github.com/SamPIngram/SimpleLLM) project. 
    It walks you through the process of training large language models.
"""
)