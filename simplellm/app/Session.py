import streamlit as st
import json
from datetime import datetime

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

if "session_loaded" not in st.session_state:
    st.session_state["session_loaded"] = False

def open_session():
    st.session_state["session_loaded"] = True

def unload_session():
    st.session_state["session_loaded"] = False

def recent_sessions():
    with open("simplellm/app/recent_sessions.json", "r") as f:
        recent_sessions = json.load(f)
    for session in recent_sessions:
        st.sidebar.button(session, on_click=open_session)

def add_new_session(new_session_name, new_session_description, new_session_path):          
    with open("simplellm/app/recent_sessions.json", "r") as f:
        recent_sessions = json.load(f)
    if len(recent_sessions) > 4:
        recent_sessions = dict(list(recent_sessions.items())[-4:])
    now = datetime.now()
    # Format the date to ddmmyy
    date_string = now.strftime("%d%m%y")
    finding_new_session_name = True
    idx = 0
    new_session_key = f"{date_string}_{new_session_name}"
    while finding_new_session_name:
        if new_session_key not in recent_sessions:
            finding_new_session_name = False
        else:
            new_session_key = f"{date_string}_{new_session_name}_{idx}"
            idx += 1
    recent_sessions[new_session_key] = {
        "name": new_session_name,
        "description": new_session_description,
        "path": new_session_path,
    }
    with open("simplellm/app/recent_sessions.json", "w") as f:
        json.dump(recent_sessions, f)
    st.session_state["session_loaded"] = True

def new_session():
    with st.form("new_session_form"):
        new_session_name = st.text_input("Session Name", placeholder="My Session")
        new_session_description = st.text_input("Session Description", placeholder="A description of my session")
        new_session_path = st.text_input("Session Path", placeholder="~/my-session")
        submit_button = st.form_submit_button(label='Create Session', on_click=add_new_session, args=(new_session_name, new_session_description, new_session_path))

if st.session_state["session_loaded"] is False:
    st.sidebar.button("New Session", on_click=new_session)
    st.sidebar.button("Load Session")
    st.sidebar.divider()
    st.sidebar.header("Recent Sessions:")
    recent_sessions()
else:
    st.sidebar.button("Save Session")
    st.sidebar.button("Unload Session", on_click=unload_session)

