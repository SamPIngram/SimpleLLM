import streamlit as st
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simplellm.configurator import TrainerConfig, DataConfig, GeneratorConfig
import os
import glob

# TODO finish session management system for webapp
st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

dir_path = os.path.dirname(os.path.realpath(__file__))

if "session_loaded" not in st.session_state:
    st.session_state["session_loaded"] = False

def save_session_state(save_name):
    # Create a path to the session_state.json file in the same directory
    file_path = os.path.join(dir_path, f'sessions/{save_name}.json')

    with open(file_path, 'w') as f:
        # Convert the session state to a dictionary and then save it as JSON
        if "config" in st.session_state:
            st.sesson_state["config"] = st.session_state["config"].__dict__
        if "DataConfig" in st.session_state:
            st.session_state["DataConfig"] = st.session_state["DataConfig"].__dict__
        if "GeneratorConfig" in st.session_state:
            st.session_state["GeneratorConfig"] = st.session_state["GeneratorConfig"].__dict__
        json.dump(st.session_state.to_dict(), f)

def load_session_state():
    st.session_state.clear()
    uploaded_file = st.file_uploader("Choose a session file", type="json")
    if uploaded_file is not None:
        state = json.load(uploaded_file)
        for key, value in state.items():
            st.session_state[key] = value
    if "config" in st.session_state:
        st.session_state["config"] = TrainerConfig().from_dict(st.session_state["config"])
    if "DataConfig" in st.session_state:
        st.session_state["DataConfig"] = DataConfig().from_dict(st.session_state["DataConfig"])
    if "GeneratorConfig" in st.session_state:
        st.session_state["GeneratorConfig"] = GeneratorConfig().from_dict(st.session_state["GeneratorConfig"])
    st.session_state["session_loaded"] = True

def open_session(session_path):
    st.session_state.clear()
    with open(session_path, 'r') as session_path:
        state = json.load(session_path)
    for key, value in state.items():
        st.session_state[key] = value
    print(st.session_state)
    if "config" in st.session_state:
        st.session_state["config"] = TrainerConfig().from_dict(st.session_state["config"])
    if "DataConfig" in st.session_state:
        st.session_state["DataConfig"] = DataConfig().from_dict(st.session_state["DataConfig"])
    if "GeneratorConfig" in st.session_state:
        st.session_state["GeneratorConfig"] = GeneratorConfig().from_dict(st.session_state["GeneratorConfig"])
    st.session_state["session_loaded"] = True

def unload_session():
    st.session_state.clear()
    st.session_state["session_loaded"] = False

def recent_sessions():
    session_files = glob.glob(os.path.join(dir_path, 'sessions/*.json'))
    # sort by date modified (newest first)
    sorted_files = sorted(session_files, key=os.path.getmtime, reverse=True)
    for i, session in enumerate(sorted_files):
        # only show the 10 most recent sessions
        if i <= 10:
            st.sidebar.button(session.split('/')[-1], on_click=open_session, args=(session,))
        # delete the rest
        # else:
        #     os.remove(session)
    return sorted_files

def add_new_session(new_session_name, new_session_description):   
    now = datetime.now()
    # Format the date to ddmmyy
    date_string = now.strftime("%d%m%y")
    finding_new_session_name = True
    idx = 0
    new_session_key = f"{date_string}_{new_session_name}"
    while finding_new_session_name:
        if new_session_key not in sessions:
            finding_new_session_name = False
        else:
            new_session_key = f"{date_string}_{new_session_name}_{idx}"
            idx += 1
    # st.session_state["session_name"] = new_session_key
    # st.session_state["session_description"] = new_session_description
    st.session_state["session_loaded"] = True

def new_session(): # Not updating based on form input will look to solve this
    with st.form(key="new_session_form", clear_on_submit=True):
        st.session_state["session_name"] = st.text_input("Session Name", value="My Session")
        st.session_state["session_description"] = st.text_input("Session Description", value="A description of my session")
        submit_button = st.form_submit_button(label='Create Session', on_click=add_new_session, args=(st.session_state["session_name"], st.session_state["session_description"]))

if st.session_state["session_loaded"] is False:
    new_session = st.sidebar.button("New Session", on_click=new_session)
    st.sidebar.button("Load Session", on_click=load_session_state)
    st.sidebar.divider()
    st.sidebar.header("Recent Sessions:")
    sessions = recent_sessions()
else:
    st.sidebar.button("Save Session", on_click=save_session_state, args=(st.session_state["session_name"],))
    st.sidebar.button("Unload Session", on_click=unload_session)

