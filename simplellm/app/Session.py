import streamlit as st
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simplellm.configurator import TrainerConfig, DataConfig, GeneratorConfig
import os
import glob
import pandas as pd
import copy

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
    state_for_saving = {}
    state_for_saving["session_name"] = st.session_state["session_name"]
    state_for_saving["session_description"] = st.session_state["session_description"]
    state_for_saving["session_loaded"] = st.session_state["session_loaded"]
    with open(file_path, 'w') as f:
        # Convert the session state to a dictionary and then save it as JSON
        if "config" in st.session_state:
            state_for_saving["config"] = st.session_state["config"].__dict__
        if "DataConfig" in st.session_state:
            state_for_saving["DataConfig"] = st.session_state["DataConfig"].__dict__
        if "GeneratorConfig" in st.session_state:
            state_for_saving["GeneratorConfig"] = st.session_state["GeneratorConfig"].__dict__
        json.dump(state_for_saving, f)

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

if st.session_state["session_loaded"] is False:
    st.sidebar.button("Load Session", on_click=load_session_state)
    st.sidebar.divider()
    st.sidebar.header("Recent Sessions:")
    sessions = recent_sessions()
else:
    st.sidebar.button("Save Session", on_click=save_session_state, args=(st.session_state["session_name"],))
    st.sidebar.button("Unload Session", on_click=unload_session)



if st.session_state["session_loaded"]:
    session_name = st.text_input("Session Name", value=st.session_state["session_name"])
    session_description = st.text_area("Session Description", value=st.session_state["session_description"])
    with st.expander("Configurations"):

        session_parms = {key: value for key, value in st.session_state.items() if key not in ["DataConfig", "config", "GeneratorConfig"]}
        config_table = pd.DataFrame.from_dict(session_parms, orient='index', columns=['Value'], dtype=str)
        st.markdown("## Session State:")
        st.table(config_table)

        if "DataConfig" in st.session_state:
            data_config_table = pd.DataFrame.from_dict(st.session_state["DataConfig"].__dict__, orient='index', columns=['Value'], dtype=str)
            st.markdown("## Data Config:")
            st.table(data_config_table)

        if "config" in st.session_state:
            training_config_table = pd.DataFrame.from_dict(st.session_state["config"].__dict__, orient='index', columns=['Value'], dtype=str)
            st.markdown("## Training Config:")
            st.table(training_config_table)

        if "GeneratorConfig" in st.session_state:
            generator_config_table = pd.DataFrame.from_dict(st.session_state["GeneratorConfig"].__dict__, orient='index', columns=['Value'], dtype=str)
            st.markdown("## Generator Config:")
            st.table(generator_config_table)
else:
    session_name = st.text_input("Session Name", placeholder="Session Name")
    session_description = st.text_area("Session Description", placeholder="Session Description")
    if st.button("Create Session"):
        now = datetime.now()
        # Format the date to ddmmyy
        date_string = now.strftime("%d%m%y")
        finding_new_session_name = True
        idx = 1
        new_session_key = f"{date_string}_{session_name}"
        session_name_list = [session.split('/')[-1].split('.json')[0] for session in sessions]
        print(session_name_list)
        while finding_new_session_name:
            if new_session_key not in session_name_list:
                finding_new_session_name = False
            else:
                new_session_key = f"{date_string}_{session_name}_{idx}"
                idx += 1
        st.session_state["session_name"] = new_session_key
        st.session_state["session_description"] = session_description
        st.session_state["session_loaded"] = True

        st.experimental_rerun()
