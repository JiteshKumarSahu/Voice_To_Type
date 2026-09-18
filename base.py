import streamlit as st
from transformers import pipeline
import torch



# 2. Load the ML Model (Cached so it only loads once)
@st.cache_resource
def load_speech_model():
    # Using the tiny model for fast, local CPU/GPU execution
    pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return pipe
