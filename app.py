import streamlit as st
from transformers import pipeline
import torch
import librosa
import io

# 1. Configure the Streamlit Page
st.set_page_config(page_title="Local Voice Recognition", page_icon="🎙️")
st.title("🎙️ Jitesh Voice Recognition App")
st.write("Record your voice below!")

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

with st.spinner("Loading Machine Learning Model..."):
    asr_pipeline = load_speech_model()

# 3. Audio Input Widget
audio_file = st.audio_input("Click the microphone to record")

# 4. Process the Audio and Run ML Inference
if audio_file is not None:
    st.audio(audio_file) # Playback the recorded audio
    
    with st.spinner("Transcribing your voice..."):
        try:
            # Read the audio bytes into a file-like object
            audio_bytes = audio_file.read()
            
            # Convert audio bytes into an array that the ML model understands
            # Whisper requires a 16kHz sampling rate
            audio_data, samplerate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            
            # Run the speech-to-text pipeline
            prediction = asr_pipeline(audio_data)
            transcript = prediction["text"]
            
            # Display the result
            st.success("Transcription Complete!")
            st.subheader("Result:")
            st.write("Jitesh: "+ transcript)
            
        except Exception as e:
            st.error(f"An error occurred during transcription: {e}")