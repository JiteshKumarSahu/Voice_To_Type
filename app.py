import streamlit as st
from transformers import pipeline
import librosa
import io
import base

# 1. Configure the Streamlit Page
st.set_page_config(page_title="Local Voice Recognition", page_icon="🎙️")
st.title("🎙️ Jitesh Voice Recognition App")
st.write("Record your voice below!")


with st.spinner("Loading Machine Learning Model..."):
    asr_pipeline = base.load_speech_model()

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