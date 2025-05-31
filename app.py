import streamlit as st
import pyttsx3
import wikipedia
import uuid
import os
import threading
import speech_recognition as sr
from dotenv import load_dotenv
import chromadb
from groq import Groq

# Load .env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq setup
client = Groq(api_key=GROQ_API_KEY)
model_name = "llama-3.3-70b-versatile"

# ChromaDB setup
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="prompt_store")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
stop_speech_flag = threading.Event()
speech_thread = None

def speak_text(text):
    global speech_thread
    stop_speech_flag.clear()
    def _speak():
        if not stop_speech_flag.is_set():
            engine.say(text)
            engine.runAndWait()
    speech_thread = threading.Thread(target=_speak)
    speech_thread.start()

def stop_speech():
    stop_speech_flag.set()
    engine.stop()

# Prompt formatter
def smart_prompt_builder(user_prompt):
    user_prompt = user_prompt.lower()
    if user_prompt.startswith("why"):
        return f"Explain this kindly to a 6-year-old: Why {user_prompt[4:]}"
    elif user_prompt.startswith("how"):
        return f"Explain how this works in a fun way for a kid: {user_prompt}"
    elif "story" in user_prompt or "tell me" in user_prompt:
        return f"Tell a short, fun story for a child: {user_prompt}"
    else:
        return f"Answer this simply and kindly like to a 6-year-old: {user_prompt}"

# Wikipedia fallback
def get_wikipedia_summary(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "Sorry, I couldn't find anything on Wikipedia."

# 🎤 Mic input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.session_state.prompt = query
            st.success(f"🗣️ You said: {query}")
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")

# Streamlit UI setup
st.set_page_config(page_title="Mini-KreatePrompt", page_icon="🧠")
st.title("🧠 Mini-KreatePrompt: AI Playground for Kids")

with st.sidebar:
    st.title("📘 How to Use")
    st.markdown("""
    1. Click “Use Mic” or type a question.
    2. Click “Get Response”.
    3. Click “🔇 Stop Voice” to stop the answer anytime.
    """)

# Initialize session prompt
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# 🎤 Mic input
if st.button("🎤 Use Mic"):
    recognize_speech()

# Text input box
typed_prompt = st.text_area("What do you want to ask?", value=st.session_state.prompt)
if typed_prompt.strip():
    st.session_state.prompt = typed_prompt

prompt = st.session_state.prompt

# Generate response
if st.button("Get Response"):
    if not prompt.strip():
        st.warning("Please type or speak your question.")
    else:
        with st.spinner("Thinking..."):
            results = collection.query(query_texts=[prompt], n_results=1)
            documents = results["documents"][0]
            distances = results["distances"][0]

            if documents and distances[0] < 0.2:
                answer = results["metadatas"][0][0]["answer"]
                st.success("✨ Found a similar question!")
            else:
                formatted = smart_prompt_builder(prompt)
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": formatted}]
                    )
                    answer = response.choices[0].message.content.strip()
                    if len(answer.split()) < 5:
                        raise ValueError("Answer too short")
                except:
                    st.info("🤖 Falling back to Wikipedia...")
                    answer = get_wikipedia_summary(prompt)

                collection.add(
                    documents=[prompt],
                    metadatas=[{"answer": answer}],
                    ids=[str(uuid.uuid4())]
                )

            st.success("✨ Here's your answer!")
            st.markdown(answer)
            speak_text(answer)

# 🔇 Stop voice button
if st.button("🔇 Stop Voice"):
    stop_speech()
