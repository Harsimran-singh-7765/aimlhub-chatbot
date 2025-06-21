# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Gemini Chatbot 💬", layout="centered")

st.title("🌟 Gemini: Your AIML Buddy")
st.write("Ask anything about AIML Club, Projects, Notes, or Career Guidance!")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", key="user_input")

if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})

    response = requests.post("http://localhost:8000/chat", json={"message": user_input})
    print("📡 Raw response text:", response.text)
    print("🛰️ Status code:", response.status_code)

    bot_reply = response.json()["response"]

    st.session_state.messages.append({"role": "bot", "text": bot_reply})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"👤 You: {msg['text']}")
    else:
        st.markdown(f"Gemini: {msg['text']}")
