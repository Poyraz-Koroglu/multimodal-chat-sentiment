import streamlit as st
import speech_recognition as sr
import pydub
from pydub import AudioSegment
import io
import tempfile
import os
import random
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Sentiment Analysis Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }

    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        max-width: 80%;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background-color: #667eea;
        color: white;
        margin-left: auto;
        text-align: right;
    }

    .bot-message {
        background-color: white;
        color: #333;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .sentiment-badge {
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .sentiment-positive {
        background-color: #4caf50;
        color: white;
    }

    .sentiment-negative {
        background-color: #f44336;
        color: white;
    }

    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e0e0e0;
        z-index: 100;
    }

    .stTextArea textarea {
        border-radius: 20px;
        border: 2px solid #667eea;
        padding: 15px;
        font-size: 16px;
    }

    .stTextArea textarea:focus {
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }

    /* Hide streamlit header and footer */
    header[data-testid="stHeader"] {
        display: none;
    }

    .stDeployButton {
        display: none;
    }

    footer {
        display: none;
    }

    /* Adjust main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
        max-width: 800px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your sentiment analysis assistant. Type a message below and I'll analyze whether your sentiment is positive or negative! 😊",
            "timestamp": datetime.now(),
            "sentiment": None
        }
    ]


import requests

FASTAPI_URL = "http://127.0.0.1:8000/predict"  # or your server URL

def get_sentiment_from_api(text=None, audio_file=None):
    """
    Call FastAPI backend for sentiment analysis.
    """
    try:
        files = {}
        data = {}

        if text:
            data["text"] = text

        if audio_file:
            files["audio"] = (audio_file.name, audio_file, audio_file.type)

        response = requests.post(FASTAPI_URL, data=data, files=files)
        response.raise_for_status()
        result = response.json()

        # API returns { "prediction": 0 or 1 }
        return "Negative" if result["prediction"] == 1 else "Positive"

    except Exception as e:
        return f"API error: {e}"


def process_audio_file(audio_file):
    """
    Send uploaded audio file to FastAPI backend for sentiment analysis.
    """
    try:
        files = {"audio": (audio_file.name, audio_file, audio_file.type)}
        response = requests.post(FASTAPI_URL, files=files)
        response.raise_for_status()

        result = response.json()
        sentiment = "Negative" if result["prediction"] == 1 else "Positive"
        return sentiment, None

    except requests.exceptions.RequestException as e:
        return None, f"API request error: {e}"
    except Exception as e:
        return None, f"Error processing audio: {e}"



def add_message(role, content, sentiment=None):
    """Add a message to the chat history"""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(),
        "sentiment": sentiment
    })


def display_messages():
    """Display all chat messages in a scrollable container"""
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                sentiment_badge = ""
                if message["sentiment"]:
                    sentiment_class = "sentiment-positive" if message[
                                                                  "sentiment"] == "Positive" else "sentiment-negative"
                    sentiment_badge = f'<div class="sentiment-badge {sentiment_class}">{message["sentiment"]} Sentiment</div>'

                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                        {sentiment_badge}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def process_user_message(user_input):
    """Process user message and generate bot response"""
    if not user_input.strip():
        return

    # Call backend instead of mock
    sentiment = get_sentiment_from_api(text=user_input)

    # Add user message
    add_message("user", user_input, sentiment)

    # Generate bot response based on sentiment
    if sentiment == "Positive":
        responses = [
            "Great! I can sense the positive vibes in your message. 😊",
            "That's wonderful! Keep that great energy going! ✨",
        ]
    else:
        responses = [
            "I notice some negative sentiment in your message. 🤗",
            "It seems you’re going through something challenging. 💙",
        ]

    bot_response = random.choice(responses)
    add_message("assistant", bot_response)


# Main app layout
st.markdown("""
<div class="main-header">
    <h2>💬 Sentiment Analysis Chatbot</h2>
    <p>Share your thoughts and discover their sentiment</p>
</div>
""", unsafe_allow_html=True)

# Display chat messages
display_messages()

# Input section at the bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Create columns for text input and buttons
input_col, button_col = st.columns([4, 1])

with input_col:
    user_input = st.text_area(
        "",
        height=80,
        placeholder="Type your message here and press Ctrl+Enter or click Send...",
        label_visibility="collapsed",
        key="user_input"
    )

with button_col:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    send_button = st.button("Send", type="primary", use_container_width=True)

    # Audio upload option
    audio_file = st.file_uploader(
        "",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Upload audio file",
        label_visibility="collapsed"
    )

# Process input
if send_button and user_input:
    process_user_message(user_input)
    # Clear the input by using session state
    st.session_state.user_input = ""
    st.rerun()

# Process audio
if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    if st.button("🎤 Process Audio"):
        with st.spinner("Processing audio..."):
            text, error = process_audio_file(audio_file)

            if text:
                st.success(f"Recognized: '{text}'")
                process_user_message(text)
                time.sleep(1)  # Brief pause to show success message
                st.rerun()
            else:
                st.error(error)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with minimal info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot analyzes the sentiment of your messages as **Positive** or **Negative**.

    **Features:**
    - 💬 Real-time sentiment analysis
    - 🎤 Voice message support
    - 📱 Mobile-friendly interface
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your sentiment analysis assistant. Type a message below and I'll analyze whether your sentiment is positive or negative! 😊",
                "timestamp": datetime.now(),
                "sentiment": None
            }
        ]
        st.rerun()

# Add JavaScript for better UX
st.markdown("""
<script>
// Auto-scroll to bottom of chat
function scrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Call scroll function after page load
setTimeout(scrollToBottom, 100);

// Handle Enter key in textarea
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        const sendButton = document.querySelector('button[kind="primary"]');
        if (sendButton) {
            sendButton.click();
        }
    }
});
</script>
""", unsafe_allow_html=True)