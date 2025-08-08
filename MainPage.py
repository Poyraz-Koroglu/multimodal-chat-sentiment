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
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        max-width: 80%;
    }

    .user-message {
        background-color: #667eea;
        color: white;
        margin-left: auto;
        text-align: right;
    }

    .bot-message {
        background-color: #f1f3f4;
        color: #333;
        border: 1px solid #e0e0e0;
    }

    .sentiment-positive {
        background-color: #4caf50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .sentiment-negative {
        background-color: #f44336;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .stAudio {
        margin: 1rem 0;
    }

    .status-message {
        text-align: center;
        color: #666;
        font-style: italic;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your sentiment analysis assistant. Type a message or upload voice recording to get started. I'll analyze whether your sentiment is positive or negative! 😊",
            "timestamp": datetime.now(),
            "sentiment": None
        }
    ]

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""


def mock_sentiment_analysis(text):
    """
    Mock sentiment analysis function - replace this with your actual model
    """
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                      'happy', 'joy', 'love', 'like', 'awesome', 'perfect', 'brilliant', 'outstanding']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'hate',
                      'dislike', 'frustrated', 'disappointed', 'worried', 'stressed', 'upset', 'annoyed']

    words = text.lower().split()
    positive_score = sum(1 for word in words if any(pos in word for pos in positive_words))
    negative_score = sum(1 for word in words if any(neg in word for neg in negative_words))

    # Add some randomness for demo purposes
    random_factor = random.choice([1, -1]) if positive_score == negative_score else 0

    if positive_score > negative_score or (positive_score == negative_score and random_factor > 0):
        return "Positive"
    else:
        return "Negative"


def process_audio_file(audio_file):
    """
    Process uploaded audio file and convert to text using speech recognition
    """
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_path = temp_file.name

        # Initialize recognizer
        recognizer = sr.Recognizer()

        # Convert audio to text
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Clean up temporary file
        os.unlink(temp_path)

        return text, None

    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please try again with clearer speech."
    except sr.RequestError as e:
        return None, f"Error with speech recognition service: {e}"
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
    """Display all chat messages"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                    {f'<div class="sentiment-{message["sentiment"].lower()}">{message["sentiment"]} Sentiment</div>' if message["sentiment"] else ""}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {message["content"]}
                </div>
            """, unsafe_allow_html=True)


def process_user_message(user_input):
    """Process user message and generate bot response"""
    if not user_input.strip():
        return

    # Analyze sentiment
    sentiment = mock_sentiment_analysis(user_input)

    # Add user message
    add_message("user", user_input, sentiment)

    # Generate bot response based on sentiment
    if sentiment == "Positive":
        bot_response = f"Great! I can sense the positive vibes in your message. Your sentiment appears to be uplifting and optimistic! 😊"
    else:
        bot_response = f"I notice some negative sentiment in your message. Would you like to talk about what's bothering you? 🤗"

    # Add bot response
    add_message("assistant", bot_response)


# Main app layout
st.markdown("""
<div class="main-header">
    <h1>💬 Sentiment Analysis Chatbot</h1>
    <p>Share your thoughts and discover their sentiment through text or voice</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for input methods
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Text Input")
    # Text input
    user_input = st.text_area(
        "Type your message here:",
        height=100,
        placeholder="Share your thoughts, feelings, or any message you'd like me to analyze..."
    )

    if st.button("Send Message", type="primary", use_container_width=True):
        if user_input:
            process_user_message(user_input)
            st.rerun()

with col2:
    st.subheader("🎤 Voice Input")
    # Audio file uploader
    audio_file = st.file_uploader(
        "Upload audio file:",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Upload an audio file to convert speech to text"
    )

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Process Audio", type="secondary", use_container_width=True):
            with st.spinner("Processing audio..."):
                text, error = process_audio_file(audio_file)

                if text:
                    st.success(f"Recognized text: '{text}'")
                    process_user_message(text)
                    st.rerun()
                else:
                    st.error(error)

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Audio recording instructions
with st.expander("🎙️ How to Record Audio"):
    st.write("""
    **For voice input, you have several options:**

    1. **Mobile devices**: Use your phone's voice recorder app, then upload the file
    2. **Desktop**: Use apps like:
        - Windows: Voice Recorder app
        - macOS: QuickTime Player or Voice Memos
        - Online: recordmp3.org or vocaroo.com

    **Supported formats**: WAV, MP3, M4A, OGG

    **Tips for better recognition**:
    - Speak clearly and at moderate pace
    - Minimize background noise
    - Keep recordings under 1 minute for faster processing
    """)

# Display chat messages
st.markdown("---")
st.subheader("💭 Chat History")

# Container for scrollable chat
chat_container = st.container()
with chat_container:
    display_messages()

# Clear chat button
if st.button("🗑️ Clear Chat History", type="secondary"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your sentiment analysis assistant. Type a message or upload voice recording to get started. I'll analyze whether your sentiment is positive or negative! 😊",
            "timestamp": datetime.now(),
            "sentiment": None
        }
    ]
    st.rerun()

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot analyzes the sentiment of your messages and categorizes them as either **Positive** or **Negative**.

    **Features:**
    - 💬 Text input analysis
    - 🎤 Voice-to-text conversion
    - 📊 Real-time sentiment analysis
    - 💾 Chat history

    **Current Status:** Demo Mode
    (Using mock sentiment analysis)
    """)

    st.header("📈 Session Stats")
    if len(st.session_state.messages) > 1:  # Exclude initial message
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        if user_messages:
            positive_count = sum(1 for msg in user_messages if msg.get("sentiment") == "Positive")
            negative_count = sum(1 for msg in user_messages if msg.get("sentiment") == "Negative")

            st.metric("Messages Sent", len(user_messages))
            st.metric("Positive Sentiment", positive_count)
            st.metric("Negative Sentiment", negative_count)

    st.header("🛠️ Setup Instructions")
    with st.expander("Installation"):
        st.code("""
pip install streamlit
pip install speechrecognition
pip install pydub
pip install pyaudio  # For microphone input
        """)

    with st.expander("Run Application"):
        st.code("streamlit run app.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit | Ready for ML model integration
</div>
""", unsafe_allow_html=True)