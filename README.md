# Athena Voice Assistant

Athena is a high-performance, fully local voice assistant designed for Linux systems. It operates as a ذكاء اصطناعي محلي مثل سيري للابتوب, providing a seamless and instant user experience without relying on cloud services.

## Key Features

*   **Zero-Latency Audio Pipeline:** Utilizes native PyAudio memory streaming combined with silence padding and auto-trimming for 0ms response times.
*   **Offline Operation:** Completely local inference for wake-word detection, speech-to-text, semantic routing, and text-to-speech.
*   **Dynamic Pill UI:** A sleek, non-intrusive PyQt5 graphical interface that adapts dynamically to the assistant's state (Idle, Listening, Speaking, Downloading).
*   **Smart Semantic Routing:** Uses Sentence-Transformers to understand user intent rather than relying solely on exact keyword matches.
*   **Advanced Audio Handling:** Solves the classic "Echo Bug" (Self-Listening) through precise microphone blocking during audio playback.
*   **Human-Like Voice Responses:** Integrates Kokoro TTS for natural, high-fidelity speech and supports instant predefined audio responses directly from RAM.

## Core Technologies

*   Python 3
*   PyQt5 (Graphical User Interface)
*   OpenWakeWord (Wake-word detection)
*   Faster-Whisper (Speech-to-Text)
*   Kokoro ONNX (Text-to-Speech)
*   PyAudio & Soundfile (Native Audio Streaming & Processing)

## System Architecture

The architecture is built heavily around background threading and memory management to prevent UI freezing and audio latency. Audio files like the wake confirmation are loaded directly into RAM upon initialization, allowing the system to bypass ALSA wake-up delays and disk read times.

## Setup and Installation

1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/athena-voice-assistant.git

2. Navigate to the project directory:
   cd athena-voice-assistant

3. Run the assistant:
   python run_ai.py