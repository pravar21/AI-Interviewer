# AI Interviewer

AI Interviewer is a command-line tool that conducts mock technical interviews
end-to-end. It uses OpenAI's large language models to generate questions,
evaluate answers, and produce feedback while relying on Google Cloud Text to
Speech and your local microphone to deliver and capture audio. The application
creates a realistic interview experience without requiring a human interviewer.

## Features

- **Dynamic question generation** powered by OpenAI's GPT models with
  configurable topics and difficulty.
- **Audio-first experience** with text-to-speech delivery of questions and
  microphone recording of your answers.
- **Automatic transcription** of recorded responses using OpenAI Whisper.
- **Immediate scoring and analysis** for each answer as well as a cumulative
  summary of strengths and weaknesses.

## Prerequisites

- Python 3.9+ (the application was built and tested against CPython)
- An OpenAI account with an API key
- A Google Cloud project with Text-to-Speech enabled and a service account key
  JSON file
- A microphone and speakers/headphones connected to your computer
- System libraries required by [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
  (e.g., `portaudio`)

## Installation

1. **Clone the repository** and navigate into it:

   ```bash
   git clone <your fork or clone URL>
   cd AI-Interviewer
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install openai google-cloud-texttospeech pyaudio
   ```

   > **Note:** Installing PyAudio on some platforms requires additional system
   > packages. Refer to the PyAudio documentation for platform-specific
   > instructions.

## Configuration

1. **OpenAI credentials** – Populate `constants.py` with your API key:

   ```python
   # constants.py
   openai_api_key = "sk-your-openai-key"
   ```

2. **Google Cloud credentials** – Create or download a service account key with
   access to the Text-to-Speech API and save the JSON file in the project root.
   Update `GOOGLE_CREDENTIALS_PATH` in `main.py` to match the filename if it
   differs from the placeholder value. The application automatically sets the
   `GOOGLE_APPLICATION_CREDENTIALS` environment variable at runtime using this
   path.

3. **Audio hardware** – Ensure your system microphone and speakers are working.
   The default configuration records responses for 30 seconds at 16 kHz mono.
   You can adjust these parameters in the `InterviewConfig` dataclass in
   `main.py` if needed.

## Running the interviewer

After configuration, run the CLI application from the project root:

```bash
python main.py
```

The program will:

1. Generate an interview question for each topic in the configuration (default:
   one question about AWS).
2. Convert each question to speech and play it through your speakers.
3. Record your verbal response for the configured duration.
4. Transcribe your answer with Whisper, provide a score, and offer analysis.
5. Summarize your overall performance after the interview concludes.

## Customization

- **Topics & difficulty** – Update the `InterviewConfig` passed to `run_interview`
  in `main.py` to change the list of topics, difficulty (`easy`, `medium`,
  `hard`, etc.), question limit, and timing.
- **Alternative voices** – Modify the `VoiceSelectionParams` in
  `SpeechInterface.synthesize_speech` to experiment with different voices or
  languages supported by Google Cloud Text-to-Speech.
- **Model selection** – Adjust the `GPT_MODEL` constant in `main.py` to use a
  different OpenAI completion model that fits your use case and account access.

## Troubleshooting

- **`pyaudio` installation errors** – Install the required system dependency
  `portaudio` using your package manager (e.g., `brew install portaudio` on
  macOS or `sudo apt-get install portaudio19-dev` on Debian/Ubuntu) before
  running `pip install pyaudio`.
- **Authentication failures** – Verify that your OpenAI API key and Google
  credentials are valid and the JSON file is accessible at the path specified
  in `main.py`.
- **Microphone issues** – Confirm that your OS permissions allow Python to
  access the microphone and that the input device is selected correctly at the
  system level.

## License

This project is provided as-is without a specific license. Please review the
OpenAI and Google Cloud terms of service for usage restrictions on their APIs.
