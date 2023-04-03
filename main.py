import openai
import pyaudio
import wave
from google.cloud import texttospeech_v1 as texttospeech
import re
import os
from constants import openai_api_key

openai.api_key = openai_api_key
gpt_model = "text-davinci-003"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "clean-algebra-382603-cbfa070a30bb.json"

def generate_question(prompt):
    print("Generating question...")
    response = openai.Completion.create(
        engine=gpt_model,
        prompt=prompt,
        max_tokens=30,
        # n=1,
        # stop=None,
        temperature=1,
    )
    print("Question generated!")
    question = response.choices[0].text.strip()
    return question

def evaluate_answer(prompt):
    response = openai.Completion.create(
        engine=gpt_model,
        prompt=prompt,
        max_tokens=10,
        # n=1,
        # stop=None,
        temperature=0.5,
    )
    text = response.choices[0].text.strip()
    score = int(re.findall(r'\d+', text)[0])
    return score

def analyze_strengths_weaknesses(prompt):
    response = openai.Completion.create(
        engine=gpt_model,
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )
    analysis = response.choices[0].text.strip()
    return analysis
def record_audio(filename, duration):
    sample_rate = 16000
    channels = 1
    format = pyaudio.paInt16

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    frames = []

    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def transcribe_speech(filename):
    with open(filename, "rb") as f:
        content = f.read()
    response = openai.SpeechToText.asr.create(file=content, model="openai/whisper", sample_rate=16000, encoding="LINEAR16")
    return response.choices[0].text

def text_to_speech(text, filename):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    with open(filename, "wb") as f:
        f.write(response.audio_content)
def play_audio(filename):
    with wave.open(filename, 'rb') as wf:
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(1024)

        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()

        p.terminate()

software_engineering_topics = [
    # "data structures",
    # "algorithms",
    # "object-oriented programming",
    # "operating systems",
    # "databases",
    # "Data Engineering",
    # "Spark",
    # "Spark Job optmization"
    "AWS"
]

interview_score = 0
total_questions = 1
strengths_weaknesses = []
difficulty = "hard"
interview_context = ""

print("AI Interviewer: Software Engineering Questions")
print("----------------------------------------------")

for i, topic in enumerate(software_engineering_topics[:total_questions]):
    question_prompt = f"Generate an interview question about {topic} that you haven't asked me before & is of {difficulty} difficulty."
    question = generate_question(question_prompt)

    tts_filename = f"question_{i+1}.wav"
    text_to_speech(question, tts_filename)
    print(f"Question {i + 1}: {question}")
    play_audio(tts_filename)

    print("Please provide your answer (recording for 10 seconds):")
    audio_filename = f"answer_{i+1}.wav"
    record_audio(audio_filename, duration=30)
    audio_file = open(f"answer_{i+1}.wav", "rb")
    candidate_answer = openai.Audio.transcribe("whisper-1", audio_file)
    print(f"Your answer: {candidate_answer}")

    interview_context += f"Question {i + 1}: {question}\nAnswer {i + 1}: {candidate_answer}\n"

    evaluation_prompt = f"Evaluate the quality of the following answer to the question '{question}': {candidate_answer}. Give a score between 0 (worst) and 10 (best)."
    score = int(evaluate_answer(evaluation_prompt))
    interview_score += score
    print(f"Score: {score}\n")

    analysis_prompt = f"Analyze the strengths and weaknesses in the following answer to the question '{question}': {candidate_answer}."
    analysis = analyze_strengths_weaknesses(analysis_prompt)
    strengths_weaknesses.append(analysis)
    print(f"Analysis:\n {analysis}\n")

# Generate cumulative strength and weakness analysis
cumulative_analysis_prompt = f"Based on the answers provided by the candidate during the interview, provide a cumulative analysis of their overall strengths and weaknesses. The interview context is as follows:\n{interview_context}"
cumulative_analysis = analyze_strengths_weaknesses(cumulative_analysis_prompt)

print("Interview Summary")
print("-----------------")
print(f"Total Score: {interview_score} / {total_questions * 10} \n")

# for i, sw in enumerate(strengths_weaknesses):
#     print(f"Question {i + 1}: {software_engineering_topics[i]}")
#     print(f"Strengths & Weaknesses:\n {sw}\n")

print("Cumulative Strengths & Weaknesses")
print("---------------------------------")
print(cumulative_analysis)