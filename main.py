"""Command line entry point for the AI interviewer workflow."""

from __future__ import annotations

import os
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import openai
import pyaudio
from google.cloud import texttospeech_v1 as texttospeech

from constants import openai_api_key

openai.api_key = openai_api_key
GPT_MODEL = "text-davinci-003"
GOOGLE_CREDENTIALS_PATH = "clean-algebra-382603-cbfa070a30bb.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH


@dataclass
class InterviewConfig:
    """Configuration values used to conduct an interview."""

    topics: List[str]
    difficulty: str = "hard"
    question_limit: int = 1
    answer_record_seconds: int = 30
    sample_rate: int = 16000
    channels: int = 1
    frames_per_buffer: int = 1024


@dataclass
class InterviewContext:
    """Collects question and answer pairs for downstream analysis."""

    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    def add_entry(self, question: str, answer: str) -> None:
        self.questions.append(question)
        self.answers.append(answer)

    def as_prompt_context(self) -> str:
        """Render the interview history as a text block for LLM prompts."""

        context_lines: List[str] = []
        for index, (question, answer) in enumerate(zip(self.questions, self.answers), start=1):
            context_lines.append(f"Question {index}: {question}")
            context_lines.append(f"Answer {index}: {answer}")
        return "\n".join(context_lines)


class OpenAIInterviewClient:
    """Handles text generation, scoring, and analysis with OpenAI models."""

    def __init__(self, model: str = GPT_MODEL) -> None:
        self.model = model

    def generate_question(self, topic: str, difficulty: str, asked_questions: Iterable[str]) -> str:
        prompt = self._build_question_prompt(topic, difficulty, asked_questions)
        print("Generating question...")
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=30,
            temperature=1,
        )
        question = response.choices[0].text.strip()
        print("Question generated!")
        return question

    def evaluate_answer(self, question: str, answer: str) -> int:
        prompt = (
            "Evaluate the quality of the following answer to the question "
            f"'{question}': {answer}. "
            "Give a score between 0 (worst) and 10 (best)."
        )
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=10,
            temperature=0.5,
        )
        text = response.choices[0].text.strip()
        score = int(re.findall(r"\d+", text)[0])
        return score

    def analyze_answer(self, question: str, answer: str) -> str:
        prompt = (
            "Analyze the strengths and weaknesses in the following answer to the "
            f"question '{question}': {answer}."
        )
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.5,
        )
        return response.choices[0].text.strip()

    def summarize_interview(self, context: InterviewContext) -> str:
        prompt = (
            "Based on the answers provided by the candidate during the interview, "
            "provide a cumulative analysis of their overall strengths and "
            "weaknesses. The interview context is as follows:\n"
            f"{context.as_prompt_context()}"
        )
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.5,
        )
        return response.choices[0].text.strip()

    @staticmethod
    def _build_question_prompt(
        topic: str, difficulty: str, asked_questions: Iterable[str]
    ) -> str:
        asked_fragment = "\n".join(f"- {question}" for question in asked_questions)
        avoid_clause = (
            "Ensure the question is unique and not present in the following list:\n"
            f"{asked_fragment}"
            if asked_fragment
            else ""
        )
        return (
            "Generate an interview question about "
            f"{topic} that you haven't asked me before and is of {difficulty} difficulty. "
            f"{avoid_clause}"
        ).strip()


class SpeechInterface:
    """Provides audio recording, transcription, and text-to-speech helpers."""

    def __init__(self, config: InterviewConfig) -> None:
        self.config = config
        self._text_to_speech_client = texttospeech.TextToSpeechClient()

    def record_audio(self, filename: Path) -> None:
        sample_rate = self.config.sample_rate
        channels = self.config.channels
        audio_format = pyaudio.paInt16

        p = pyaudio.PyAudio()
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self.config.frames_per_buffer,
        )

        frames = []
        frame_count = int(sample_rate / self.config.frames_per_buffer * self.config.answer_record_seconds)

        for _ in range(frame_count):
            data = stream.read(self.config.frames_per_buffer)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(str(filename), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(p.get_sample_size(audio_format))
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(frames))

    @staticmethod
    def transcribe_audio(filename: Path) -> str:
        with open(filename, "rb") as audio_file:
            return openai.Audio.transcribe("whisper-1", audio_file)

    def synthesize_speech(self, text: str, filename: Path) -> None:
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        response = self._text_to_speech_client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        with open(filename, "wb") as audio_file:
            audio_file.write(response.audio_content)

    @staticmethod
    def play_audio(filename: Path) -> None:
        with wave.open(str(filename), "rb") as wav_file:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=audio.get_format_from_width(wav_file.getsampwidth()),
                channels=wav_file.getnchannels(),
                rate=wav_file.getframerate(),
                output=True,
            )

            data = wav_file.readframes(1024)
            while data:
                stream.write(data)
                data = wav_file.readframes(1024)

            stream.stop_stream()
            stream.close()
            audio.terminate()


class InterviewSession:
    """Coordinates the end-to-end interview flow."""

    def __init__(
        self,
        config: InterviewConfig,
        ai_client: OpenAIInterviewClient,
        speech_interface: SpeechInterface,
    ) -> None:
        self.config = config
        self.ai_client = ai_client
        self.speech_interface = speech_interface
        self.context = InterviewContext()
        self.scores: List[int] = []

    def conduct(self) -> None:
        print("AI Interviewer: Software Engineering Questions")
        print("----------------------------------------------")

        asked_questions: List[str] = []

        for index, topic in enumerate(self.config.topics[: self.config.question_limit], start=1):
            question = self.ai_client.generate_question(
                topic, self.config.difficulty, asked_questions
            )
            asked_questions.append(question)

            tts_filename = Path(f"question_{index}.wav")
            self.speech_interface.synthesize_speech(question, tts_filename)
            print(f"Question {index}: {question}")
            self.speech_interface.play_audio(tts_filename)

            print(
                "Please provide your answer (recording for "
                f"{self.config.answer_record_seconds} seconds):"
            )
            answer_filename = Path(f"answer_{index}.wav")
            self.speech_interface.record_audio(answer_filename)
            candidate_answer = self.speech_interface.transcribe_audio(answer_filename)
            print(f"Your answer: {candidate_answer}")

            self.context.add_entry(question, candidate_answer)
            score = self.ai_client.evaluate_answer(question, candidate_answer)
            self.scores.append(score)
            print(f"Score: {score}\n")

            analysis = self.ai_client.analyze_answer(question, candidate_answer)
            print(f"Analysis:\n{analysis}\n")

        self._summarize_results()

    def _summarize_results(self) -> None:
        total_score = sum(self.scores)
        max_score = self.config.question_limit * 10

        print("Interview Summary")
        print("-----------------")
        print(f"Total Score: {total_score} / {max_score}\n")

        cumulative_analysis = self.ai_client.summarize_interview(self.context)
        print("Cumulative Strengths & Weaknesses")
        print("---------------------------------")
        print(cumulative_analysis)


def run_interview() -> None:
    """Entry point to conduct a single interview session."""

    config = InterviewConfig(topics=["AWS"])
    ai_client = OpenAIInterviewClient()
    speech_interface = SpeechInterface(config)
    session = InterviewSession(config, ai_client, speech_interface)
    session.conduct()


if __name__ == "__main__":
    run_interview()
