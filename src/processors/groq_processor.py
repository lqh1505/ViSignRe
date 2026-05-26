"""Groq LLM integration for sentence refinement."""

import os
import logging
import json
from groq import Groq
from dotenv import load_dotenv
from config import Config

load_dotenv()

class GroqProcessor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.client = Groq(api_key=api_key)

    @staticmethod
    def _labels_to_display(word_list: list) -> list:
        return [
            Config.LABEL_DISPLAY.get(w, w)
            for w in word_list
            if Config.LABEL_DISPLAY.get(w, w)
        ]

    def _call_llm(self, prompt: str) -> str:
        try:
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=256,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            logging.error("Groq API error: %s", e)
            return ""

    def enhance_transcript(self, word_list: list) -> dict:
        if not word_list:
            return {"enhanced": ""}

        display_words = self._labels_to_display(word_list)
        if not display_words:
            return {"enhanced": ""}

        prompt = Config.LLM_PROMPT_ENHANCE.format(
            constraint=Config.get_llm_constraint(),
            words=", ".join(display_words),
        )
        result = self._call_llm(prompt)
        return {"enhanced": result if result else " ".join(display_words)}

    def generate_sentence(self, word_list: list) -> dict:
        if not word_list:
            return {"sentence": "", "explanation": None, "params": {}}

        display_words = self._labels_to_display(word_list)
        fallback = " ".join(display_words) if display_words else " ".join(word_list)
        default_params = {"gender": "nam"}

        prompt = Config.LLM_PROMPT_GENERATE.format(
            constraint=Config.get_llm_constraint(),
            words=", ".join(display_words),
        )
        raw = self._call_llm(prompt)

        try:
            clean_json = raw.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()
            result_dict = json.loads(clean_json)

            return {
                "sentence": result_dict.get("sentence", fallback),
                "explanation": result_dict.get("explanation"),
                "params": result_dict.get("params", default_params),
            }
        except json.JSONDecodeError:
            logging.error("Invalid JSON from Groq: %s", raw)
            return {
                "sentence": fallback,
                "explanation": "JSON parse error",
                "params": default_params,
            }
