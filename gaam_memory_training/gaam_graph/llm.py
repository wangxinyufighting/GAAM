from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()


class LLMError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleLLM:
    model: str = os.getenv("GAAM_LLM_MODEL", os.getenv("DEEPSEEK_MODEL", os.getenv("LLM_MODEL", "deepseek-v4-flash")))
    api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY"))
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    timeout: int = 120
    temperature: float = 0.0

    def chat_json(self, system: str, user: str, schema_hint: Optional[str] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("DEEPSEEK_API_KEY is not set. Use --no-llm for dry run.")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout,
        )
        extra_body = None
        if "deepseek" in self.base_url.lower():
            extra_body = {"thinking": {"type": "disabled"}}
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                stream=False,
                extra_body=extra_body,
            )
        except OpenAIError as exc:
            raise LLMError(f"LLM API error: {exc}") from exc

        content = response.choices[0].message.content or ""
        return parse_json_object(content)


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))
