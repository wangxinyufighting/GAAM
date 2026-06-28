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
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "extra_body": extra_body,
        }
        try:
            response = client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except Exception:
            try:
                response = client.chat.completions.create(**request)
            except Exception as exc:
                raise LLMError(f"LLM API error: {exc}") from exc

        try:
            content = response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"LLM API error: {exc}") from exc
        return parse_json_object(content)


@dataclass
class LocalHFChatLLM:
    """Local Hugging Face chat model with the same chat_json interface."""

    model_path: str
    max_new_tokens: int = 2048
    temperature: float = 0.0
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise LLMError(
                "LocalHFChatLLM requires torch and transformers. Install the native VERL "
                "training environment or use the API backend."
            ) from exc

        dtype = self._resolve_dtype(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

    def chat_json(self, system: str, user: str, schema_hint: Optional[str] = None) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return parse_json_object(content)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"

    def _resolve_dtype(self, torch: Any) -> Any:
        value = str(self.torch_dtype).lower()
        if value in {"auto", ""}:
            return "auto"
        if value in {"float16", "fp16"}:
            return torch.float16
        if value in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if value in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype for LocalHFChatLLM: {self.torch_dtype}")


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
