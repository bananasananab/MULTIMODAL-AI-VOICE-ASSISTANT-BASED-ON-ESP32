"""
LLM (Large Language Model) Module
- ollama: Local Ollama
- cloud: OpenAI compatible API
"""

import json
import logging
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Local Ollama LLM."""

    def __init__(self, config: dict, system_prompt: str):
        self.url = config.get("url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5")
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []

    async def initialize(self):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info(f"Ollama available models: {models}")
                if self.model not in models and f"{self.model}:latest" not in models:
                    logger.warning(f"Model {self.model} not found, please run: ollama pull {self.model}")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")

    def reset_conversation(self):
        self.conversation_history = []

    async def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False}
            )
            resp.raise_for_status()
            result = resp.json()
            reply = result.get("message", {}).get("content", "")

        self.conversation_history.append({"role": "assistant", "content": reply})
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

        logger.info(f"Ollama reply: {reply[:100]}...")
        return reply

    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        full_reply = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", f"{self.url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_reply += token
                            yield token
                    except json.JSONDecodeError:
                        continue

        self.conversation_history.append({"role": "assistant", "content": full_reply})
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]


class CloudLLM:
    """Cloud OpenAI compatible API."""

    def __init__(self, config: dict, system_prompt: str):
        self.url = config.get("url", "https://api.openai.com/v1")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-4o")
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []

    async def initialize(self):
        pass

    def reset_conversation(self):
        self.conversation_history = []

    async def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages, "stream": False}
            )
            resp.raise_for_status()
            result = resp.json()
            reply = result["choices"][0]["message"]["content"]

        self.conversation_history.append({"role": "assistant", "content": reply})
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

        logger.info(f"Cloud LLM reply: {reply[:100]}...")
        return reply

    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        full_reply = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", f"{self.url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        token = data["choices"][0].get("delta", {}).get("content", "")
                        if token:
                            full_reply += token
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        self.conversation_history.append({"role": "assistant", "content": full_reply})
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]


def create_llm(config: dict):
    """Create LLM instance from config."""
    provider = config.get("provider", "ollama")
    system_prompt = config.get("system_prompt", "You are an AI assistant.")

    if provider == "ollama":
        return OllamaLLM(config.get("ollama", {}), system_prompt)
    elif provider == "cloud":
        return CloudLLM(config.get("cloud", {}), system_prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
