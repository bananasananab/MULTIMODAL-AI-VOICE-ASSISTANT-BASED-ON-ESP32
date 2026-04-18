"""
Vision (Multimodal LLM) Module
- ollama: Local Ollama (llava, gemma4, minicpm-v, etc.)
- cloud: OpenAI compatible API (GPT-4o, etc.)
"""

import base64
import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaVision:
    """Local Ollama vision model."""

    def __init__(self, config: dict):
        self.url = config.get("url", "http://localhost:11434")
        self.model = config.get("model", "llava")

    async def initialize(self):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.url}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                if self.model not in models and f"{self.model}:latest" not in models:
                    logger.warning(f"Vision model {self.model} not found, please run: ollama pull {self.model}")
        except Exception as e:
            logger.error(f"Ollama Vision connection failed: {e}")

    async def describe(self, image_data: bytes, question: str = None) -> str:
        """Image -> text description."""
        if question is None:
            question = "Describe this image. Focus on objects, people, obstacles, and the overall scene."

        b64_image = base64.b64encode(image_data).decode('utf-8')

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": question,
                    "images": [b64_image],
                    "stream": False
                }
            )
            resp.raise_for_status()
            result = resp.json()
            description = result.get("response", "")

        logger.info(f"Vision description: {description[:100]}...")
        return description


class CloudVision:
    """Cloud vision API (OpenAI compatible)."""

    def __init__(self, config: dict):
        self.url = config.get("url", "https://api.openai.com/v1")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-4o")

    async def initialize(self):
        pass

    async def describe(self, image_data: bytes, question: str = None) -> str:
        """Image -> text description."""
        if question is None:
            question = "Describe this image. Focus on objects, people, obstacles, and the overall scene."

        b64_image = base64.b64encode(image_data).decode('utf-8')

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                        ]
                    }],
                    "max_tokens": 300
                }
            )
            resp.raise_for_status()
            result = resp.json()
            description = result["choices"][0]["message"]["content"]

        logger.info(f"Cloud Vision description: {description[:100]}...")
        return description


def create_vision(config: dict):
    """Create Vision instance from config."""
    provider = config.get("provider", "ollama")
    if provider == "ollama":
        return OllamaVision(config.get("ollama", {}))
    elif provider == "cloud":
        return CloudVision(config.get("cloud", {}))
    else:
        raise ValueError(f"Unsupported Vision provider: {provider}")
