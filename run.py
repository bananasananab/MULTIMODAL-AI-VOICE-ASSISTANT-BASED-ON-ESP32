"""
Niko AI Assistant - Server Entry Point
"""

import os
import re
import sys
import logging
import yaml
from pathlib import Path
from aiohttp import web

from server.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def _load_dotenv(env_path: str = ".env"):
    """Load .env file into os.environ (simple key=value, no quotes handling)."""
    path = Path(env_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    logger.info(f"Loaded environment from {env_path}")


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR} placeholders in config values from os.environ."""
    if isinstance(obj, str):
        # Replace ${VAR} or ${VAR:-default} patterns
        def _replace(match):
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
            else:
                var_name, default = var_expr, match.group(0)
            return os.environ.get(var_name, default)
        return re.sub(r'\$\{([^}]+)\}', _replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration file, resolving ${ENV_VAR} placeholders."""
    # Load .env first (same directory as config)
    env_path = Path(config_path).parent / ".env"
    _load_dotenv(str(env_path))

    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve ${VAR} placeholders from environment
    config = _resolve_env_vars(config)

    logger.info(f"Config loaded: {config_path}")
    logger.info(f"  ASR:    {config.get('asr', {}).get('provider', '?')}")
    logger.info(f"  LLM:    {config.get('llm', {}).get('provider', '?')}")
    logger.info(f"  TTS:    {config.get('tts', {}).get('provider', '?')}")
    logger.info(f"  Vision: {config.get('vision', {}).get('provider', '?')}")
    return config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8000)

    app = create_app(config)

    logger.info("=" * 50)
    logger.info("Niko AI Assistant Server Started")
    logger.info(f"Listening: http://{host}:{port}")
    logger.info(f"OTA URL:   http://{host}:{port}/niko/ota/")
    logger.info(f"WebSocket: ws://{host}:{port}/ws")
    logger.info(f"Monitor:   http://{host}:{port}/monitor")
    logger.info("=" * 50)
    logger.info(f"Set ESP32 firmware OTA_URL to:")
    logger.info(f"  http://<your-pc-ip>:{port}/niko/ota/")
    logger.info("=" * 50)

    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
