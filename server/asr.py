"""
ASR (Automatic Speech Recognition) Module
- funasr: Local FunASR (Alibaba DAMO Academy, excellent for Chinese)
- whisper: Local faster-whisper
- aliyun: Alibaba Cloud Paraformer (cloud)
"""

import io
import json
import wave
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FunASR:
    """Local FunASR speech recognition (Alibaba DAMO Academy)."""

    def __init__(self, config: dict):
        # Model options:
        #   paraformer-zh - Chinese offline recognition (recommended)
        #   SenseVoiceSmall - Multilingual, supports emotion detection
        self.model_name = config.get("model", "paraformer-zh")
        self.device = config.get("device", "cuda")
        self.model = None

    async def initialize(self):
        if self.model is not None:
            return

        from funasr import AutoModel
        logger.info(f"Loading FunASR model: {self.model_name} on {self.device}")
        self.model = AutoModel(
            model=self.model_name,
            device=self.device,
            vad_model="fsmn-vad",
            punc_model="ct-punc",
        )
        logger.info("FunASR model loaded")

    async def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> str:
        """PCM data -> text."""
        await self.initialize()

        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        result = self.model.generate(
            input=audio_array,
            batch_size_s=300,
        )

        text = ""
        if result and len(result) > 0:
            for item in result:
                text += item.get("text", "")

        text = text.strip()
        logger.info(f"FunASR result: {text}")
        return text


class WhisperASR:
    """Local faster-whisper speech recognition."""

    def __init__(self, config: dict):
        self.model_size = config.get("model", "base")
        self.device = config.get("device", "cuda")
        self.language = config.get("language", "en")
        self.model = None

    async def initialize(self):
        if self.model is not None:
            return

        from faster_whisper import WhisperModel
        compute_type = "float16" if self.device == "cuda" else "int8"
        logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute_type
        )
        logger.info("Whisper model loaded")

    async def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> str:
        """PCM data -> text."""
        await self.initialize()

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        wav_buffer.seek(0)

        segments, info = self.model.transcribe(
            wav_buffer, language=self.language, beam_size=5, vad_filter=True
        )

        text = "".join(segment.text for segment in segments).strip()
        logger.info(f"Whisper ASR result: {text}")
        return text


class AliyunASR:
    """Alibaba Cloud ASR (DashScope fun-asr-realtime, streaming mode)."""

    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "fun-asr-realtime")
        self.language = config.get("language", "en")

    async def initialize(self):
        if not self.api_key:
            logger.warning("Alibaba Cloud API key not configured!")
        else:
            import dashscope
            dashscope.api_key = self.api_key
            logger.info(f"Aliyun ASR initialized: model={self.model}, language={self.language}")

    async def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> str:
        """PCM data -> text (via DashScope streaming recognition)."""
        import asyncio
        import threading
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

        dashscope.api_key = self.api_key

        model = self.model
        lang = self.language
        chunk_size = 3200  # bytes per frame (100ms at 16kHz, 16bit mono)

        class _Callback(RecognitionCallback):
            def __init__(self):
                self.sentences = []
                self.error = None
                self.done = threading.Event()

            def on_open(self):
                logger.debug("Aliyun ASR connection opened")

            def on_event(self, result: RecognitionResult):
                sentence = result.get_sentence()
                if sentence and 'text' in sentence:
                    if RecognitionResult.is_sentence_end(sentence):
                        self.sentences.append(sentence['text'])
                        logger.debug(f"Aliyun ASR sentence: {sentence['text']}")

            def on_complete(self):
                logger.debug("Aliyun ASR recognition complete")
                self.done.set()

            def on_error(self, result):
                self.error = str(result)
                logger.error(f"Aliyun ASR error: {self.error}")
                self.done.set()

            def on_close(self):
                self.done.set()

        def _recognize():
            callback = _Callback()
            recognition = Recognition(
                model=model,
                format='pcm',
                sample_rate=sample_rate,
                semantic_punctuation_enabled=False,
                callback=callback,
                language_hints=[lang],
            )

            recognition.start()

            # Send PCM data in chunks
            offset = 0
            while offset < len(pcm_data):
                end = min(offset + chunk_size, len(pcm_data))
                recognition.send_audio_frame(pcm_data[offset:end])
                offset = end

            recognition.stop()
            callback.done.wait(timeout=30)

            if callback.error:
                raise Exception(f"Aliyun ASR error: {callback.error}")

            return "".join(callback.sentences)

        try:
            text = await asyncio.get_event_loop().run_in_executor(None, _recognize)
            text = text.strip()
            logger.info(f"Aliyun ASR result: {text}")
            return text
        except Exception as e:
            logger.error(f"Aliyun ASR exception: {e}", exc_info=True)
            return ""


def create_asr(config: dict):
    """Create ASR instance from config."""
    provider = config.get("provider", "funasr")
    if provider == "funasr":
        return FunASR(config.get("funasr", {}))
    elif provider == "whisper":
        return WhisperASR(config.get("whisper", {}))
    elif provider == "aliyun":
        return AliyunASR(config.get("aliyun", {}))
    else:
        raise ValueError(f"Unsupported ASR provider: {provider}")
