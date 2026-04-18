"""
TTS (Text-to-Speech) Module
- edge: Microsoft Edge TTS (free)
"""

import io
import logging
import edge_tts

logger = logging.getLogger(__name__)


class EdgeTTS:
    """Microsoft Edge TTS synthesis."""

    def __init__(self, config: dict):
        self.voice = config.get("voice", "en-US-GuyNeural")
        self.rate = config.get("rate", "+0%")

    async def initialize(self):
        logger.info(f"Edge TTS initialized: voice={self.voice}, rate={self.rate}")

    async def synthesize(self, text: str) -> bytes:
        """Text -> MP3 bytes (with retry)."""
        if not text.strip():
            return b''

        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(
                    text=text, voice=self.voice, rate=self.rate
                )

                mp3_buffer = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_buffer.write(chunk["data"])

                mp3_data = mp3_buffer.getvalue()
                if mp3_data:
                    logger.debug(f"TTS synthesized: '{text[:30]}...' -> {len(mp3_data)} bytes MP3")
                    return mp3_data
            except Exception as e:
                logger.warning(f"TTS attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1)

        logger.error(f"TTS synthesis failed: {text[:30]}...")
        return b''

    async def synthesize_to_pcm(self, text: str, target_sample_rate: int = 24000) -> bytes:
        """Text -> PCM (16-bit signed LE, mono) using PyAV."""
        mp3_data = await self.synthesize(text)
        if not mp3_data:
            return b''

        import av
        import numpy as np

        container = av.open(io.BytesIO(mp3_data), format='mp3')
        audio_stream = container.streams.audio[0]

        frames = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            frames.append(arr)
        container.close()

        if not frames:
            return b''

        audio_data = np.concatenate(frames, axis=1) if frames[0].ndim > 1 else np.concatenate(frames)

        if audio_data.ndim > 1:
            audio_data = audio_data[0]

        src_rate = audio_stream.rate or 24000

        if src_rate != target_sample_rate:
            num_samples = int(len(audio_data) * target_sample_rate / src_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data) - 1, num_samples),
                np.arange(len(audio_data)),
                audio_data.astype(np.float64)
            )

        if audio_data.dtype in (np.float32, np.float64):
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

        pcm_data = audio_data.tobytes()
        logger.info(f"TTS PCM: '{text[:30]}...' -> {len(pcm_data)} bytes "
                     f"({len(pcm_data) / 2 / target_sample_rate:.1f}s)")
        return pcm_data


class Pyttsx3TTS:
    """Local offline TTS using system voices (SAPI5 on Windows, espeak on Linux)."""

    def __init__(self, config: dict):
        self.voice_id = config.get("voice_id", "") or None
        self.rate = config.get("rate", 150)

    async def initialize(self):
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        logger.info(f"pyttsx3 available voices:")
        for v in voices:
            logger.info(f"  {v.id} - {v.name}")
        engine.stop()
        logger.info(f"pyttsx3 TTS initialized: voice_id={self.voice_id}, rate={self.rate}")

    async def synthesize(self, text: str) -> bytes:
        """Text -> WAV bytes."""
        if not text.strip():
            return b''

        import pyttsx3
        import tempfile
        import os
        import asyncio

        voice_id = self.voice_id
        rate = self.rate

        def _synth():
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            if voice_id:
                engine.setProperty('voice', voice_id)
            tmp_path = tempfile.mktemp(suffix='.wav')
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()
            with open(tmp_path, 'rb') as f:
                wav_data = f.read()
            os.unlink(tmp_path)
            return wav_data

        wav_data = await asyncio.get_event_loop().run_in_executor(None, _synth)
        return wav_data

    async def synthesize_to_pcm(self, text: str, target_sample_rate: int = 24000) -> bytes:
        """Text -> PCM (16-bit signed LE, mono)."""
        import wave
        import numpy as np

        wav_data = await self.synthesize(text)
        if not wav_data:
            return b''

        wav_buf = io.BytesIO(wav_data)
        with wave.open(wav_buf, 'rb') as wf:
            src_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            pcm_raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(pcm_raw, dtype=np.int16)

        # Convert to mono if stereo
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]

        # Resample if needed
        if src_rate != target_sample_rate:
            num_samples = int(len(audio) * target_sample_rate / src_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, num_samples),
                np.arange(len(audio)),
                audio.astype(np.float64)
            ).astype(np.int16)

        pcm_data = audio.tobytes()
        logger.info(f"pyttsx3 PCM: '{text[:30]}...' -> {len(pcm_data)} bytes "
                     f"({len(pcm_data) / 2 / target_sample_rate:.1f}s)")
        return pcm_data


def create_tts(config: dict):
    """Create TTS instance from config."""
    provider = config.get("provider", "edge")
    if provider == "edge":
        return EdgeTTS(config.get("edge", {}))
    elif provider == "pyttsx3":
        return Pyttsx3TTS(config.get("pyttsx3", {}))
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")
