"""
WebSocket Protocol Handler
Implements the ESP32 WebSocket protocol for Niko AI Assistant
"""

import json
import uuid
import time
import asyncio
import logging
from typing import Optional

import aiohttp
from aiohttp import web

from .opus_codec import OpusDecoder, OpusEncoder, extract_opus_payload, pack_opus_payload
from .asr import create_asr
from .llm import create_llm
from .tts import create_tts
from .vision import create_vision

logger = logging.getLogger(__name__)


class DeviceSession:
    """Manages a single device WebSocket session."""

    def __init__(self, ws: web.WebSocketResponse, config: dict):
        self.ws = ws
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.device_id: str = ""
        self.protocol_version: int = 1

        # Audio config
        audio_cfg = config.get("audio", {})
        self.input_sample_rate = audio_cfg.get("input_sample_rate", 16000)
        self.input_frame_duration = audio_cfg.get("input_frame_duration", 60)
        self.output_sample_rate = audio_cfg.get("output_sample_rate", 24000)
        self.output_frame_duration = audio_cfg.get("output_frame_duration", 60)

        # OPUS codec
        self.decoder = OpusDecoder(
            self.input_sample_rate, 1, self.input_frame_duration
        )
        self.encoder = OpusEncoder(
            self.output_sample_rate, 1, self.output_frame_duration
        )

        # AI modules
        self.asr = create_asr(config.get("asr", {}))
        self.llm = create_llm(config.get("llm", {}))
        self.tts = create_tts(config.get("tts", {}))
        self.vision = create_vision(config.get("vision", {}))

        # Recording buffer
        self.audio_buffer: list[bytes] = []
        self.is_listening = False
        self.listening_mode = "auto"
        self._silence_timer: Optional[asyncio.TimerHandle] = None
        self._audio_frame_count = 0
        self._has_voice = False
        self._silent_frames = 0

        # MCP pending results (id -> Future)
        self._pending_mcp_results: dict[int, asyncio.Future] = {}

        # State
        self.is_speaking = False
        self.abort_speaking = False

        logger.info(f"Session created: {self.session_id}")

    async def initialize(self):
        """Lazy-initialize AI modules in background (non-blocking)."""
        self._initialized = False
        asyncio.create_task(self._lazy_init())

    async def _lazy_init(self):
        """Background AI module initialization."""
        try:
            await self.llm.initialize()
            await self.tts.initialize()
            await self.vision.initialize()
            await self.asr.initialize()
            self._initialized = True
            logger.info("AI modules initialized")
        except Exception as e:
            logger.error(f"AI module init failed: {e}", exc_info=True)

    # ========== Send Messages ==========

    async def send_json(self, data: dict):
        """Send JSON text frame."""
        try:
            await self.ws.send_str(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to send JSON: {e}")

    async def send_audio(self, opus_data: bytes):
        """Send OPUS audio binary frame."""
        try:
            payload = pack_opus_payload(opus_data, self.protocol_version)
            await self.ws.send_bytes(payload)
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")

    # ========== Protocol Handlers ==========

    async def handle_hello(self, msg: dict):
        """Handle device hello message, send hello response."""
        self.protocol_version = msg.get("version", 1)
        features = msg.get("features", {})
        audio_params = msg.get("audio_params", {})

        logger.info(f"Device hello: version={self.protocol_version}, "
                     f"features={features}, audio={audio_params}")

        # Send hello response (must arrive within 10 seconds)
        await self.send_json({
            "type": "hello",
            "transport": "websocket",
            "session_id": self.session_id,
            "audio_params": {
                "sample_rate": self.output_sample_rate,
                "frame_duration": self.output_frame_duration
            }
        })
        logger.info(f"Hello response sent: session_id={self.session_id}")

        # Send MCP initialize to configure vision API URL
        server_cfg = self.config.get("server", {})
        port = server_cfg.get("port", 8000)
        device_ip = server_cfg.get("device_ip", "172.20.10.10")
        await self.send_json({
            "type": "mcp",
            "payload": {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 0,
                "params": {
                    "capabilities": {
                        "vision": {
                            "url": f"http://{device_ip}:{port}/vision/explain",
                            "token": ""
                        }
                    }
                }
            }
        })
        logger.info("MCP initialize sent (vision API configured)")

    async def handle_listen(self, msg: dict):
        """Handle listen message (start/stop recording)."""
        state = msg.get("state", "")
        mode = msg.get("mode", "auto")

        if state == "start":
            self.is_listening = True
            self.listening_mode = mode
            self.audio_buffer.clear()
            self.abort_speaking = True
            logger.info(f"Recording started (mode={mode})")

        elif state == "stop":
            self.is_listening = False
            logger.info(f"Recording stopped, collected {len(self.audio_buffer)} frames")
            asyncio.create_task(self.process_audio())

        elif state == "detect":
            wake_word = msg.get("text", "")
            logger.info(f"Wake word detected: {wake_word}")
            self.is_listening = True
            self.audio_buffer.clear()
            self.abort_speaking = True

    async def handle_abort(self, msg: dict):
        """Handle abort message (user interruption)."""
        reason = msg.get("reason", "")
        logger.info(f"User abort, reason={reason}")
        self.abort_speaking = True

    async def handle_mcp(self, msg: dict):
        """Handle MCP message (tool call results)."""
        payload = msg.get("payload", {})

        logger.info(f"MCP message: {json.dumps(payload, ensure_ascii=False)[:300]}")

        msg_id = payload.get("id")
        if msg_id is not None and "result" in payload:
            future = self._pending_mcp_results.pop(msg_id, None)
            if future and not future.done():
                future.set_result(payload["result"])
                logger.info(f"MCP result delivered: id={msg_id}")
                return

        if msg_id is not None and "error" in payload:
            future = self._pending_mcp_results.pop(msg_id, None)
            if future and not future.done():
                future.set_result({"error": payload["error"]})
                return

        if "result" in payload:
            await self._handle_mcp_result(payload)

    async def _handle_mcp_result(self, payload: dict):
        """Handle MCP tool execution results."""
        result = payload.get("result", {})
        content = result.get("content", [])

        for item in content:
            if item.get("type") == "image":
                import base64
                image_b64 = item.get("data", "")
                if image_b64:
                    image_data = base64.b64decode(image_b64)
                    asyncio.create_task(self.process_vision(image_data))

    # ========== Core Processing Pipeline ==========

    async def handle_audio_frame(self, data: bytes):
        """Process incoming audio binary frame."""
        if not self.is_listening:
            return

        if len(self.audio_buffer) == 0:
            logger.info(f"First audio frame received: {len(data)} bytes")

        opus_data = extract_opus_payload(data, self.protocol_version)
        if not opus_data:
            return

        pcm = self.decoder.decode(opus_data)
        self.audio_buffer.append(pcm)
        self._audio_frame_count += 1

        # Simple VAD: calculate PCM volume (RMS)
        import struct as _struct
        samples = _struct.unpack(f'<{len(pcm)//2}h', pcm)
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5

        if rms > 120:  # Voice detected (noise floor ~80-90)
            self._has_voice = True
            self._silent_frames = 0
        else:
            if self._has_voice:
                self._silent_frames += 1

        # After voice detected, 25 consecutive silent frames (~1.5s) = speech ended
        if self._has_voice and self._silent_frames > 25:
            logger.info(f"VAD: Speech ended, total {self._audio_frame_count} frames")
            self.is_listening = False
            self._audio_frame_count = 0
            self._has_voice = False
            self._silent_frames = 0
            asyncio.create_task(self.process_audio())

        # Max recording 15 seconds (250 frames) to prevent infinite recording
        if self._audio_frame_count > 250:
            logger.info(f"VAD: Max recording time reached, {self._audio_frame_count} frames")
            self.is_listening = False
            self._audio_frame_count = 0
            self._has_voice = False
            self._silent_frames = 0
            asyncio.create_task(self.process_audio())

        if self._audio_frame_count % 50 == 0:
            logger.info(f"Collected {self._audio_frame_count} frames, rms={rms:.0f}, "
                        f"has_voice={self._has_voice}, silent={self._silent_frames}")

    async def process_audio(self):
        """Full pipeline: ASR -> Sensor -> LLM -> TTS."""
        if not self.audio_buffer:
            logger.warning("Audio buffer empty, skipping")
            return

        try:
            # 1. Merge all PCM frames
            pcm_data = b''.join(self.audio_buffer)
            self.audio_buffer.clear()
            logger.info(f"Processing audio: {len(pcm_data)} bytes "
                        f"({len(pcm_data) / 2 / self.input_sample_rate:.1f}s)")

            # 2. ASR speech recognition
            text = await self.asr.transcribe(pcm_data, self.input_sample_rate)
            if not text:
                logger.info("ASR: No recognition result")
                await self.send_json({"type": "tts", "state": "start"})
                await self.send_json({"type": "tts", "state": "stop"})
                return

            # Send STT result to device display
            await self.send_json({"type": "stt", "text": text})

            # 3. Auto-trigger sensors based on speech content
            sensor_context = await self._auto_sensor(text)

            # 4. LLM chat (with sensor context if available)
            self.abort_speaking = False
            llm_input = text
            if sensor_context:
                llm_input = f"{text}\n\n[Sensor Data]\n{sensor_context}"
            reply = await self.llm.chat(llm_input)
            if not reply:
                return

            # 5. TTS + send audio
            await self.speak(reply)

        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)
            try:
                await self.send_json({"type": "tts", "state": "start"})
                await self.send_json({"type": "tts", "state": "stop"})
            except:
                pass
            await self.send_json({
                "type": "alert",
                "status": "Error",
                "message": f"Processing error: {str(e)}",
                "emotion": "sad"
            })

    async def speak(self, text: str):
        """Convert text to speech and send to device."""
        if not text.strip():
            return

        try:
            self.is_speaking = True
            self.abort_speaking = False

            await self.send_json({"type": "tts", "state": "start"})

            sentences = self._split_sentences(text)

            for sentence in sentences:
                if self.abort_speaking:
                    logger.info("TTS interrupted")
                    break

                if not sentence.strip():
                    continue

                await self.send_json({
                    "type": "tts",
                    "state": "sentence_start",
                    "text": sentence
                })

                pcm_data = await self.tts.synthesize_to_pcm(
                    sentence, self.output_sample_rate
                )
                if not pcm_data:
                    continue

                opus_frames = self.encoder.encode_pcm_stream(pcm_data)

                for opus_frame in opus_frames:
                    if self.abort_speaking:
                        break
                    await self.send_audio(opus_frame)
                    await asyncio.sleep(self.output_frame_duration / 1000.0 * 0.8)

            await self.send_json({"type": "tts", "state": "stop"})

        except Exception as e:
            logger.error(f"TTS send error: {e}", exc_info=True)
        finally:
            self.is_speaking = False

    async def process_vision(self, image_data: bytes):
        """Process camera image: vision recognition -> LLM -> TTS."""
        try:
            description = await self.vision.describe(image_data)
            if description:
                context = f"[Camera observation] {description}"
                reply = await self.llm.chat(context)
                if reply:
                    await self.speak(reply)
        except Exception as e:
            logger.error(f"Vision processing error: {e}", exc_info=True)

    # ========== Auto Sensor Invocation ==========

    async def _auto_sensor(self, text: str) -> str:
        """Auto-invoke sensors based on speech content, return context string."""
        context_parts = []

        # Keyword matching (Chinese + English)
        photo_keywords = ["拍照", "看看", "摄像头", "拍一下", "看一下", "前面有什么",
                          "周围", "环境", "场景", "什么东西", "看到", "观察",
                          "photo", "camera", "look", "see", "what's ahead", "surroundings"]
        distance_keywords = ["距离", "多远", "测距", "障碍物", "超声波", "挡着",
                             "前方", "前面有没有", "会撞",
                             "distance", "how far", "obstacle", "ultrasonic", "ahead"]

        need_photo = any(kw in text.lower() for kw in photo_keywords)
        need_distance = any(kw in text.lower() for kw in distance_keywords)

        if need_distance:
            logger.info("Auto-trigger: Ultrasonic distance")
            dist_result = await self._mcp_call_device(
                "self.ultrasound.get_distance", {}
            )
            if dist_result:
                try:
                    import json as _json
                    if isinstance(dist_result, str):
                        dist_result = _json.loads(dist_result)
                    dist_mm = dist_result.get("distance_mm", "unknown")
                    context_parts.append(f"Ultrasonic distance: obstacle at {dist_mm}mm ahead")
                    logger.info(f"Ultrasonic result: {dist_mm}mm")
                except Exception as e:
                    logger.warning(f"Failed to parse distance data: {e}")

        if need_photo:
            logger.info("Auto-trigger: Camera photo")
            photo_result = await self._mcp_call_device(
                "self.camera.take_photo",
                {"question": "Describe the scene, focus on obstacles, people, and road conditions"}
            )
            if photo_result and isinstance(photo_result, str):
                context_parts.append(f"Camera description: {photo_result}")
                logger.info(f"Photo description: {photo_result[:100]}...")

        return "\n".join(context_parts)

    async def _mcp_call_device(self, tool_name: str, arguments: dict, timeout: float = 10.0):
        """Send MCP tool call to device and wait for result."""
        import random
        msg_id = random.randint(10000, 99999)

        future = asyncio.get_event_loop().create_future()
        self._pending_mcp_results[msg_id] = future

        await self.send_json({
            "type": "mcp",
            "payload": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": msg_id,
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
        })

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            content = result.get("content", [])
            for item in content:
                if item.get("type") == "text":
                    return item.get("text", "")
            return result
        except asyncio.TimeoutError:
            self._pending_mcp_results.pop(msg_id, None)
            logger.warning(f"MCP call timeout: {tool_name}")
            return None
        except Exception as e:
            self._pending_mcp_results.pop(msg_id, None)
            logger.error(f"MCP call failed: {tool_name}, {e}")
            return None

    # ========== Utility Methods ==========

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'([。！？；\.\!\?\;])', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            result.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]


async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    """WebSocket connection handler."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    config = request.app["config"]

    device_id = request.headers.get("Device-Id", "unknown")
    protocol_version = int(request.headers.get("Protocol-Version", "1"))

    logger.info(f"WebSocket connected: device={device_id}, version={protocol_version}")

    session = DeviceSession(ws, config)
    session.device_id = device_id
    session.protocol_version = protocol_version

    await session.initialize()

    from .app import active_sessions
    active_sessions[device_id] = session

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    logger.info(f"Text message: type={msg_type}, data={json.dumps(data, ensure_ascii=False)[:200]}")

                    if msg_type == "hello":
                        await session.handle_hello(data)
                    elif msg_type == "listen":
                        await session.handle_listen(data)
                    elif msg_type == "abort":
                        await session.handle_abort(data)
                    elif msg_type == "mcp":
                        await session.handle_mcp(data)
                    else:
                        logger.debug(f"Unhandled message type: {msg_type}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {msg.data[:100]}")

            elif msg.type == aiohttp.WSMsgType.BINARY:
                await session.handle_audio_frame(msg.data)

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")

    except Exception as e:
        logger.error(f"WebSocket handler exception: {e}", exc_info=True)
    finally:
        active_sessions.pop(device_id, None)
        logger.info(f"WebSocket disconnected: device={device_id}")

    return ws
