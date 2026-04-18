"""
OPUS Codec Module
- Decode OPUS audio frames from ESP32 to PCM
- Encode PCM to OPUS audio frames for ESP32
"""

import os
import sys
import struct
import ctypes
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Ensure opus.dll is found before importing opuslib
# Search conda environment paths first
_opus_search_paths = [
    os.path.join(sys.prefix, "Library", "bin"),           # conda env
    os.path.join(os.path.dirname(sys.executable), "Library", "bin"),
    os.path.dirname(os.path.abspath(__file__)),            # project dir
]
for _path in _opus_search_paths:
    _opus_dll = os.path.join(_path, "opus.dll")
    if os.path.exists(_opus_dll):
        os.environ["PATH"] = _path + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Found opus.dll: {_opus_dll}")
        break

try:
    import opuslib
    HAS_OPUSLIB = True
except (ImportError, OSError, Exception) as e:
    HAS_OPUSLIB = False
    logger.warning(f"opuslib unavailable: {e}")
    logger.warning("Silent audio will be used. Please install opuslib and libopus.")


class OpusDecoder:
    """Decode OPUS frames from ESP32 -> PCM (16-bit signed, mono)"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 frame_duration: int = 60):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = frame_duration
        # Samples per frame
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.decoder = None

        if HAS_OPUSLIB:
            self.decoder = opuslib.Decoder(sample_rate, channels)
            logger.info(f"OPUS decoder initialized: {sample_rate}Hz, {channels}ch, "
                        f"{frame_duration}ms, frame_size={self.frame_size}")

    def decode(self, opus_data: bytes) -> bytes:
        """Decode a single OPUS frame, return PCM (16-bit signed LE)."""
        if self.decoder is None:
            # Return silent frame when no decoder available
            return b'\x00' * (self.frame_size * self.channels * 2)

        try:
            pcm = self.decoder.decode(opus_data, self.frame_size)
            return pcm
        except Exception as e:
            logger.error(f"OPUS decode error: {e}")
            return b'\x00' * (self.frame_size * self.channels * 2)


class OpusEncoder:
    """Encode PCM -> OPUS frames for ESP32"""

    def __init__(self, sample_rate: int = 24000, channels: int = 1,
                 frame_duration: int = 60, bitrate: int = 32000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.encoder = None

        if HAS_OPUSLIB:
            self.encoder = opuslib.Encoder(
                sample_rate, channels,
                opuslib.APPLICATION_AUDIO
            )
            self.encoder.bitrate = bitrate
            logger.info(f"OPUS encoder initialized: {sample_rate}Hz, {channels}ch, "
                        f"{frame_duration}ms, bitrate={bitrate}")

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode a single PCM frame to OPUS."""
        if self.encoder is None:
            return b''

        try:
            opus_data = self.encoder.encode(pcm_data, self.frame_size)
            return opus_data
        except Exception as e:
            logger.error(f"OPUS encode error: {e}")
            return b''

    def encode_pcm_stream(self, pcm_data: bytes) -> list[bytes]:
        """
        Split PCM data into frames and encode each one.
        Returns list of OPUS frames.
        """
        bytes_per_frame = self.frame_size * self.channels * 2  # 16-bit = 2 bytes
        opus_frames = []
        offset = 0

        while offset + bytes_per_frame <= len(pcm_data):
            frame_pcm = pcm_data[offset:offset + bytes_per_frame]
            opus_frame = self.encode(frame_pcm)
            if opus_frame:
                opus_frames.append(opus_frame)
            offset += bytes_per_frame

        # Handle remaining data shorter than one frame: pad with zeros
        if offset < len(pcm_data):
            remaining = pcm_data[offset:]
            padded = remaining + b'\x00' * (bytes_per_frame - len(remaining))
            opus_frame = self.encode(padded)
            if opus_frame:
                opus_frames.append(opus_frame)

        logger.debug(f"PCM {len(pcm_data)} bytes -> {len(opus_frames)} OPUS frames")
        return opus_frames


def extract_opus_payload(data: bytes, protocol_version: int = 1) -> bytes:
    """
    Extract OPUS payload from WebSocket binary frame.
    Parse different frame header formats based on protocol version.
    """
    if protocol_version == 1:
        # V1: No header, entire frame is OPUS data
        return data

    elif protocol_version == 2:
        # V2: 16-byte header
        # [version:2][type:2][reserved:4][timestamp:4][payload_size:4][payload:N]
        if len(data) < 16:
            logger.warning(f"V2 frame too short: {len(data)} bytes")
            return b''
        payload_size = struct.unpack('!I', data[12:16])[0]
        return data[16:16 + payload_size]

    elif protocol_version == 3:
        # V3: 4-byte header
        # [type:1][reserved:1][payload_size:2][payload:N]
        if len(data) < 4:
            logger.warning(f"V3 frame too short: {len(data)} bytes")
            return b''
        payload_size = struct.unpack('!H', data[2:4])[0]
        return data[4:4 + payload_size]

    else:
        logger.warning(f"Unknown protocol version: {protocol_version}")
        return data


def pack_opus_payload(opus_data: bytes, protocol_version: int = 1) -> bytes:
    """
    Pack OPUS data into WebSocket binary frame.
    """
    if protocol_version == 1:
        return opus_data

    elif protocol_version == 2:
        header = struct.pack('!HHIII',
                             2,                  # version
                             0,                  # type = OPUS audio
                             0,                  # reserved
                             0,                  # timestamp
                             len(opus_data))     # payload_size
        return header + opus_data

    elif protocol_version == 3:
        header = struct.pack('!BBH',
                             0,                  # type = OPUS audio
                             0,                  # reserved
                             len(opus_data))     # payload_size
        return header + opus_data

    else:
        return opus_data
