"""
Main Application - HTTP OTA endpoint + WebSocket routing + Monitor panel
"""

import json
import time
import asyncio
import logging
from pathlib import Path
from aiohttp import web

from .protocol import handle_websocket

logger = logging.getLogger(__name__)

# Global active device sessions (for API calls)
active_sessions = {}


async def handle_ota(request: web.Request) -> web.Response:
    """OTA endpoint: POST /niko/ota/ - Returns WebSocket connection info."""
    config = request.app["config"]
    server_cfg = config.get("server", {})
    device_ip = server_cfg.get("device_ip", request.host.split(":")[0])
    port = server_cfg.get("port", 8000)
    ws_path = server_cfg.get("ws_path", "/ws")

    device_id = request.headers.get("Device-Id", "unknown")
    client_id = request.headers.get("Client-Id", "unknown")
    user_agent = request.headers.get("User-Agent", "unknown")

    body = {}
    if request.content_type == "application/json":
        try:
            body = await request.json()
        except Exception:
            pass

    logger.info(f"OTA request: device={device_id}, client={client_id}, ua={user_agent}")
    if body:
        app_info = body.get("application", {})
        logger.info(f"  Firmware: {app_info.get('name', '?')} v{app_info.get('version', '?')}")

    response_data = {
        "websocket": {
            "url": f"ws://{device_ip}:{port}{ws_path}",
            "token": "",
            "version": 1
        },
        "server_time": {
            "timestamp": int(time.time() * 1000),
            "timezone_offset": 480
        }
    }

    logger.info(f"OTA response: ws_url={response_data['websocket']['url']}")
    return web.json_response(response_data)


async def handle_ota_get(request: web.Request) -> web.Response:
    """Handle GET OTA requests."""
    return await handle_ota(request)


async def handle_vision_api(request: web.Request) -> web.Response:
    """Vision API endpoint: POST /vision/describe"""
    config = request.app["config"]
    from .vision import create_vision
    vision = create_vision(config.get("vision", {}))

    data = await request.read()
    question = request.query.get("question", None)

    try:
        description = await vision.describe(data, question)
        return web.json_response({"description": description})
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_vision_explain(request: web.Request) -> web.Response:
    """
    Vision explain endpoint: POST /vision/explain
    ESP32 camera POSTs multipart/form-data with question and JPEG image here.
    """
    config = request.app["config"]
    from .vision import create_vision
    vision = create_vision(config.get("vision", {}))

    try:
        reader = await request.multipart()
        question = None
        image_data = None

        async for part in reader:
            if part.name == "question":
                question = (await part.read()).decode('utf-8')
            elif part.name == "file":
                image_data = await part.read()

        if not image_data:
            return web.Response(text='{"success": false, "message": "No image"}', status=400)

        # Save latest image for monitor panel
        import base64
        request.app["latest_image"] = base64.b64encode(image_data).decode('utf-8')

        description = await vision.describe(image_data, question)
        request.app["latest_description"] = description
        logger.info(f"Vision explain: question={question}, desc={description[:100]}")
        return web.Response(text=description, content_type='text/plain')

    except Exception as e:
        logger.error(f"Vision explain error: {e}", exc_info=True)
        return web.Response(text=f'{{"success": false, "message": "{str(e)}"}}', status=500)


async def handle_latest_image(request: web.Request) -> web.Response:
    """Return the latest captured image and description."""
    image = request.app.get("latest_image", "")
    desc = request.app.get("latest_description", "")
    return web.json_response({"image": image, "description": desc})


async def handle_mcp_call(request: web.Request) -> web.Response:
    """MCP tool call API: POST /api/mcp/call - Monitor panel calls device MCP tools."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    tool_name = data.get("tool", "")
    arguments = data.get("arguments", {})
    msg_id = data.get("id", 1)

    if not tool_name:
        return web.json_response({"error": "Missing tool name"}, status=400)

    if not active_sessions:
        return web.json_response({"error": "No device connected"}, status=503)

    session = list(active_sessions.values())[0]

    future = asyncio.get_event_loop().create_future()
    session._pending_mcp_results[msg_id] = future

    mcp_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": msg_id,
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    await session.send_json({
        "type": "mcp",
        "payload": mcp_request
    })

    logger.info(f"MCP request sent: tool={tool_name}, id={msg_id}")

    try:
        result = await asyncio.wait_for(future, timeout=10.0)
        logger.info(f"MCP result: {json.dumps(result, ensure_ascii=False)[:200]}")

        response = {"result": result}
        content = result.get("content", [])
        for item in content:
            if item.get("type") == "image":
                response["image"] = item.get("data", "")
            elif item.get("type") == "text":
                text = item.get("text", "")
                response["result"] = text
                try:
                    response["result"] = json.loads(text)
                except:
                    pass

        # Trigger TTS announcement on ESP32 in background
        if tool_name in ("self.camera.take_photo", "self.ultrasound.get_distance"):
            asyncio.create_task(_announce_on_device(session, tool_name, result))

        return web.json_response(response)

    except asyncio.TimeoutError:
        session._pending_mcp_results.pop(msg_id, None)
        return web.json_response({"error": "MCP call timeout"}, status=504)
    except Exception as e:
        session._pending_mcp_results.pop(msg_id, None)
        return web.json_response({"error": str(e)}, status=500)


async def _announce_on_device(session, tool_name: str, mcp_result: dict):
    """After monitor triggers a sensor, send result through LLM -> TTS -> ESP32."""
    try:
        content = mcp_result.get("content", [])
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        text = "\n".join(text_parts).strip()

        if tool_name == "self.camera.take_photo":
            if text:
                llm_input = f"[Camera observation] {text}"
            else:
                llm_input = "[Camera observation] Image captured but no description available."
        elif tool_name == "self.ultrasound.get_distance":
            try:
                dist_data = json.loads(text) if text else {}
                dist_mm = dist_data.get("distance_mm", "unknown")
                llm_input = f"[Sensor Data]\nUltrasonic distance: obstacle at {dist_mm}mm ahead"
            except (json.JSONDecodeError, AttributeError):
                llm_input = f"[Sensor Data]\nUltrasonic reading: {text}"
        else:
            return

        reply = await session.llm.chat(llm_input)
        if reply:
            await session.speak(reply)

    except Exception as e:
        logger.error(f"Announce on device error: {e}", exc_info=True)


async def handle_monitor(request: web.Request) -> web.Response:
    """Monitor panel page."""
    html_path = Path(__file__).parent / "static" / "monitor.html"
    return web.FileResponse(html_path)


def create_app(config: dict) -> web.Application:
    """Create aiohttp application."""
    app = web.Application()
    app["config"] = config

    ws_path = config.get("server", {}).get("ws_path", "/ws")

    # Routes
    app.router.add_post("/niko/ota/", handle_ota)
    app.router.add_get("/niko/ota/", handle_ota_get)
    app.router.add_post("/niko/ota", handle_ota)
    app.router.add_get("/niko/ota", handle_ota_get)
    app.router.add_get(ws_path, handle_websocket)
    app.router.add_post("/vision/describe", handle_vision_api)
    app.router.add_post("/vision/explain", handle_vision_explain)
    app.router.add_get("/api/latest_image", handle_latest_image)
    app.router.add_post("/api/mcp/call", handle_mcp_call)
    app.router.add_get("/monitor", handle_monitor)

    # Health check
    app.router.add_get("/", lambda r: web.json_response({
        "service": "niko-ai-server",
        "status": "running"
    }))

    logger.info("Routes registered:")
    logger.info(f"  OTA:       POST /niko/ota/")
    logger.info(f"  WebSocket: GET  {ws_path}")
    logger.info(f"  Vision:    POST /vision/explain")
    logger.info(f"  MCP API:   POST /api/mcp/call")
    logger.info(f"  Monitor:   GET  /monitor")
    logger.info(f"  Health:    GET  /")

    return app
