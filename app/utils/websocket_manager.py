import asyncio
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Maps call_id -> set of subscriber WebSockets
        self.live_analysis_subscribers: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect_analysis(self, call_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            if call_id not in self.live_analysis_subscribers:
                self.live_analysis_subscribers[call_id] = set()
            self.live_analysis_subscribers[call_id].add(websocket)
        logger.info(f"WebSocket client connected for call_id: {call_id}")

    async def disconnect_analysis(self, call_id: str, websocket: WebSocket):
        async with self._lock:
            if call_id in self.live_analysis_subscribers:
                self.live_analysis_subscribers[call_id].discard(websocket)
                if not self.live_analysis_subscribers[call_id]:
                    del self.live_analysis_subscribers[call_id]
        logger.info(f"WebSocket client disconnected for call_id: {call_id}")

    async def broadcast_to_call(self, call_id: str, message: dict):
        subscribers = set()
        async with self._lock:
            if call_id in self.live_analysis_subscribers:
                subscribers = set(self.live_analysis_subscribers[call_id])

        disconnected = set()
        for websocket in subscribers:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending message to client on call {call_id}: {e}")
                disconnected.add(websocket)

        if disconnected:
            async with self._lock:
                if call_id in self.live_analysis_subscribers:
                    self.live_analysis_subscribers[call_id].difference_update(disconnected)

manager = ConnectionManager()
