# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""WebSocket server that exposes a policy to robot clients using msgpack frames."""

import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        # Robots connect over the network, so the default listens on every interface.
        host: str = "0.0.0.0",  # nosec B104
        port: int = 10093,
        idle_timeout: int = -1,  # 新增参数，单位秒，-1表示永不关闭
        metadata: dict | None = None,
    ) -> None:
        """Create the server; nothing listens until `serve_forever` is called.

        Args:
            policy: Object with `predict_action(**payload)` and, optionally, `reset()`.
            host: Interface to listen on.
            port: TCP port to listen on.
            idle_timeout: Seconds without a message before the server shuts down; -1 never shuts down.
            metadata: Dict sent to every client right after it connects.
        """
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Run the server in a new event loop until it stops."""
        asyncio.run(self.run())

    async def run(self):
        """Serve connections until the idle timeout closes the server, or forever without one."""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """Close the server once no message has arrived for `idle_timeout` seconds."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # 每次收到消息刷新活跃时间
                ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: object) -> dict:
        """Route one client message to the policy and return the reply.

        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        # Basic payload sanity: checked before any msg.get(), which a non-dict message would raise on.
        # A message without a type is an infer request, hence "inference_result".
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": "default",
                "error": {"message": "Payload must be a dict", "payload_type": str(type(msg))},
            }

        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            try:
                # Strip routing fields — only pass observation payload to the model
                payload = {k: v for k, v in msg.items() if k not in ("type", "request_id")}
                # If the client sends {"examples": [...]}, pass through as-is.
                # Otherwise (robot client sends observations as a flat dict), wrap it.
                if "examples" not in payload:
                    payload = {"examples": [payload]}
                output_dict = self._policy.predict_action(**payload)

            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                **output_dict,  # spreads e.g. "actions": array at the top level
            }

        # reset --> clear smoothing state between episodes
        elif mtype == "reset":
            try:
                if hasattr(self._policy, "reset"):
                    self._policy.reset()
                return {"status": "ok", "ok": True, "type": "reset", "request_id": req_id}
            except Exception as e:
                logging.exception("Policy reset error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }

        # unknown request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
