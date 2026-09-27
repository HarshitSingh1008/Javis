import json

with open('E:/Jarvis 2.0/venv/nexus_overlay/websocket_server.py', 'rb') as f:
    content = f.read()

# Replace set_input_callback signature and _handle_incoming_message
old_callback = b'def set_input_callback(self, callback: Callable[[str], None]) -> None:\r\n        """Set callback for text input from overlay."""\r\n        self._input_callback = callback'

new_callback = b'def set_input_callback(self, callback: Callable[[str], Awaitable[str]]) -> None:\r\n        """Set async callback for text input from overlay. Returns response."""\r\n        self._input_callback = callback'

if old_callback in content:
    content = content.replace(old_callback, new_callback)
    print('Updated set_input_callback')
else:
    print('Old callback pattern not found')

# Update _handle_incoming_message to await callback and send response
old_handle = b'        if msg_type == "input":\r\n            text = data.get("text", "").strip()\r\n            if text:\r\n                logger.debug("Overlay text input: %s", text[:80])\r\n                # Notify integration to process text (orchestrator will be invoked)\r\n                if self._input_callback:\r\n                    try:\r\n                        self._input_callback(text)\r\n                    except Exception as e:\r\n                        logger.error("Input callback error: %s", e)'

new_handle = b'        if msg_type == "input":\r\n            text = data.get("text", "").strip()\r\n            if text:\r\n                logger.debug("Overlay text input: %s", text[:80])\r\n                # Notify integration to process text and await response\r\n                if self._input_callback:\r\n                    try:\r\n                        response = await self._input_callback(text)\r\n                        if response:\r\n                            await self._send_to_client(websocket, {\r\n                                "type": "response",\r\n                                "text": response\r\n                            })\r\n                    except Exception as e:\r\n                        logger.error("Input callback error: %s", e)'

if old_handle in content:
    content = content.replace(old_handle, new_handle)
    print('Updated _handle_incoming_message')
else:
    print('Old handle pattern not found')

# Add _send_to_client method before set_input_callback
old_set_input = b'    def set_input_callback(self, callback: Callable[[str], Awaitable[str]]) -> None:\r\n        """Set async callback for text input from overlay. Returns response."""\r\n        self._input_callback = callback'

new_set_input = b'    async def _send_to_client(self, websocket, message: dict) -> None:\r\n        """Send a message to a specific client."""\r\n        try:\r\n            await websocket.send(json.dumps(message))\r\n        except Exception as e:\r\n            logger.debug("Failed to send to client: %s", e)\r\n\r\n    def set_input_callback(self, callback: Callable[[str], Awaitable[str]]) -> None:\r\n        """Set async callback for text input from overlay. Returns response."""\r\n        self._input_callback = callback'

if old_set_input in content:
    content = content.replace(old_set_input, new_set_input)
    print('Added _send_to_client method')
else:
    print('Set input pattern not found')

with open('E:/Jarvis 2.0/venv/nexus_overlay/websocket_server.py', 'wb') as f:
    f.write(content)

print('Done')