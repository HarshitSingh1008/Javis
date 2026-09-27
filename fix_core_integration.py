with open('E:/Jarvis 2.0/venv/nexus_overlay/core_integration.py', 'rb') as f:
    content = f.read()

# Update _process_text_input to be async and return response
old_method = b'    def _process_text_input(self, text: str) -> None:\r\n\r\n        """\r\n        Route transcribed (or typed) text through the orchestrator. The\r\n        orchestrator drives the thinking->speaking cycle; the overlay just\r\n        reflects whatever the orchestrator reports.\r\n        """\r\n\r\n        if not self._orchestrator:\r\n            return\r\n        try:\r\n            # Orchestrator uses run() method, not submit()\r\n            import asyncio\r\n            loop = asyncio.get_event_loop()\r\n            if loop.is_running():\r\n                # Schedule in the running loop\r\n                asyncio.create_task(self._orchestrator.run(text))\r\n            else:\r\n                loop.run_until_complete(self._orchestrator.run(text))\r\n        except Exception as e:\r\n            logger.warning("Overlay: submit text to orchestrator failed: %s", e)\r\n            self._ws_server.send_state("error")'

new_method = b'    async def _process_text_input(self, text: str) -> str:\r\n\r\n        """\r\n        Route transcribed (or typed) text through the orchestrator. The\r\n        orchestrator drives the thinking->speaking cycle; the overlay just\r\n        reflects whatever the orchestrator reports.\r\n        Returns the final response text.\r\n        """\r\n\r\n        if not self._orchestrator:\r\n            return ""\r\n        try:\r\n            # Orchestrator uses run() method\r\n            result = await self._orchestrator.run(text)\r\n            return result.get("final_response", "")\r\n        except Exception as e:\r\n            logger.warning("Overlay: submit text to orchestrator failed: %s", e)\r\n            self._ws_server.send_state("error")\r\n            return ""'

if old_method in content:
    content = content.replace(old_method, new_method)
    print('Updated _process_text_input')
else:
    print('Old method pattern not found')

with open('E:/Jarvis 2.0/venv/nexus_overlay/core_integration.py', 'wb') as f:
    f.write(content)

print('Done')