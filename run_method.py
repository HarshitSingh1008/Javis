async def run(
            self,
            user_input: str,
            session_context: Optional[Dict[str, Any]] = None,
        ) -> NexusState:
            """
            Run a complete agent cycle from input to response.

            This is the main entry point. It executes all graph nodes in order:
            classify -> context -> reason -> (tool_loop) -> memory -> format.

            Args:
                user_input: The user's message.
                session_context: Optional context from the session (working memory, history).

            Returns:
                Final NexusState with response, metrics, and execution record.

            Note:
                This method is thread-safe for concurrent calls up to max_concurrent_runs.
                Excess calls will wait for a slot to become available.
            """
            start = time.perf_counter()

            # Concurrency control - acquire slot
            async with self._run_lock:
                while self._active_runs >= self._max_concurrent_runs:
                    await asyncio.sleep(0.1)
                self._active_runs += 1

            try:
                # Create initial state
                working_memory = session_context.get("working_memory", []) if session_context else []
                conversation_history = session_context.get("history", []) if session_context else []

                state = make_initial_state(
                    user_input=user_input,
                    session_id=str(uuid.uuid4()),
                    input_type="text",
                    working_memory=working_memory,
                    conversation_history=conversation_history,
                )

                # ---- Node: Intent Classification ----
                state = await self._run_intent_classify(state)

                # ---- Intent short-circuit: skip tool loop for pure chat only ----
                # Only "chat" intent (greetings, conversation) skips the tool loop.
                # "simple_command" still needs tools (e.g., open notepad, check ram).
                intent = state.get("intent", "")
                if intent == "chat":
                    logger.info("Intent 'chat' -- skipping tool loop, routing to direct response")
                    state = await self._run_agent_reason(state)
                    state = await self._run_memory_write(state)
                    state = await self._run_response_format(state)
                    state["total_duration_ms"] = (time.perf_counter() - start) * 1000
                    return state

                # ---- LAW 1.1: Playbook Match ----
                # Check if the goal matches a known playbook before entering the
                # general ReAct loop. If matched, execute the playbook instead.
                state = await self._run_playbook_match(state)
                if state.get("playbook_name"):
                    # Playbook was matched and executed -- skip to memory write
                    logger.info("Playbook '%s' completed, skipping to memory write", state["playbook_name"])
                    state = await self._run_memory_write(state)
                    state = await self._run_response_format(state)
                    state["total_duration_ms"] = (time.perf_counter() - start) * 1000
                    return state

                # ---- Node: Context Loading ----
                state = await self._run_context_load(state)

                # ---- Agent Reasoning + Tool Execution Loop ----
                # This is the main ReAct loop: reason -> act -> observe -> reason...
                max_iterations = self._settings.AGENT_MAX_ITERATIONS

                while state.get("iteration_count", 0) < max_iterations:
                    state["iteration_count"] = state.get("iteration_count", 0) + 1

                    # Check for errors from previous steps
                    errors = state.get("errors", [])
                    if len(errors) > 0 and state.get("iteration_count", 0) > max_iterations // 2:
                        logger.warning("Too many errors, terminating agent loop.")
                        break

                    # ---- Node: Agent Reasoning ----
                    state = await self._run_agent_reason(state)

                    # Check if we have a final response (agent decided no tool needed)
                    if state.get("final_response"):
                        break

                    # ---- Node: Tool Selection ----
                    tool_name, tool_input, is_ui = self._extract_tool_call(state)

                    if not tool_name:
                        # Agent chose to respond directly (not use a tool)
                        break

                    # ---- Node: Tool Execution ----
                    state = await self._run_tool_execute(state, tool_name, tool_input, is_ui)

                    # ---- Self-Healing: Capability Synthesis ----
                    gap = self._detect_capability_gap(state)
                    if gap:
                        state["gap_encountered"] = gap
                        state = await self._run_capability_synthesis(state, user_input, gap)

                        # If synthesis succeeded, retry the tool
                        if state.get("synthesis_triggered") and state.get("gap_recovery_tool"):
                            state = await self._run_tool_execute(
                                state,
                                state["gap_recovery_tool"],
                                tool_input,
                                is_ui,
                            )

                # ---- Node: Memory Write ----
                state = await self._run_memory_write(state)

                # ---- Node: Response Format ----
                state = await self._run_response_format(state)

                # Finalize
                state["total_duration_ms"] = (time.perf_counter() - start) * 1000

                # Log completion
                self._audit_logger.log(
                    event_type="AGENT_STEP",
                    data={
                        "intent": state.get("intent"),
                        "duration_ms": state["total_duration_ms"],
                        "llm_calls": state.get("llm_calls", 0),
                        "synthesis_triggered": state.get("synthesis_triggered", False),
                        "iteration_count": state.get("iteration_count", 0),
                    },
                    module="nexus_brain.orchestrator",
                    function_name="run",
                    duration_ms=state["total_duration_ms"],
                    success=bool(state.get("final_response")),
                )

                return state
            finally:
                # Release concurrency slot
                async with self._run_lock:
                    self._active_runs = max(0, self._active_runs - 1)