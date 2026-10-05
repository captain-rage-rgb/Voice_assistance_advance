"""
Core Agent Runtime orchestrating events, action queues, execution, context, and state.
"""

import asyncio
import logging
from typing import Optional
from app.agent.action import Action, ActionStatus
from app.agent.action_queue import ActionQueue
from app.agent.cancellation import CancellationToken
from app.agent.context import ActionResult, ExecutionOutcome, Observation, TaskContext
from app.agent.events import Event, EventBus, EventType
from app.agent.executor import ActionExecutor, MockExecutor
from app.agent.validator import ActionValidator
from app.agent.verifier import Verifier
from app.config.settings import settings
from app.state.state_manager import AgentLifecycleState, StateManager


class AgentRuntime:
    """
    Asynchronous Core Agent Runtime (Phases 1-8).
    Coordinates event dispatch, state transitions, action queueing,
    context-aware planning, post-action verification, bounded recovery,
    unified cancellation tokens, and graceful shutdown.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[StateManager] = None,
        action_queue: Optional[ActionQueue] = None,
        executor: Optional[ActionExecutor] = None,
        validator: Optional[ActionValidator] = None,
        verifier: Optional[Verifier] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.state_manager = state_manager or StateManager()
        self.action_queue = action_queue or ActionQueue()
        self.executor = executor or MockExecutor()
        self.logger = logger or logging.getLogger("voice_agent.runtime")
        self.validator = validator or ActionValidator(logger=self.logger)
        self.verifier = verifier or Verifier(
            max_recovery_per_step=settings.max_recovery_attempts_per_step,
            max_total_recovery=settings.max_total_recoveries_per_run,
            logger=self.logger,
        )

        self._worker_task: Optional[asyncio.Task] = None
        self._current_executing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Active run tracking and cancellation tokens
        self._active_run_token: Optional[CancellationToken] = None
        self._run_tokens: dict[str, CancellationToken] = {}
        self._run_lock = asyncio.Lock()

        # Task contexts per run (bounded context tracking)
        self._task_contexts: dict[str, TaskContext] = {}

        # Subscribe default internal handlers
        self._setup_event_subscriptions()

    @property
    def active_run_id(self) -> Optional[str]:
        """Return the ID of the currently active run, if any."""
        return self._active_run_token.run_id if self._active_run_token else None

    def get_or_create_task_context(self, run_id: Optional[str] = None) -> TaskContext:
        """Get or create the TaskContext for a specific run ID."""
        target_id = run_id or (self.active_run_id if self._active_run_token else "default_run")
        if target_id not in self._task_contexts:
            self._task_contexts[target_id] = TaskContext(run_id=target_id)
        return self._task_contexts[target_id]

    async def get_or_create_run_token(self) -> CancellationToken:
        """Get the current run token or create a new one."""
        async with self._run_lock:
            if self._active_run_token is None or self._active_run_token.is_cancelled:
                self._active_run_token = CancellationToken(logger=self.logger)
                self._run_tokens[self._active_run_token.run_id] = self._active_run_token
                self.state_manager.set_current_run_id(self._active_run_token.run_id)
                self.get_or_create_task_context(self._active_run_token.run_id)
            return self._active_run_token

    def is_run_cancelled(self, run_id: Optional[str]) -> bool:
        """Check if a specific run_id was cancelled."""
        if not run_id:
            return False
        token = self._run_tokens.get(run_id)
        return token.is_cancelled if token else False

    def _setup_event_subscriptions(self) -> None:
        """Register default runtime event handlers on the event bus."""
        self.event_bus.subscribe(EventType.INTENT_DETECTED, self._handle_intent_detected)
        self.event_bus.subscribe(EventType.USER_INTERRUPTED, self._handle_user_interrupted)

    async def start(self) -> None:
        """Start the agent runtime worker loop."""
        if self._worker_task and not self._worker_task.done():
            self.logger.warning("Runtime is already running.")
            return

        self.logger.info("AGENT STARTED: Starting runtime loop...")
        self.state_manager.set_lifecycle(AgentLifecycleState.RUNNING)
        self._stop_event.clear()

        self._worker_task = asyncio.create_task(self._process_queue())
        await self.event_bus.publish(Event(EventType.SYSTEM_STARTED))

    async def stop(self) -> None:
        """Gracefully shut down the agent runtime."""
        self.logger.info("SHUTDOWN REQUESTED: Stopping runtime...")
        self.state_manager.set_lifecycle(AgentLifecycleState.STOPPING)
        await self.event_bus.publish(Event(EventType.SYSTEM_STOPPING))

        self._stop_event.set()

        if self._current_executing_task and not self._current_executing_task.done():
            self._current_executing_task.cancel()

        cancelled_actions = await self.action_queue.clear_all()
        for action in cancelled_actions:
            self.logger.info("ACTION CANCELLED: Pending action %s cancelled during shutdown", action.action_id)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id}))

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self.state_manager.set_current_action(None)
        self.state_manager.set_pending_count(0)
        self.state_manager.set_lifecycle(AgentLifecycleState.STOPPED)
        self.logger.info("AGENT STOPPED: Graceful shutdown complete.")
        await self.event_bus.publish(Event(EventType.SYSTEM_STOPPED))

    async def submit_action(self, action: Action) -> None:
        """Validate and add an action directly to the queue."""
        if action.run_id is None:
            token = await self.get_or_create_run_token()
            action.run_id = token.run_id

        # Reject stale action if its run has already been cancelled
        if self.is_run_cancelled(action.run_id):
            self.logger.info("STALE ACTION REJECTED: Action %s belongs to cancelled run %s", action.action_id, action.run_id)
            action.mark_cancelled()
            return

        # Schema validation (Phase 7)
        is_valid, error_reason = self.validator.validate(action)
        if not is_valid:
            self.logger.warning("ACTION VALIDATION FAILED: %s (Action: %s)", error_reason, action.action_type)
            action.mark_failed(error_reason or "Validation failed")
            await self.event_bus.publish(
                Event(EventType.ACTION_FAILED, {"action_id": action.action_id, "run_id": action.run_id, "error": error_reason})
            )
            return

        self.logger.info("ACTION QUEUED: %s (ID: %s, Run: %s)", action.action_type, action.action_id, action.run_id)
        await self.action_queue.enqueue(action)
        self.state_manager.set_pending_count(self.action_queue.pending_count)
        await self.event_bus.publish(
            Event(EventType.ACTION_QUEUED, {"action_id": action.action_id, "action_type": action.action_type, "run_id": action.run_id})
        )

    async def interrupt(self, reason: str = "User requested cancellation") -> None:
        """
        Trigger an immediate interruption using unified cancellation.
        Discards queued actions and cancels active execution/recovery.
        """
        self.logger.info("USER INTERRUPTED: %s", reason)
        self.state_manager.request_cancellation()

        if self._active_run_token and not self._active_run_token.is_cancelled:
            await self._active_run_token.cancel(reason=reason)

        cancelled_actions = await self.action_queue.clear_all()
        for action in cancelled_actions:
            self.logger.info("ACTION CANCELLED: Pending action %s cancelled", action.action_id)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id, "run_id": action.run_id}))

        if self._current_executing_task and not self._current_executing_task.done():
            self._current_executing_task.cancel()

        self.state_manager.set_pending_count(0)
        self.state_manager.set_current_run_id(None)
        await self.event_bus.publish(Event(EventType.USER_INTERRUPTED, {"reason": reason}))

    async def request_clarification(self, question: str, run_id: Optional[str] = None) -> None:
        """Ask a concise clarification to the user and return control to listening mode."""
        self.logger.info("CLARIFICATION REQUESTED: %s", question)
        target_run_id = run_id or self.active_run_id
        await self.event_bus.publish(
            Event(EventType.CLARIFICATION_REQUESTED, {"question": question, "run_id": target_run_id})
        )

    async def start_new_command(self, intent_type: str, arguments: Optional[dict] = None) -> CancellationToken:
        """Interrupts any active run and starts a fresh new run with the given command."""
        if self._active_run_token and not self._active_run_token.is_cancelled:
            await self.interrupt(reason=f"Superseded by new command: {intent_type}")
        token = await self.get_or_create_run_token()
        action = Action(action_type=intent_type, arguments=arguments or {}, run_id=token.run_id)
        await self.submit_action(action)
        return token

    async def _handle_intent_detected(self, event: Event) -> None:
        """Convert an INTENT_DETECTED event into a queued Action."""
        intent_type = event.payload.get("intent") or event.payload.get("action_type") or "unknown_action"
        arguments = event.payload.get("arguments", {})
        priority = event.payload.get("priority", 0)

        # Explicit barge-in interruption intent
        if intent_type == "INTERRUPT":
            await self.interrupt(reason="User barge-in: stop/cancel intent detected")
            return

        # Explicit clarification request
        if intent_type == "CLARIFICATION":
            question = arguments.get("question", "Could you please clarify your request?")
            await self.request_clarification(question, run_id=event.payload.get("run_id"))
            return

        # Check if the event came from an already cancelled run
        incoming_run_id = event.payload.get("run_id")
        if incoming_run_id and self.is_run_cancelled(incoming_run_id):
            self.logger.info("STALE INTENT REJECTED: Intent %s belongs to cancelled run %s", intent_type, incoming_run_id)
            return

        # If incoming intent is from a new command while previous run is active and executing:
        if event.payload.get("new_command", False) and self._active_run_token and not self._active_run_token.is_cancelled:
            await self.interrupt(reason=f"Interrupted by new command: {intent_type}")

        # If run is currently in cancellation cooldown or explicitly cancelled without a fresh run started:
        if self._active_run_token and self._active_run_token.is_cancelled and not incoming_run_id:
            if not event.payload.get("new_command", False):
                self.logger.info("INTENT REJECTED: Runtime is currently stopped/cancelled for run %s", self._active_run_token.run_id)
                return
            self._active_run_token = None

        token = await self.get_or_create_run_token()
        context = self.get_or_create_task_context(token.run_id)
        if not context.original_intent:
            context.original_intent = intent_type
        context.current_objective = intent_type

        action = Action(action_type=intent_type, arguments=arguments, priority=priority, run_id=token.run_id)
        self.logger.info("EVENT RECEIVED: Intent detected -> Created action %s (Run: %s)", action.action_type, action.run_id)
        await self.event_bus.publish(Event(EventType.ACTION_CREATED, {"action_id": action.action_id, "run_id": action.run_id}))
        await self.submit_action(action)

    async def _handle_user_interrupted(self, event: Event) -> None:
        """Handle USER_INTERRUPTED event by ensuring clean state."""
        self.state_manager.clear_cancellation()

    async def _process_queue(self) -> None:
        """Continuous worker loop processing actions sequentially from the queue."""
        while not self._stop_event.is_set():
            try:
                try:
                    action = await asyncio.wait_for(self.action_queue.get_next(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if self._stop_event.is_set():
                    break

                await self._execute_action(action)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("AGENT_ERROR: Unexpected error in queue worker: %s", str(e), exc_info=True)
                self.state_manager.record_error(str(e))
                await self.event_bus.publish(Event(EventType.AGENT_ERROR, {"error": str(e)}))

    async def _execute_action(self, action: Action) -> None:
        """Executes a single action with lifecycle events, verification, and recovery."""
        # Stale run check
        if self.is_run_cancelled(action.run_id):
            action.mark_cancelled()
            self.logger.info("STALE ACTION DISCARDED: Action %s skipped because run %s was cancelled", action.action_id, action.run_id)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id, "run_id": action.run_id}))
            return

        token = self._run_tokens.get(action.run_id) or self._active_run_token
        if token and token.is_cancelled:
            action.mark_cancelled()
            self.logger.info("ACTION CANCELLED: Action %s skipped because token is cancelled", action.action_id)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id, "run_id": action.run_id}))
            return

        context = self.get_or_create_task_context(action.run_id)

        # Context-aware deduplication check: avoid repeating already completed actions in the same run
        if context.is_action_already_completed(action.action_type, action.arguments):
            self.logger.info("ACTION SKIPPED: %s already completed in run %s", action.action_type, action.run_id)
            action.mark_completed({"skipped": True, "reason": "Already completed in this run"})
            result_record = ActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                outcome=ExecutionOutcome.SKIPPED,
                action_arguments=action.arguments,
                run_id=action.run_id,
                message="Skipped: already completed in this run",
            )
            context.record_completed_action(result_record)
            await self.event_bus.publish(
                Event(EventType.ACTION_COMPLETED, {"action_id": action.action_id, "run_id": action.run_id, "skipped": True})
            )
            return

        action.mark_running()
        self.state_manager.set_current_action(action)
        self.state_manager.set_pending_count(self.action_queue.pending_count)

        self.logger.info("ACTION STARTED: %s (ID: %s, Run: %s)", action.action_type, action.action_id, action.run_id)
        await self.event_bus.publish(Event(EventType.ACTION_STARTED, {"action_id": action.action_id, "run_id": action.run_id}))

        # Capture pre-action observation
        obs_before = Observation(observation_type="system_state", data={"action": action.action_type}, run_id=action.run_id)
        context.record_observation(obs_before)

        try:
            # Execute action with cooperative cancellation token
            self._current_executing_task = asyncio.create_task(self.executor.execute(action, token=token))
            raw_result = await self._current_executing_task
            self._current_executing_task = None

            # Post-action verification for meaningful actions
            if settings.enable_post_action_verification and self.verifier.should_verify(action):
                await self.event_bus.publish(
                    Event(EventType.VERIFICATION_STARTED, {"action_id": action.action_id, "run_id": action.run_id})
                )
                is_verified, v_message = await self.verifier.verify_action(action, raw_result, token=token)

                if is_verified:
                    await self.event_bus.publish(
                        Event(EventType.VERIFICATION_SUCCEEDED, {"action_id": action.action_id, "run_id": action.run_id, "message": v_message})
                    )
                    action.mark_completed(raw_result)
                    result_record = ActionResult(
                        action_id=action.action_id,
                        action_type=action.action_type,
                        outcome=ExecutionOutcome.VERIFIED,
                        action_arguments=action.arguments,
                        run_id=action.run_id,
                        observation_before=obs_before,
                        message=v_message,
                    )
                    context.record_completed_action(result_record)
                    self.logger.info("ACTION COMPLETED & VERIFIED: %s", action.action_type)
                    await self.event_bus.publish(
                        Event(EventType.ACTION_COMPLETED, {"action_id": action.action_id, "run_id": action.run_id, "result": raw_result, "verified": True})
                    )
                else:
                    await self.event_bus.publish(
                        Event(EventType.VERIFICATION_FAILED, {"action_id": action.action_id, "run_id": action.run_id, "reason": v_message})
                    )
                    # Attempt bounded recovery
                    recovered = await self._attempt_recovery(action, context, token, initial_error=v_message, obs_before=obs_before)
                    if not recovered:
                        if token and token.is_cancelled:
                            action.mark_cancelled()
                            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id, "run_id": action.run_id}))
                        else:
                            action.mark_failed(f"Verification failed: {v_message}")
                            result_record = ActionResult(
                                action_id=action.action_id,
                                action_type=action.action_type,
                                outcome=ExecutionOutcome.VERIFICATION_FAILED,
                                action_arguments=action.arguments,
                                run_id=action.run_id,
                                observation_before=obs_before,
                                error=v_message,
                            )
                            context.record_completed_action(result_record)
                            await self.event_bus.publish(
                                Event(EventType.ACTION_FAILED, {"action_id": action.action_id, "run_id": action.run_id, "error": v_message})
                            )
            else:
                action.mark_completed(raw_result)
                result_record = ActionResult(
                    action_id=action.action_id,
                    action_type=action.action_type,
                    outcome=ExecutionOutcome.COMPLETED,
                    action_arguments=action.arguments,
                    run_id=action.run_id,
                    observation_before=obs_before,
                )
                context.record_completed_action(result_record)
                self.logger.info("ACTION COMPLETED: %s (Result: %s)", action.action_type, raw_result)
                await self.event_bus.publish(
                    Event(EventType.ACTION_COMPLETED, {"action_id": action.action_id, "run_id": action.run_id, "result": raw_result})
                )

        except asyncio.CancelledError:
            action.mark_cancelled()
            self.logger.info("ACTION CANCELLED: %s was cancelled during execution", action.action_type)
            result_record = ActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                outcome=ExecutionOutcome.CANCELLED,
                action_arguments=action.arguments,
                run_id=action.run_id,
                observation_before=obs_before,
            )
            context.record_completed_action(result_record)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id, "run_id": action.run_id}))
        except Exception as exc:
            exec_error = str(exc)
            self.logger.warning("ACTION EXECUTION ERROR: %s: %s", action.action_type, exec_error)
            recovered = await self._attempt_recovery(action, context, token, initial_error=exec_error, obs_before=obs_before)
            if not recovered:
                action.mark_failed(exec_error)
                self.state_manager.record_error(exec_error)
                result_record = ActionResult(
                    action_id=action.action_id,
                    action_type=action.action_type,
                    outcome=ExecutionOutcome.FAILED,
                    action_arguments=action.arguments,
                    run_id=action.run_id,
                    observation_before=obs_before,
                    error=exec_error,
                )
                context.record_completed_action(result_record)
                await self.event_bus.publish(
                    Event(EventType.ACTION_FAILED, {"action_id": action.action_id, "run_id": action.run_id, "error": exec_error})
                )
        finally:
            self._current_executing_task = None
            self.state_manager.set_current_action(None)
            self.state_manager.set_pending_count(self.action_queue.pending_count)

    async def _attempt_recovery(
        self,
        action: Action,
        context: TaskContext,
        token: Optional[CancellationToken],
        initial_error: Optional[str] = None,
        obs_before: Optional[Observation] = None,
    ) -> bool:
        """
        Attempts bounded recovery loop when an action fails or verification fails.
        Returns True if recovered successfully, False otherwise.
        """
        attempts = 0
        while self.verifier.can_recover(action, context, attempts):
            if token and token.is_cancelled:
                self.logger.info("RECOVERY CANCELLED: Token cancelled during recovery")
                return False

            if self.is_run_cancelled(action.run_id):
                self.logger.info("RECOVERY CANCELLED: Run %s cancelled during recovery", action.run_id)
                return False

            attempts += 1
            context.recovery_count += 1
            self.logger.info(
                "RECOVERY ATTEMPT %d: Attempting recovery for %s (Run: %s)",
                attempts,
                action.action_type,
                action.run_id,
            )
            await self.event_bus.publish(
                Event(EventType.RECOVERY_STARTED, {"action_id": action.action_id, "attempt": attempts, "run_id": action.run_id})
            )

            recovery_action = self.verifier.create_recovery_action(action, attempt_num=attempts)

            try:
                self._current_executing_task = asyncio.create_task(self.executor.execute(recovery_action, token=token))
                rec_res = await self._current_executing_task
            except asyncio.CancelledError:
                self.logger.info("RECOVERY CANCELLED: Execution cancelled during recovery attempt")
                return False
            except Exception as rec_err:
                self.logger.warning("Recovery execution attempt %d failed: %s", attempts, rec_err)
                continue
            finally:
                self._current_executing_task = None

            if token and token.is_cancelled:
                return False

            try:
                rec_verified, rec_reason = await self.verifier.verify_action(recovery_action, rec_res, token=token)
            except asyncio.CancelledError:
                return False

            if rec_verified:
                self.logger.info("RECOVERY SUCCEEDED on attempt %d for action %s", attempts, action.action_type)
                await self.event_bus.publish(
                    Event(EventType.RECOVERY_COMPLETED, {"action_id": action.action_id, "attempt": attempts, "run_id": action.run_id})
                )
                action.mark_completed(rec_res)
                result_record = ActionResult(
                    action_id=action.action_id,
                    action_type=action.action_type,
                    outcome=ExecutionOutcome.RECOVERED,
                    action_arguments=action.arguments,
                    run_id=action.run_id,
                    observation_before=obs_before,
                    recovery_attempts=attempts,
                    message=f"Recovered on attempt {attempts}",
                )
                context.record_completed_action(result_record)
                await self.event_bus.publish(
                    Event(EventType.ACTION_COMPLETED, {"action_id": action.action_id, "run_id": action.run_id, "result": rec_res, "recovered": True})
                )
                return True
            else:
                self.logger.warning("Recovery verification attempt %d failed: %s", attempts, rec_reason)

        self.logger.warning("RECOVERY EXHAUSTED: Could not recover %s after %d attempts", action.action_type, attempts)
        await self.event_bus.publish(
            Event(EventType.RECOVERY_FAILED, {"action_id": action.action_id, "attempts": attempts, "run_id": action.run_id})
        )
        return False
