# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Customer Support Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CustomerSupportAction, CustomerSupportObservation
except ImportError:
    from models import CustomerSupportAction, CustomerSupportObservation

from ..task_env.tasks import easy_task, medium_task, hard_task
MAX_STEPS = 10
class CustomerSupportEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = CustomerSupportEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Customer Support Env environment ready!"
        >>>
        >>> obs = env.step(CustomerSupportAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    VALID_ACTIONS = {
    "apologize",
    "provide_status_update",
    "refund",
    "offer_refund",
    "give_discount",
    "escalate_to_human",
    "ask_info",
    }

    def __init__(self):
        """Initialize the customer_support_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._current_obs = None
        self._current_task = None
        self._actions_taken = []

    def reset(self, task="easy") -> CustomerSupportObservation:
        """
        Reset the environment.

        Returns:
            CustomerSupportObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.action_history = []

        if task == "easy":
            self._current_task = easy_task
        elif task == "medium":
            self._current_task = medium_task
        elif task == "hard":
            self._current_task = hard_task
        else:
            raise ValueError(f"Unknown task: {task}. Choose easy, medium or hard.")
        self._current_obs = CustomerSupportObservation(
            **self._current_task.initial_state,
            reward=0.0,
            done=False
        )
            
        return self._current_obs

    def step(self, action):  # type: ignore[override]

        if self._current_task is None or self._current_obs is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self._state.step_count += 1

        action_name = action if isinstance(action, str) else action.action

        prev_obs = self._current_obs

        # -----------------------------
        # 1️⃣ ACTION VALIDATION
        # -----------------------------
        if action_name not in self.VALID_ACTIONS:
            reward = -0.2
            done = False

            return self._current_obs, reward, done, {
                "error": f"Invalid action: {action_name}"
            }

        # -----------------------------
        # 2️⃣ TRACK ACTION HISTORY
        # -----------------------------
        last_action = self.action_history[-1] if self.action_history else None
        self.action_history.append(action_name)

        repeated_action = last_action == action_name

        # -----------------------------
        # 3️⃣ BASE TASK SCORE
        # -----------------------------
        base_score = self._current_task.evaluate(self.action_history)

        reward = 0.0  # IMPORTANT: start fresh

        # -----------------------------
        # 4️⃣ ACTION-LEVEL REWARD
        # -----------------------------
        if action_name in {"apologize", "provide_status_update", "ask_info"}:
            reward += 0.3

        elif action_name in {"refund", "offer_refund"}:
            if prev_obs.order_status != "refunded":
                reward += 0.5
            else:
                reward -= 0.2  # unnecessary refund

        elif action_name == "give_discount":
            reward += 0.3

        elif action_name == "escalate_to_human":
            reward += 0.2

        # -----------------------------
        # 5️⃣ EDGE CASES
        # -----------------------------
        # 🔁 Repeated action
        if repeated_action:
            reward -= 0.2

       

        # ⚔️ Conflicting actions (acting after resolution)
        if prev_obs.order_status == "refunded" and action_name != "close_ticket":
            reward -= 0.2

        # -----------------------------
        # 6️⃣ STEP PENALTY
        # -----------------------------
        reward -= 0.05

        # -----------------------------
        # 7️⃣ FINAL BONUS
        # -----------------------------
        if base_score == 1.0:
            reward += 1.0

        # -----------------------------
        # 8️⃣ STATE UPDATE
        # -----------------------------
        updated_sentiment = prev_obs.sentiment
        updated_order_status = prev_obs.order_status

        if action_name == "apologize":
            if updated_sentiment == "angry":
                updated_sentiment = "frustrated"
            elif updated_sentiment == "frustrated":
                updated_sentiment = "calmer"
            else:
                updated_sentiment = "neutral"

        elif action_name == "provide_status_update":
            updated_order_status = "update_provided"
            if updated_sentiment == "frustrated":
                updated_sentiment = "calmer"

        elif action_name in {"refund", "offer_refund"}:
            updated_order_status = "refunded"

        elif action_name == "escalate_to_human":
            updated_order_status = "escalated"

        elif action_name == "give_discount":
            updated_order_status = "discount_applied"

        # -----------------------------
        # 9️⃣ DONE CONDITIONS
        # -----------------------------
        task_completed = base_score == 1.0
        max_steps_reached = self._state.step_count >= MAX_STEPS

        recent_actions = self.action_history[-3:]
        repeated_wrong = (
            len(recent_actions) == 3
            and len(set(recent_actions)) == 1
            and base_score == 0.0
        )

        done = task_completed or max_steps_reached or repeated_wrong

        termination_reason = (
            "task_completed"
            if task_completed
            else "max_steps_reached"
            if max_steps_reached
            else "repeated_wrong_actions"
            if repeated_wrong
            else "in_progress"
        )

        # -----------------------------
        # 🔟 UPDATE OBS
        # -----------------------------
        self._current_obs = CustomerSupportObservation(
            user_query=prev_obs.user_query,
            sentiment=updated_sentiment,
            issue_type=prev_obs.issue_type,
            order_status=updated_order_status,
            attempts=prev_obs.attempts + 1,
            reward=reward,
            done=done,
        )
         # 🌀 No progress (same sentiment + status)
        if (
            prev_obs.sentiment == updated_sentiment
            and prev_obs.order_status == updated_order_status
        ):
            reward -= 0.1
        info = {
            "task_goal": self._current_task.goal,
            "actions_taken": list(self.action_history),
            "task_score": base_score,
            "repeated_action": repeated_action,
            "termination_reason": termination_reason,
            "steps_taken": self._state.step_count,
        }

        return self._current_obs, reward, done, info
    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
