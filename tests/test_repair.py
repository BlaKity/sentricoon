from __future__ import annotations

import json
import unittest

from sentricoon.diagnoser import Diagnosis
from sentricoon.llm import LLMRouter, MockBackend, RouterConfig
from sentricoon.repair import RepairDecision, Repairer
from sentricoon.state import Step


def _diag(fix_strategy: str = "retry", **kw) -> Diagnosis:
    return Diagnosis(
        failure_type=kw.get("failure_type", "tool_error"),
        root_cause=kw.get("root_cause", "thing broke"),
        fix_strategy=fix_strategy,
        ruled_out=kw.get("ruled_out"),
        evidence=kw.get("evidence"),
    )


def _step() -> Step:
    return Step(
        action="file.write",
        args={"path": "/wrong/place.txt", "content": "x"},
        expected_state="file exists",
    )


def _router_with_local_response(content: str):
    local = MockBackend(script=[content], name="local", model="local-1")
    cloud = MockBackend(script=[], name="cloud", model="cloud-1")
    router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
    router.register(local)
    router.register(cloud)
    return router, local, cloud


# ============================================================================
# Strategy routing — no LLM involved
# ============================================================================


class TestRepairStrategyRouting(unittest.TestCase):
    def test_retry_strategy(self):
        d = Repairer().repair(_step(), _diag("retry"), attempts_so_far=0)
        self.assertEqual(d.strategy, "retry")
        self.assertIsNone(d.step)

    def test_replan_strategy(self):
        d = Repairer().repair(_step(), _diag("replan"), attempts_so_far=0)
        self.assertEqual(d.strategy, "replan")
        self.assertIsNone(d.step)

    def test_abort_strategy(self):
        d = Repairer().repair(_step(), _diag("abort"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")

    def test_unknown_strategy_falls_back_to_abort(self):
        # In practice Diagnoser coerces unknowns to "abort"; this guards the
        # case where a future Diagnosis is constructed programmatically.
        d = Repairer().repair(_step(), _diag("frobnicate"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("unknown fix_strategy", d.reason)


class TestRepairBudget(unittest.TestCase):
    def test_budget_exhausted_forces_abort_regardless_of_diagnosis(self):
        repairer = Repairer(max_attempts_per_step=3)
        d = repairer.repair(_step(), _diag("retry"), attempts_so_far=3)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("budget", d.reason)

    def test_budget_one_short_still_allows_retry(self):
        repairer = Repairer(max_attempts_per_step=3)
        d = repairer.repair(_step(), _diag("retry"), attempts_so_far=2)
        self.assertEqual(d.strategy, "retry")

    def test_custom_budget_respected(self):
        repairer = Repairer(max_attempts_per_step=1)
        d = repairer.repair(_step(), _diag("retry"), attempts_so_far=1)
        self.assertEqual(d.strategy, "abort")


# ============================================================================
# fix_input — uses LLMRole.INPUT_REPAIR (local backend)
# ============================================================================


class TestFixInput(unittest.TestCase):
    def test_fix_input_routes_to_local_backend(self):
        router, local, cloud = _router_with_local_response(json.dumps({
            "args": {"path": "/tmp/correct.txt", "content": "x"}
        }))
        Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        # CRITICAL: input_repair must go to LOCAL — the cost lever
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(cloud.calls), 0)
        self.assertEqual(router.call_counts.get("local"), 1)

    def test_fix_input_returns_rewritten_step(self):
        router, _, _ = _router_with_local_response(json.dumps({
            "args": {"path": "/tmp/correct.txt", "content": "y"}
        }))
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "fix_input")
        self.assertIsNotNone(d.step)
        self.assertEqual(d.step.args["path"], "/tmp/correct.txt")
        self.assertEqual(d.step.args["content"], "y")
        # action and expected_state preserved
        self.assertEqual(d.step.action, "file.write")
        self.assertEqual(d.step.expected_state, "file exists")

    def test_fix_input_preserves_risk_level(self):
        router, _, _ = _router_with_local_response(json.dumps({"args": {"path": "/x"}}))
        step = Step(action="file.delete", args={"path": "/wrong"},
                    expected_state="file gone", risk_level="high")
        d = Repairer(router).repair(step, _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.step.risk_level, "high")

    def test_fix_input_without_router_aborts(self):
        d = Repairer(router=None).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("no LLM router", d.reason)

    def test_fix_input_non_json_aborts(self):
        router, _, _ = _router_with_local_response("not json")
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("non-JSON", d.reason)

    def test_fix_input_non_object_aborts(self):
        router, _, _ = _router_with_local_response(json.dumps(["a"]))
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("non-object", d.reason)

    def test_fix_input_missing_args_key_aborts(self):
        router, _, _ = _router_with_local_response(json.dumps({"foo": "bar"}))
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("'args'", d.reason)

    def test_fix_input_args_not_object_aborts(self):
        router, _, _ = _router_with_local_response(json.dumps({"args": "wrong"}))
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")

    def test_fix_input_refuses_action_change(self):
        """fix_input must NOT change the action — that's replan's job."""
        router, _, _ = _router_with_local_response(json.dumps({
            "action": "shell.run",  # trying to change action
            "args": {"argv": ["something"]},
        }))
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("change action", d.reason)

    def test_fix_input_backend_unreachable_aborts(self):
        local = MockBackend(script=[], name="local", model="local-1")  # exhausted
        cloud = MockBackend(script=[], name="cloud", model="cloud-1")
        router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        router.register(local)
        router.register(cloud)
        d = Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        self.assertEqual(d.strategy, "abort")
        self.assertIn("unreachable", d.reason)


class TestRepairBudgetVsFixInput(unittest.TestCase):
    """Budget exhaustion must short-circuit BEFORE the LLM call is made."""

    def test_budget_short_circuits_fix_input(self):
        router, local, _ = _router_with_local_response(json.dumps({"args": {"path": "/x"}}))
        Repairer(router, max_attempts_per_step=3).repair(
            _step(), _diag("fix_input"), attempts_so_far=3,
        )
        # No LLM call should have been made
        self.assertEqual(len(local.calls), 0)


class TestRepairTaskAwareness(unittest.TestCase):
    """fix_input must see the user's original task so it can respect
    constraints like 'do not use sudo'. Regression test for the
    2026-05-18 sudo escalation."""

    def test_task_passed_into_input_repair_prompt(self):
        router, local, _ = _router_with_local_response(json.dumps({
            "args": {"argv": ["find", "/tmp"]},  # no sudo — respecting constraint
        }))
        Repairer(router).repair(
            _step(), _diag("fix_input"),
            attempts_so_far=0,
            task="Find files in /tmp. Do not use sudo.",
        )
        user_msg = local.calls[0].messages[-1].content
        self.assertIn("Original user task", user_msg)
        self.assertIn("Do not use sudo", user_msg)
        self.assertIn("HARD constraints", user_msg)

    def test_task_omitted_when_none(self):
        router, local, _ = _router_with_local_response(json.dumps({
            "args": {"path": "/x"},
        }))
        Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        user_msg = local.calls[0].messages[-1].content
        self.assertNotIn("Original user task", user_msg)

    def test_system_prompt_includes_constraint_handling(self):
        router, local, _ = _router_with_local_response(json.dumps({
            "args": {"path": "/x"},
        }))
        Repairer(router).repair(_step(), _diag("fix_input"), attempts_so_far=0)
        sys_msg = local.calls[0].messages[0].content
        self.assertIn("RESPECT USER CONSTRAINTS", sys_msg)


if __name__ == "__main__":
    unittest.main()
