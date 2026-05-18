from __future__ import annotations

import json
import unittest

from sentricoon.diagnoser import Diagnoser, Diagnosis
from sentricoon.llm import LLMRouter, MockBackend, RouterConfig
from sentricoon.state import Step
from sentricoon.tools.base import ToolResult
from sentricoon.verifier import Verification


def _router(content: str):
    cloud = MockBackend(script=[content], name="cloud", model="cloud-1")
    local = MockBackend(script=[], name="local", model="local-1")
    router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
    router.register(cloud)
    router.register(local)
    return router, cloud, local


def _step():
    return Step(
        action="shell.run",
        args={"argv": ["net", "start", "Audiosrv"]},
        expected_state="service 'Audiosrv' is RUNNING",
        risk_level="high",
    )


def _result():
    return ToolResult(success=True, extra={"returncode": 2}, error=None,
                      output={"stdout": "", "stderr": "service does not exist"})


def _verif():
    return Verification(success=False, reason="non-zero exit code: 2", layer="hard")


# ============================================================================
# Routing + system prompt
# ============================================================================


class TestDiagnoserRouting(unittest.TestCase):
    def test_routes_to_cloud_diagnoser(self):
        router, cloud, local = _router(json.dumps({
            "failure_type": "tool_error", "root_cause": "service missing",
            "fix_strategy": "replan",
        }))
        Diagnoser(router).diagnose(_step(), _result(), _verif())
        # DIAGNOSER routes to cloud per default config
        self.assertEqual(len(cloud.calls), 1)
        self.assertEqual(len(local.calls), 0)
        self.assertEqual(router.call_counts["cloud"], 1)

    def test_system_prompt_disavows_planning(self):
        router, cloud, _ = _router(json.dumps({
            "failure_type": "tool_error", "root_cause": "x", "fix_strategy": "abort"
        }))
        Diagnoser(router).diagnose(_step(), _result(), _verif())
        sys_msg = cloud.calls[0].messages[0]
        self.assertEqual(sys_msg.role, "system")
        self.assertIn("did NOT plan", sys_msg.content)

    def test_user_prompt_includes_verification(self):
        router, cloud, _ = _router(json.dumps({
            "failure_type": "tool_error", "root_cause": "x", "fix_strategy": "abort"
        }))
        Diagnoser(router).diagnose(_step(), _result(), _verif())
        user_msg = cloud.calls[0].messages[-1]
        self.assertIn("non-zero exit code", user_msg.content)  # verification reason
        self.assertIn("service does not exist", user_msg.content)  # tool output
        self.assertIn("Audiosrv", user_msg.content)  # step args


# ============================================================================
# Parsing + validation
# ============================================================================


class TestDiagnoserParsing(unittest.TestCase):
    def test_valid_response_parsed(self):
        router, _, _ = _router(json.dumps({
            "failure_type": "tool_error",
            "root_cause": "service does not exist on this system",
            "fix_strategy": "replan",
            "ruled_out": "transient network",
            "evidence": "stderr says 'service does not exist'",
        }))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertIsInstance(d, Diagnosis)
        self.assertEqual(d.failure_type, "tool_error")
        self.assertEqual(d.fix_strategy, "replan")
        self.assertEqual(d.ruled_out, "transient network")
        self.assertEqual(d.evidence, "stderr says 'service does not exist'")

    def test_unknown_failure_type_coerced(self):
        router, _, _ = _router(json.dumps({
            "failure_type": "made_up_type",
            "root_cause": "x",
            "fix_strategy": "retry",
        }))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertEqual(d.failure_type, "unknown")

    def test_unknown_fix_strategy_coerced_to_abort(self):
        router, _, _ = _router(json.dumps({
            "failure_type": "tool_error",
            "root_cause": "x",
            "fix_strategy": "delete_universe",
        }))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertEqual(d.fix_strategy, "abort")

    def test_missing_root_cause_gets_placeholder(self):
        router, _, _ = _router(json.dumps({
            "failure_type": "unknown",
            "fix_strategy": "abort",
        }))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertIn("no root cause", d.root_cause)

    def test_empty_string_ruled_out_becomes_none(self):
        router, _, _ = _router(json.dumps({
            "failure_type": "unknown", "root_cause": "x",
            "fix_strategy": "abort", "ruled_out": "",
        }))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertIsNone(d.ruled_out)

    def test_null_ruled_out_becomes_none(self):
        router, _, _ = _router(json.dumps({
            "failure_type": "unknown", "root_cause": "x",
            "fix_strategy": "abort", "ruled_out": None,
        }))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertIsNone(d.ruled_out)


# ============================================================================
# Error paths
# ============================================================================


class TestDiagnoserErrorPaths(unittest.TestCase):
    def test_non_json_response_aborts(self):
        router, _, _ = _router("definitely not json")
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertEqual(d.failure_type, "unknown")
        self.assertEqual(d.fix_strategy, "abort")
        self.assertIn("non-JSON", d.root_cause)

    def test_non_object_json_aborts(self):
        router, _, _ = _router(json.dumps(["a", "b"]))
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertEqual(d.fix_strategy, "abort")
        self.assertIn("non-object", d.root_cause)

    def test_backend_unreachable_aborts(self):
        backend = MockBackend(script=[], name="cloud", model="cloud-1")
        local = MockBackend(script=[], name="local", model="local-1")
        router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        router.register(backend)
        router.register(local)
        d = Diagnoser(router).diagnose(_step(), _result(), _verif())
        self.assertEqual(d.fix_strategy, "abort")
        self.assertIn("unreachable", d.root_cause)


class TestDiagnosisDict(unittest.TestCase):
    def test_to_dict_round_trip(self):
        d = Diagnosis(
            failure_type="bad_input", root_cause="wrong path",
            fix_strategy="fix_input", ruled_out="env_changed",
            evidence="errno 2",
        )
        as_dict = d.to_dict()
        self.assertEqual(as_dict["failure_type"], "bad_input")
        self.assertEqual(as_dict["fix_strategy"], "fix_input")
        # round-trips through json
        self.assertEqual(json.loads(json.dumps(as_dict)), as_dict)


class TestDiagnoserTaskAwareness(unittest.TestCase):
    """The diagnoser must see the original task so it can respect constraints
    like 'do not use sudo'. Regression test for the 2026-05-18 sudo escalation."""

    def test_task_passed_through_to_prompt(self):
        router, cloud, _ = _router(json.dumps({
            "failure_type": "tool_error", "root_cause": "x", "fix_strategy": "abort",
        }))
        Diagnoser(router).diagnose(
            _step(), _result(), _verif(),
            task="Find files in /tmp larger than 50MB. Do not use sudo.",
        )
        user_msg = cloud.calls[0].messages[-1].content
        self.assertIn("Original user task", user_msg)
        self.assertIn("Do not use sudo", user_msg)

    def test_task_omitted_when_none(self):
        router, cloud, _ = _router(json.dumps({
            "failure_type": "tool_error", "root_cause": "x", "fix_strategy": "abort",
        }))
        Diagnoser(router).diagnose(_step(), _result(), _verif())  # no task=
        user_msg = cloud.calls[0].messages[-1].content
        # No phantom task block
        self.assertNotIn("Original user task", user_msg)

    def test_system_prompt_includes_constraint_handling(self):
        router, cloud, _ = _router(json.dumps({
            "failure_type": "tool_error", "root_cause": "x", "fix_strategy": "abort",
        }))
        Diagnoser(router).diagnose(_step(), _result(), _verif())
        sys_msg = cloud.calls[0].messages[0].content
        self.assertIn("USER CONSTRAINT", sys_msg)
        self.assertIn("do not use sudo", sys_msg.lower())


if __name__ == "__main__":
    unittest.main()
