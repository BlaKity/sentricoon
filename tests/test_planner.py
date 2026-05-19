from __future__ import annotations

import json
import unittest

from unittest.mock import patch

from sentricoon.catalog import TOOL_CATALOG, catalog_for_prompt
from sentricoon.llm import LLMRouter, MockBackend, RouterConfig
from sentricoon.memory import Memory
from sentricoon.planner import (
    PlanError,
    Planner,
    _parse_and_validate,
    host_context_block,
    plan_risk_warnings,
    weak_predicate_warnings,
)
from sentricoon.safety.policies import ACTION_ALLOWLIST
from sentricoon.state import Step


def _router_with_plan(plan_json: str):
    cloud = MockBackend(script=[plan_json], name="cloud", model="cloud-1")
    local = MockBackend(script=[], name="local", model="local-1")
    router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
    router.register(cloud)
    router.register(local)
    return router, cloud, local


VALID_PLAN = json.dumps({
    "steps": [
        {
            "action": "system.info",
            "args": {},
            "expected_state": "system.info returns a 'platform' key",
            "risk_level": "low",
        },
        {
            "action": "process.list",
            "args": {},
            "expected_state": "process list contains at least one Audio service process",
            "risk_level": "low",
        },
    ]
})


# ============================================================================
# Catalog
# ============================================================================


class TestToolCatalog(unittest.TestCase):
    def test_every_catalog_entry_is_in_allowlist(self):
        for entry in TOOL_CATALOG:
            self.assertIn(entry["name"], ACTION_ALLOWLIST, msg=f"catalog has {entry['name']!r} but it's not in ACTION_ALLOWLIST")

    def test_every_allowlist_action_is_in_catalog(self):
        catalog_names = {entry["name"] for entry in TOOL_CATALOG}
        for action in ACTION_ALLOWLIST:
            self.assertIn(action, catalog_names, msg=f"{action!r} is in ACTION_ALLOWLIST but missing from catalog")

    def test_catalog_for_prompt_mentions_every_action(self):
        text = catalog_for_prompt()
        for entry in TOOL_CATALOG:
            self.assertIn(entry["name"], text)

    def test_catalog_filters_os_specific_actions(self):
        windows = catalog_for_prompt(system="Windows")
        self.assertNotIn("ubuntu.systemd.failed_units", windows)
        self.assertNotIn("systemctl", windows)

        linux = catalog_for_prompt(system="Linux")
        self.assertIn("ubuntu.systemd.failed_units", linux)


# ============================================================================
# Plan parsing & validation
# ============================================================================


class TestParseAndValidate(unittest.TestCase):
    def test_valid_plan_produces_steps(self):
        steps = _parse_and_validate(VALID_PLAN)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].action, "system.info")
        self.assertEqual(steps[0].expected_state, "system.info returns a 'platform' key")
        self.assertEqual(steps[0].risk_level, "low")

    def test_non_json_rejected(self):
        with self.assertRaisesRegex(PlanError, "non-JSON"):
            _parse_and_validate("not json")

    def test_non_object_rejected(self):
        with self.assertRaisesRegex(PlanError, "not a JSON object"):
            _parse_and_validate(json.dumps([{"action": "system.info"}]))

    def test_missing_steps_key_rejected(self):
        with self.assertRaisesRegex(PlanError, "missing required 'steps'"):
            _parse_and_validate(json.dumps({"plan": []}))

    def test_steps_not_list_rejected(self):
        with self.assertRaisesRegex(PlanError, "must be a list"):
            _parse_and_validate(json.dumps({"steps": "nope"}))

    def test_empty_plan_rejected(self):
        with self.assertRaisesRegex(PlanError, "empty"):
            _parse_and_validate(json.dumps({"steps": []}))

    def test_non_object_step_rejected(self):
        with self.assertRaisesRegex(PlanError, "not a JSON object"):
            _parse_and_validate(json.dumps({"steps": ["just a string"]}))

    def test_missing_action_rejected(self):
        with self.assertRaisesRegex(PlanError, "missing non-empty 'action'"):
            _parse_and_validate(json.dumps({"steps": [{"args": {}, "expected_state": "x"}]}))

    def test_action_not_in_allowlist_rejected(self):
        bad = json.dumps({"steps": [{"action": "rm.everything", "args": {}, "expected_state": "x"}]})
        with self.assertRaisesRegex(PlanError, "not in allowlist"):
            _parse_and_validate(bad)

    def test_args_not_object_rejected(self):
        bad = json.dumps({"steps": [{"action": "system.info", "args": "wrong", "expected_state": "x"}]})
        with self.assertRaisesRegex(PlanError, "must be an object"):
            _parse_and_validate(bad)

    def test_missing_expected_state_rejected(self):
        bad = json.dumps({"steps": [{"action": "system.info", "args": {}}]})
        with self.assertRaisesRegex(PlanError, "expected_state"):
            _parse_and_validate(bad)

    def test_empty_expected_state_rejected(self):
        bad = json.dumps({"steps": [{"action": "system.info", "args": {}, "expected_state": "   "}]})
        with self.assertRaisesRegex(PlanError, "expected_state"):
            _parse_and_validate(bad)

    def test_bad_risk_level_rejected(self):
        bad = json.dumps({"steps": [{"action": "system.info", "args": {}, "expected_state": "x", "risk_level": "extreme"}]})
        with self.assertRaisesRegex(PlanError, "risk_level"):
            _parse_and_validate(bad)

    def test_default_risk_level_is_low(self):
        plan = json.dumps({"steps": [{"action": "system.info", "args": {}, "expected_state": "platform present"}]})
        steps = _parse_and_validate(plan)
        self.assertEqual(steps[0].risk_level, "low")

    def test_default_args_is_empty_dict(self):
        plan = json.dumps({"steps": [{"action": "system.info", "expected_state": "x"}]})
        steps = _parse_and_validate(plan)
        self.assertEqual(steps[0].args, {})


# ============================================================================
# Planner orchestration
# ============================================================================


class TestPlannerCreatePlan(unittest.TestCase):
    def test_routes_to_cloud_backend(self):
        """Critical: planner MUST go to cloud, not local. The cost split."""
        router, cloud, local = _router_with_plan(VALID_PLAN)
        steps = Planner(router).create_plan("fix my speakers", goal_class="audio_no_sound")
        self.assertEqual(len(steps), 2)
        self.assertEqual(len(cloud.calls), 1)
        self.assertEqual(len(local.calls), 0)
        self.assertEqual(router.call_counts["cloud"], 1)

    def test_user_message_is_task(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("fix my speakers")
        user_msg = cloud.calls[0].messages[-1]
        self.assertEqual(user_msg.role, "user")
        self.assertEqual(user_msg.content, "fix my speakers")

    def test_goal_class_appears_in_system_prompt(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("fix my speakers", goal_class="audio_no_sound")
        sys_msg = cloud.calls[0].messages[0]
        self.assertEqual(sys_msg.role, "system")
        self.assertIn("audio_no_sound", sys_msg.content)

    def test_catalog_appears_in_system_prompt(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("x")
        sys_msg = cloud.calls[0].messages[0]
        for action in ACTION_ALLOWLIST:
            self.assertIn(action, sys_msg.content)

    def test_backend_error_wraps_as_plan_error(self):
        backend = MockBackend(script=[], name="cloud", model="cloud-1")  # exhausted → raises
        local = MockBackend(script=[], name="local", model="local-1")
        router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        router.register(backend)
        router.register(local)
        with self.assertRaisesRegex(PlanError, "unreachable"):
            Planner(router).create_plan("x")

    def test_invalid_plan_raises_plan_error(self):
        router, _, _ = _router_with_plan(json.dumps({"steps": [{"action": "rogue.action", "args": {}, "expected_state": "x"}]}))
        with self.assertRaisesRegex(PlanError, "not in allowlist"):
            Planner(router).create_plan("x")

    def test_uses_json_mode(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("x")
        self.assertTrue(cloud.calls[0].json_mode)


class TestPlannerQuarantineHook(unittest.TestCase):
    def test_no_quarantine_says_so_in_prompt(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router, memory=Memory()).create_plan("x")
        sys_msg = cloud.calls[0].messages[0]
        self.assertIn("No prior failed approaches", sys_msg.content)

    def test_quarantined_sequences_appear_in_prompt(self):
        class _MemoryWithQuarantine(Memory):
            def quarantined_action_sequences(self):
                return [
                    ["shell.run", "process.kill"],
                    ["file.delete"],
                ]
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router, memory=_MemoryWithQuarantine()).create_plan("x")
        sys_msg = cloud.calls[0].messages[0]
        self.assertIn("Avoid these failed approaches", sys_msg.content)
        self.assertIn("shell.run -> process.kill", sys_msg.content)
        self.assertIn("file.delete", sys_msg.content)


class TestPlannerReplan(unittest.TestCase):
    def test_replan_includes_failure_context(self):
        replan_json = json.dumps({"steps": [{"action": "system.info", "args": {}, "expected_state": "platform present", "risk_level": "low"}]})
        router, cloud, _ = _router_with_plan(replan_json)
        failed = Step(action="shell.run", args={"argv": ["bad"]}, expected_state="x")
        diagnosis = {"failure_type": "tool_error", "root_cause": "command not found"}
        Planner(router).replan("fix speakers", failed, diagnosis)
        user_msg = cloud.calls[0].messages[-1]
        self.assertIn("FAILED", user_msg.content)
        self.assertIn("shell.run", user_msg.content)
        self.assertIn("command not found", user_msg.content)

    def test_replan_routes_to_cloud(self):
        replan_json = json.dumps({"steps": [{"action": "system.info", "args": {}, "expected_state": "x", "risk_level": "low"}]})
        router, cloud, local = _router_with_plan(replan_json)
        failed = Step(action="shell.run", expected_state="x")
        Planner(router).replan("t", failed, {})
        self.assertEqual(len(cloud.calls), 1)
        self.assertEqual(len(local.calls), 0)


class TestHostContextBlock(unittest.TestCase):
    def test_linux_block_mentions_systemctl(self):
        block = host_context_block(system="Linux", platform_string="Linux-6.8", machine="x86_64")
        self.assertIn("Linux", block)
        self.assertIn("systemctl", block)
        self.assertIn("pactl", block)  # Linux audio hint
        self.assertNotIn("sc query", block)
        self.assertNotIn("launchctl", block)

    def test_windows_block_mentions_sc_query(self):
        block = host_context_block(system="Windows", platform_string="Windows-10", machine="AMD64")
        self.assertIn("Windows", block)
        self.assertIn("sc query", block)
        self.assertIn("Audiosrv", block)  # Windows audio service mentioned
        self.assertNotIn("systemctl", block)
        self.assertNotIn("launchctl", block)

    def test_darwin_block_mentions_launchctl(self):
        block = host_context_block(system="Darwin", platform_string="macOS-14", machine="arm64")
        self.assertIn("Darwin", block)
        self.assertIn("launchctl", block)
        self.assertNotIn("systemctl", block)
        self.assertNotIn("sc query", block)

    def test_unknown_os_falls_back_to_generic(self):
        block = host_context_block(system="BeOS", platform_string="BeOS-R5", machine="ppc")
        self.assertIn("BeOS", block)
        self.assertIn("POSIX", block)  # generic-hint fallback

    def test_includes_platform_string_and_arch(self):
        block = host_context_block(
            system="Linux",
            platform_string="Linux-6.8.0-111-generic-x86_64",
            machine="x86_64",
        )
        self.assertIn("Linux-6.8.0-111-generic-x86_64", block)
        self.assertIn("x86_64", block)


class TestPlannerHostAware(unittest.TestCase):
    def test_planner_prompt_includes_host_context(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("x")
        sys_msg = cloud.calls[0].messages[0].content
        self.assertIn("HOST CONTEXT", sys_msg)
        self.assertIn("Operating system:", sys_msg)

    def test_planner_on_linux_emits_systemctl_hint(self):
        with patch("sentricoon.planner.platform.system", return_value="Linux"):
            router, cloud, _ = _router_with_plan(VALID_PLAN)
            Planner(router).create_plan("x")
            sys_msg = cloud.calls[0].messages[0].content
            self.assertIn("systemctl", sys_msg)
            self.assertNotIn("sc query", sys_msg)

    def test_planner_on_windows_emits_sc_query_hint(self):
        with patch("sentricoon.planner.platform.system", return_value="Windows"):
            router, cloud, _ = _router_with_plan(VALID_PLAN)
            Planner(router).create_plan("x")
            sys_msg = cloud.calls[0].messages[0].content
            self.assertIn("sc query", sys_msg)
            self.assertIn("Audiosrv", sys_msg)
            self.assertNotIn("systemctl", sys_msg)


class TestWeakPredicateWarnings(unittest.TestCase):
    def _step(self, expected_state: str) -> Step:
        return Step(action="shell.run", expected_state=expected_state)

    def test_strong_predicates_no_warnings(self):
        plan = [
            self._step("exit code is 0 AND stdout contains 'Active: active (running)'"),
            self._step("system.info output has key 'system' equal to 'Linux'"),
            self._step("file '/etc/resolv.conf' exists and contains 'nameserver'"),
        ]
        self.assertEqual(weak_predicate_warnings(plan), [])

    def test_displayed_is_flagged(self):
        # Long enough to pass the length check, so the pattern triggers
        ws = weak_predicate_warnings([
            self._step("the kernel version number is displayed to the user")
        ])
        self.assertEqual(len(ws), 1)
        self.assertIn("passive", ws[0])

    def test_short_predicate_dominates_over_pattern(self):
        # A predicate that is BOTH too short AND has 'is displayed' should
        # be flagged for length, not for pattern — length is the dominant
        # issue and we don't double-warn per step.
        ws = weak_predicate_warnings([self._step("X is displayed")])  # 14 chars
        self.assertEqual(len(ws), 1)
        self.assertIn("too short", ws[0])

    def test_retrieved_is_flagged(self):
        ws = weak_predicate_warnings([self._step("OS information is retrieved successfully")])
        self.assertEqual(len(ws), 1)

    def test_should_is_flagged(self):
        ws = weak_predicate_warnings([
            self._step("the audio service should be running on this machine")
        ])
        self.assertEqual(len(ws), 1)
        self.assertIn("hopeful", ws[0])

    def test_subjective_is_flagged(self):
        ws = weak_predicate_warnings([
            self._step("the system output looks normal and within expected range")
        ])
        self.assertEqual(len(ws), 1)
        self.assertIn("subjective", ws[0])

    def test_too_short_is_flagged(self):
        ws = weak_predicate_warnings([self._step("ok")])
        self.assertEqual(len(ws), 1)
        self.assertIn("too short", ws[0])
        self.assertIn("specific observable", ws[0])

    def test_multiple_weak_steps_all_flagged(self):
        plan = [
            self._step("exit code is 0 AND stdout contains 'OK string here'"),  # strong
            self._step("memory usage is shown to the user"),  # weak
            self._step("disk space is reported in human-readable format"),  # weak
        ]
        ws = weak_predicate_warnings(plan)
        self.assertEqual(len(ws), 2)
        self.assertIn("step 2", ws[0])
        self.assertIn("step 3", ws[1])

    def test_warning_mentions_action_and_predicate(self):
        plan = [Step(action="system.info", expected_state="OS information is retrieved properly")]
        ws = weak_predicate_warnings(plan)
        self.assertIn("system.info", ws[0])
        self.assertIn("retrieved", ws[0])

    def test_observed_real_world_predicates_from_first_run(self):
        """The three predicates from the user's actual run on 2026-05-18 —
        all three should be flagged. Lock in the regression."""
        plan = [
            Step(action="system.info", expected_state="OS information is retrieved"),
            Step(action="shell.run", expected_state="kernel version is displayed"),
            Step(action="shell.run", expected_state="memory usage is displayed"),
        ]
        ws = weak_predicate_warnings(plan)
        self.assertEqual(len(ws), 3)


class TestPromptStrengthening(unittest.TestCase):
    """The strengthened prompt should contain the new accept/reject table."""

    def test_prompt_has_accept_reject_sections(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("x")
        sys_msg = cloud.calls[0].messages[0].content
        self.assertIn("ACCEPT", sys_msg)
        self.assertIn("REJECT", sys_msg)
        self.assertIn("displayed", sys_msg)  # the bad example we saw in the wild
        self.assertIn("nameserver", sys_msg)  # one of the good examples

    def test_prompt_warns_against_sudo(self):
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("x")
        sys_msg = cloud.calls[0].messages[0].content
        self.assertIn("privilege escalation", sys_msg.lower())
        self.assertIn("sudo", sys_msg.lower())

    def test_prompt_has_explicit_args_schema_example(self):
        """Regression: planner emitted args as a list because the schema
        wasn't shown concretely. Lock in the explicit example."""
        router, cloud, _ = _router_with_plan(VALID_PLAN)
        Planner(router).create_plan("x")
        sys_msg = cloud.calls[0].messages[0].content
        # The canonical example for shell.run's args shape
        self.assertIn('"argv"', sys_msg)
        # Counter-example warning
        self.assertIn("BAD:", sys_msg)
        # Explicit guidance for the no-args case
        self.assertIn("{}", sys_msg)


class TestPlanRiskWarnings(unittest.TestCase):
    def _shell(self, argv: list[str]) -> Step:
        return Step(
            action="shell.run",
            args={"argv": argv},
            expected_state="x" * 40,  # strong-enough predicate
        )

    def test_no_warnings_for_safe_argv(self):
        plan = [self._shell(["systemctl", "status", "Audiosrv"])]
        self.assertEqual(plan_risk_warnings(plan), [])

    def test_sudo_flagged(self):
        plan = [self._shell(["sudo", "find", "/tmp", "-size", "+50M"])]
        ws = plan_risk_warnings(plan)
        self.assertEqual(len(ws), 1)
        self.assertIn("privilege escalation", ws[0])
        self.assertIn("sudo", ws[0])

    def test_runas_flagged(self):
        plan = [self._shell(["runas", "/user:Administrator", "powershell"])]
        ws = plan_risk_warnings(plan)
        self.assertEqual(len(ws), 1)
        self.assertIn("runas", ws[0])

    def test_su_doas_gsudo_all_flagged(self):
        for tok in ("su", "doas", "gsudo"):
            ws = plan_risk_warnings([self._shell([tok, "command"])])
            self.assertEqual(len(ws), 1, msg=tok)

    def test_sudo_in_middle_of_argv_not_flagged(self):
        # `echo sudo` is not privilege escalation; only the head is checked
        plan = [self._shell(["echo", "sudo"])]
        self.assertEqual(plan_risk_warnings(plan), [])

    def test_case_insensitive(self):
        plan = [self._shell(["SUDO", "find"])]
        self.assertEqual(len(plan_risk_warnings(plan)), 1)

    def test_non_shell_action_ignored(self):
        plan = [Step(action="file.read", args={"path": "sudo"}, expected_state="x" * 40)]
        self.assertEqual(plan_risk_warnings(plan), [])

    def test_real_world_sudo_find_regression(self):
        """Lock in the exact pattern from the user's 2026-05-18 run."""
        plan = [self._shell(["sudo", "find", "/tmp", "-type", "f", "-size", "+50M"])]
        ws = plan_risk_warnings(plan)
        self.assertEqual(len(ws), 1)
        self.assertIn("sudo find", ws[0])


if __name__ == "__main__":
    unittest.main()
