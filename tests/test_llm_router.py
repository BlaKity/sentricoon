from __future__ import annotations

import unittest

from sentricoon.llm import (
    LLMBackend,
    LLMRequest,
    LLMRole,
    LLMRouter,
    Message,
    MockBackend,
    RouterConfig,
)
from sentricoon.llm.router import RouterError


def _req(text: str = "hi") -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content=text)])


class TestMockBackend(unittest.TestCase):
    def test_script_returns_in_order(self):
        b = MockBackend(script=["one", "two"])
        self.assertEqual(b.complete(_req()).content, "one")
        self.assertEqual(b.complete(_req()).content, "two")

    def test_script_exhaustion_raises(self):
        b = MockBackend(script=["only"])
        b.complete(_req())
        with self.assertRaises(RuntimeError):
            b.complete(_req())

    def test_responder_callable(self):
        b = MockBackend(responder=lambda r: f"echo:{r.messages[-1].content}")
        self.assertEqual(b.complete(_req("hello")).content, "echo:hello")

    def test_records_calls(self):
        b = MockBackend(script=["a", "b"])
        b.complete(_req("first"))
        b.complete(_req("second"))
        self.assertEqual(len(b.calls), 2)
        self.assertEqual(b.calls[0].messages[0].content, "first")


class TestLLMRouter(unittest.TestCase):
    def _hybrid(self):
        cloud = MockBackend(responder=lambda r: "cloud-reply", name="cloud", model="cloud-1")
        local = MockBackend(responder=lambda r: "local-reply", name="local", model="local-1")
        router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        router.register(cloud)
        router.register(local)
        return router, cloud, local

    def test_planner_routes_to_cloud(self):
        router, cloud, local = self._hybrid()
        resp = router.complete(LLMRole.PLANNER, _req())
        self.assertEqual(resp.content, "cloud-reply")
        self.assertEqual(resp.backend, "cloud")
        self.assertEqual(router.call_counts.get("cloud"), 1)
        self.assertIsNone(router.call_counts.get("local"))

    def test_verifier_routes_to_local(self):
        router, cloud, local = self._hybrid()
        resp = router.complete(LLMRole.VERIFIER, _req())
        self.assertEqual(resp.backend, "local")

    def test_diagnoser_and_critic_route_to_cloud(self):
        router, *_ = self._hybrid()
        self.assertEqual(router.complete(LLMRole.DIAGNOSER, _req()).backend, "cloud")
        self.assertEqual(router.complete(LLMRole.CRITIC, _req()).backend, "cloud")

    def test_input_repair_routes_to_local(self):
        router, *_ = self._hybrid()
        self.assertEqual(router.complete(LLMRole.INPUT_REPAIR, _req()).backend, "local")

    def test_unregistered_backend_raises(self):
        router = LLMRouter(RouterConfig.default(cloud="cloud", local="local"))
        # No backends registered
        with self.assertRaises(RouterError):
            router.complete(LLMRole.PLANNER, _req())

    def test_all_one_collapses_roles(self):
        backend = MockBackend(responder=lambda r: "single", name="solo", model="solo-1")
        router = LLMRouter(RouterConfig.all_one("solo"))
        router.register(backend)
        for role in LLMRole:
            self.assertEqual(router.complete(role, _req()).backend, "solo")

    def test_default_backend_fallback(self):
        cloud = MockBackend(responder=lambda r: "x", name="cloud", model="c")
        cfg = RouterConfig(role_map={}, default_backend="cloud")
        router = LLMRouter(cfg)
        router.register(cloud)
        self.assertEqual(router.complete(LLMRole.PLANNER, _req()).backend, "cloud")

    def test_no_default_no_map_raises(self):
        cloud = MockBackend(responder=lambda r: "x", name="cloud", model="c")
        router = LLMRouter(RouterConfig(role_map={}, default_backend=None))
        router.register(cloud)
        with self.assertRaises(RouterError):
            router.complete(LLMRole.PLANNER, _req())

    def test_call_counts_track_per_backend(self):
        router, *_ = self._hybrid()
        router.complete(LLMRole.PLANNER, _req())
        router.complete(LLMRole.PLANNER, _req())
        router.complete(LLMRole.VERIFIER, _req())
        router.complete(LLMRole.VERIFIER, _req())
        router.complete(LLMRole.VERIFIER, _req())
        # 5 calls total: 2 cloud + 3 local — the cost split the user wants
        self.assertEqual(router.call_counts["cloud"], 2)
        self.assertEqual(router.call_counts["local"], 3)


class TestBackendProtocol(unittest.TestCase):
    def test_mock_satisfies_protocol(self):
        b = MockBackend(script=["x"])
        self.assertIsInstance(b, LLMBackend)


if __name__ == "__main__":
    unittest.main()
