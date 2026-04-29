"""HTTP client that exposes the IncidentEnv reset/step interface against a
remote env server. Implements the EnvProtocol so `run_episode` doesn't need to
know whether it's talking to a local or remote env."""
import time

import httpx

from .env import StepResult


class EnvClient:
    def __init__(self, base_url: str, timeout: float = 30.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._session_id: str | None = None
        # Fail-fast preflight: a wrong --env-url should error before the rollout
        # starts, not after the agent has already loaded and a /reset is issued.
        # Let httpx exceptions propagate.
        r = self._client.get("/health")
        r.raise_for_status()

    def __enter__(self) -> "EnvClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self, seed: int) -> tuple[str, dict]:
        r = self._client.post("/reset", json={"seed": seed})
        r.raise_for_status()
        data = r.json()
        self._session_id = data["session_id"]
        return data["observation"], data["info"]

    def step(self, action: dict) -> StepResult:
        # Retry only transient transport-layer failures. 4xx/5xx are caller-or-
        # server logic errors and surface via raise_for_status() unretried.
        backoffs = [0.5, 1.0, 2.0]
        attempt = 0
        while True:
            try:
                r = self._client.post(
                    "/step",
                    json={"session_id": self._session_id, "action": action},
                )
                break
            except (httpx.ConnectError, httpx.ReadError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
                attempt += 1
        r.raise_for_status()
        data = r.json()
        return StepResult(
            observation=data["observation"],
            reward=data["reward"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            info=data["info"],
        )

    def close(self) -> None:
        self._client.close()
