from dataclasses import dataclass
from pathlib import Path

from .generator import generate_incident
from .tools import tail_logs, query_metrics, resolve

WORKDIR_ROOT = Path("/tmp/sregym")


@dataclass
class StepResult:
    observation: str # What the agent sees next
    reward: float # How well did you do in this turn?
    terminated: bool # episode is done and ended naturally
    truncated: bool # episode hit the step limit
    info: dict # contains debugging, ground truth, etc.


class IncidentEnv:
    MAX_STEPS = 10

    def reset(self, seed: int):
        # generate_incident wipes any prior workdir at WORKDIR_ROOT/ep-<seed>/
        # before regenerating — idempotent, so a crashed episode doesn't leak
        # state into the next reset.
        self.incident = generate_incident(seed, WORKDIR_ROOT)
        self.step_count = 0
        self.done = False # This is internal environment state, not part of the API
        obs = self._render_alert()
        # info carries metadata the coordinator needs but the agent never sees;
        # incident_type is task-meta for traces and works over HTTP without
        # exposing env.incident.
        return obs, {"seed": seed, "incident_type": self.incident.incident_type}

    def step(self, action: dict):
        self.step_count += 1
        tool = action.get("tool")
        args = action.get("args", {})

        if tool == "resolve":
            reward, breakdown = resolve(self.incident, **args)
            self.done = True
            return StepResult(
                observation="(episode complete)",
                reward=reward,
                terminated=True,
                truncated=False,
                info={"breakdown": breakdown, "ground_truth": {
                    "root_cause": self.incident.root_cause,
                    "correct_action": self.incident.correct_action,
                }},
            )

        if tool == "tail_logs":
            obs = tail_logs(self.incident, **args)
        elif tool == "query_metrics":
            obs = query_metrics(self.incident, **args)
        else:
            obs = f"ERROR: unknown tool {tool}"

        truncated = self.step_count >= self.MAX_STEPS
        return StepResult(
            observation=obs,
            reward=-0.02,  # small per-step cost, see below
            terminated=False,
            truncated=truncated,
            info={},
        )

    def _render_alert(self) -> str:
        return f"""ALERT: {self.incident.alert_text}
        Affected: {self.incident.affected_service}

        Tools available:
        tail_logs(service, lines=50)
        query_metrics(service, metric)
        resolve(root_cause, action)
        """
