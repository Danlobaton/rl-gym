from dataclasses import dataclass
from pathlib import Path

from .generator import generate_incident
from .tools import tail_logs, query_metrics

WORKDIR_ROOT = Path("/tmp/sregym")


@dataclass
class StepResult:
    observation: str # What the agent sees next
    reward: float # Always 0 — scoring lives in Rubric, not Env
    terminated: bool # episode is done and ended naturally
    truncated: bool # episode hit the step limit
    info: dict # ground_truth on terminal steps; otherwise empty


class IncidentEnv:
    """Environment dynamics only — state, transitions, observations, termination.

    Scoring is the Rubric's job. step() always returns reward=0; the rubric
    computes the episode-level score after the trajectory is complete.
    """
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
        return obs, {
            "seed": seed,
            "incident_type": self.incident.incident_type,
            "affected_service": self.incident.affected_service,
        }

    def step(self, action: dict):
        self.step_count += 1
        tool = action.get("tool")
        args = action.get("args", {})

        if tool == "resolve":
            self.done = True
            # Ground truth is a *fact* the env knows — emitting it isn't scoring,
            # it's exposing the hidden state the rubric needs to grade against.
            return StepResult(
                observation="(episode complete)",
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"ground_truth": {
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
            reward=0.0,
            terminated=False,
            truncated=truncated,
            info={},
        )

    def _render_alert(self) -> str:
        return f"""ALERT: {self.incident.alert_text}
        Affected: {self.incident.affected_service}

        Tools available:
        tail_logs(service, lines=50)
        query_metrics(metric)
        resolve(root_cause, action)
        """
