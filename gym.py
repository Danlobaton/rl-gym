from dataclasses import dataclass
import random

from incidents import INCIDENTS, Incident

@dataclass
class StepResult:
    observation: str # What the agent sees next
    reward: float # How well did you do in this turn?
    terminated: bool # episode is done and ended naturally
    truncated: bool # episode hit the step limit
    info: dict # contains debugging, ground truth, etc.

# Tools
def tail_logs(incident: Incident, service: str, lines: int = 50) -> str:
    if service not in incident.logs:
        return f"ERROR: no log file for {service}"
    return "\n".join(incident.logs[service][-lines:])

def query_metrics(incident: Incident, service: str, metric: str) -> str:
    if metric not in incident.metrics:
        return f"ERROR: metric {metric} not found"
    values = incident.metrics[metric]
    return "\n".join(f"t={i} value={v}" for i, v in enumerate(values))


def resolve(incident: Incident, root_cause: str, action: str) -> tuple[float, dict]:
    """Resolve the incident by calling the correct action to resolve the incident."""
    breakdown = {"completion": 0.5, "root_cause": 0.0, "action": 0.0}
    if root_cause == incident.root_cause:
        breakdown["root_cause"] = 1.5
    else:
        breakdown["root_cause"] = -1.0
    if action == incident.correct_action:
        breakdown["action"] = 1.0
    else:
        breakdown["action"] = -0.5
    return sum(breakdown.values()), breakdown

class IncidentEnv:
    MAX_STEPS = 10

    def reset(self, seed: int):
        rng = random.Random(seed)
        self.incident = rng.choice(INCIDENTS)
        self.step_count = 0
        self.done = False # This is internal environment state, not part of the API
        obs = self._render_alert()
        return obs, {"seed": seed}

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

        # read-only tool
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
