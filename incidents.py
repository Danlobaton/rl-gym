from dataclasses import dataclass
from pathlib import Path


@dataclass
class Incident:
    incident_type: str
    affected_service: str
    alert_text: str
    root_cause: str # the root cause of the incident e.g. "oomkilled"
    correct_action: str  # the correct action to take to resolve the incident
    workdir: Path  # per-episode dir; tools shell out against logs/, metrics/, kubectl_describe.txt
