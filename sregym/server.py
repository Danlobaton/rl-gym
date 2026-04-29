"""HTTP frontend for IncidentEnv (OpenEnv spec).

The env is stateful, HTTP is not. We bridge that gap with a session token:
/reset mints one, /step requires it. Mismatched or missing tokens become
clean 409s rather than silent state corruption — see /step's three guards.

One container = one episode at a time; horizontal scaling = more containers.
"""
import threading
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .env import IncidentEnv

app = FastAPI(title="sregym env", version="0.1")

# Shared env state — _lock guards transitions (FastAPI's sync handlers run
# from a thread pool). _session_id is the receipt for the latest /reset;
# _episode_active flips False when the env returns terminated/truncated.
_env = IncidentEnv()
_lock = threading.Lock()
_session_id: str | None = None
_episode_active: bool = False


class ResetRequest(BaseModel):
    seed: int


class ResetResponse(BaseModel):
    observation: str
    info: dict[str, Any]
    session_id: str


class StepRequest(BaseModel):
    session_id: str
    action: dict[str, Any]


class StepResponse(BaseModel):
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# /reset always succeeds. A second call rotates the token and displaces any
# in-flight session; the displaced client learns of the conflict at its next
# /step (mismatch branch) rather than silently writing against new state.
@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    global _session_id, _episode_active
    with _lock:
        new_id = uuid.uuid4().hex
        obs, info = _env.reset(req.seed)
        _session_id = new_id
        _episode_active = True
    return ResetResponse(observation=obs, info=info, session_id=new_id)


# Three guards, three distinct details — coordinators route recovery on the
# detail string: forgot-reset vs. someone-else-reset vs. stepped-past-done.
# _session_id is intentionally retained on termination so the third branch
# fires for "polled past done"; clearer than the second branch's "mismatch".
@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    global _episode_active
    with _lock:
        if _session_id is None:
            raise HTTPException(status_code=409, detail="no episode active — call /reset first")
        if req.session_id != _session_id:
            raise HTTPException(status_code=409, detail="session_id mismatch — env may have been reset")
        if not _episode_active:
            raise HTTPException(status_code=409, detail="episode terminated — call /reset to start a new one")
        result = _env.step(req.action)
        if result.terminated or result.truncated:
            _episode_active = False
    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
        info=result.info,
    )
