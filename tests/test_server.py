"""HTTP tests for env_server — closes the gap that let session-id corruption
ship silently. Uses FastAPI TestClient so tests run in-process without a
real socket. The app holds module-level state (_env, _session_id,
_episode_active), so every test starts by establishing a known state.
"""
from fastapi.testclient import TestClient

from sregym import server as env_server
from sregym.server import app


def test_health_returns_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_reset_returns_session_id_and_observation():
    client = TestClient(app)
    r = client.post("/reset", json={"seed": 7})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["observation"], str)
    assert len(data["observation"]) > 0
    assert data["info"]["seed"] == 7
    assert "incident_type" in data["info"]
    assert isinstance(data["session_id"], str)
    assert len(data["session_id"]) == 32
    int(data["session_id"], 16)  # raises if not hex


def test_reset_returns_different_session_ids():
    client = TestClient(app)
    a = client.post("/reset", json={"seed": 7}).json()["session_id"]
    b = client.post("/reset", json={"seed": 7}).json()["session_id"]
    assert a != b


def test_step_with_correct_session_works():
    client = TestClient(app)
    sid = client.post("/reset", json={"seed": 7}).json()["session_id"]
    r = client.post("/step", json={
        "session_id": sid,
        "action": {"tool": "tail_logs", "args": {"service": "web"}},
    })
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["reward"], float)
    assert isinstance(data["terminated"], bool)
    assert isinstance(data["truncated"], bool)


def test_step_before_reset_returns_409():
    # Force the "no session ever started" branch by clearing module state
    # directly — TestClient can't get there otherwise because the module
    # initializes at import time and other tests have already called /reset.
    env_server._session_id = None
    env_server._episode_active = False
    client = TestClient(app)
    r = client.post("/step", json={
        "session_id": "x" * 32,
        "action": {"tool": "tail_logs", "args": {"service": "web"}},
    })
    assert r.status_code == 409
    assert r.json()["detail"] == "no episode active — call /reset first"


def test_step_with_wrong_session_returns_409():
    client = TestClient(app)
    client.post("/reset", json={"seed": 7})
    r = client.post("/step", json={
        "session_id": "deadbeef" * 4,
        "action": {"tool": "tail_logs", "args": {"service": "web"}},
    })
    assert r.status_code == 409
    assert r.json()["detail"] == "session_id mismatch — env may have been reset"


def test_step_after_terminate_returns_409():
    client = TestClient(app)
    reset_data = client.post("/reset", json={"seed": 7}).json()
    sid = reset_data["session_id"]
    # resolve always terminates regardless of correctness
    r = client.post("/step", json={
        "session_id": sid,
        "action": {"tool": "resolve", "args": {"root_cause": "x", "action": "y"}},
    })
    assert r.status_code == 200
    assert r.json()["terminated"] is True
    r2 = client.post("/step", json={
        "session_id": sid,
        "action": {"tool": "tail_logs", "args": {"service": "web"}},
    })
    assert r2.status_code == 409
    assert r2.json()["detail"] == "episode terminated — call /reset to start a new one"


def test_concurrent_clients_dont_silently_race():
    # The original bug: client A holds an old session, client B resets, A's
    # next /step silently operated on B's incident. Now A must get 409.
    client_a = TestClient(app)
    client_b = TestClient(app)
    token_a = client_a.post("/reset", json={"seed": 1}).json()["session_id"]
    token_b = client_b.post("/reset", json={"seed": 99}).json()["session_id"]
    assert token_a != token_b
    r = client_a.post("/step", json={
        "session_id": token_a,
        "action": {"tool": "tail_logs", "args": {"service": "web"}},
    })
    assert r.status_code == 409
    assert r.json()["detail"] == "session_id mismatch — env may have been reset"
