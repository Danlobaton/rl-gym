# Stupid Simple RL Gym

> Sometimes this repo, some times it doesn't. I am constantly experiementing with it and breaking it.
> Funny story: When I told my my mom I was making an AI gym she thought I meant an AI powered workout app.

## The Story of This Repo

I got a Friday-night email about “a company making AI gyms,” and I instantly got curious. I dug through papers, talks, and examples until I understood the concepts.

**The problem:** I still had no clue what an RL gym felt like in practice.

**The answer:** I did what usually works best for me: built one from scratch and learned by banging my head against the wall until the “ohhh, sh**” moments turned into actual understanding.

So an embarrassing amount of weekend time later, I had a working RL training environment.

This repo is an SRE incident-triage environment. An LLM agent gets paged about a `CrashLoopBackOff` or a latency spike, then has to investigate and resolve the incident before its step budget runs out. The agent has three tools — `tail_logs(service, lines)` for shelling out to real `tail`/`grep`, `query_metrics(metric)` for reading the per-episode metric CSVs, and:

```python
resolve(root_cause, action)
```

…which scores the diagnosis and ends the episode.

I picked SRE because I was on-call the week before, it sucked, and it was the freshest thing on my mind.

**What I learned:** TL;DR this is hard. See [RESEARCH_LOG.md](./RESEARCH_LOG.md) for a running log of my learnings, progress, and design rationale.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the (separately maintained) design rationale.

## Things this isn't

- A production training environment. The reward function is loud, the task distribution is five hand-tuned templates, and I've trained exactly zero models against this.
- Original research. The patterns here are stolen, with affection, from Gymnasium, EnvPool, `verifiers`, and the public posts you've all written.
- Subtle.

## Roadmap (a.k.a. things I would ship if I had more time/sleep/caffeine)

- [x] **Traces for better debugging** this one is obvious
- [ ] **Token-level trajectories:** IN PROGRESS
- [ ] **A Rust-based execution path that skips the orchestrator entirely** and drives kernel container primitives directly. Trade Fargate's 5+ second task startup for sub-second cold starts at high concurrency.
- [x] **Add a defense againt reward hacking**
- [ ] **In-CLI `--concurrency N` rollout** that fans out across N containers via `asyncio.gather`. The horizontal-scaling path exists (multiple env containers + session tokens) but there's no single-command entry point yet.
- [ ] **Trainer-side integration** for token-fidelity trajectories: load the `inference.vllm` outputs (token_ids + logprobs already captured) into a PPO/GRPO trainer. The trace schema is ready; the consuming pipeline isn't built.
- [ ] **Trace upload to S3** from the coordinator container so the CDK-provisioned bucket actually receives traces in production deployments.
- [ ] **Mutating tools** (`restart`, `rollback`, `page`) and the `kubectl_describe` reader, plus a `ScriptedAgent` baseline so the floor is measurable without burning LLM tokens.

## Quickstart

```bash
pip install -e '.[coordinator]'
export ANTHROPIC_API_KEY=sk-ant-...   # or drop it in a .env at the repo root

# Run 5 rollouts (default) and write traces to ./traces/
python main.py rollout

# More episodes — one shared run_id across all of them
python main.py rollout --episodes 20
```

### Trace layout

Traces are written as JSONL under nested Hive-style partitions, with both the
`run_id` and the `seed` embedded in the filename so you can grep without
opening files:

```text
traces/
  dt=2026-04-28/                                            # UTC date
    ts=15-30-45Z/                                           # UTC start time of the run
      run_id=abc123def456_seed=000000.jsonl
      run_id=abc123def456_seed=000001.jsonl
      ...
    ts=18-12-03Z/                                           # a second run, same day
      run_id=def456abc789_seed=000000.jsonl
      ...
```

One `run_id` per `rollout` invocation; one file per episode; one `ts=` partition
per run, so multiple runs on the same day don't collide. Inside each file,
line 1 is trace metadata (run_id, run_started_at, seed, agent, task, reward,
ground truth, …) and lines 2..n are step records.

### Inspect a trace — `show`

```bash
# Pretty-print one trace (observations truncated to a 100-char preview)
python main.py show traces/dt=2026-04-28/ts=15-30-45Z/run_id=abc123def456_seed=000003.jsonl

# Print full observations — what the agent actually saw at step 7
python main.py show <path> --full

# Summary mode: one header-line per match. Cheap even on thousands of files.
python main.py show --filter 'traces/dt=2026-04-28/**/*.jsonl'      # whole day
python main.py show --filter 'traces/dt=2026-04-28/ts=15-30-45Z/*.jsonl'  # one run
```

### Verify env determinism — `replay`

```bash
# Re-runs the env from the trace's seed, feeds the recorded actions back,
# and checks observations + rewards are byte-identical. Exits non-zero on
# divergence — wedge it into CI to catch determinism regressions.
python main.py replay traces/dt=2026-04-28/ts=15-30-45Z/run_id=abc123def456_seed=000003.jsonl

# Audit every trace under traces/ — one PASS/FAIL line per file, plus totals.
# Exits non-zero if any trace fails. Drop into CI to guard the determinism
# contract across your whole archive.
python main.py replay audit
```

Replay verifies the env, not the agent. The LLM is allowed to be stochastic;
the env must be a pure function of (seed, action sequence).

### Run the env as a container — `env_server`

Per OpenEnv, the env can run as a FastAPI service inside a container; the
coordinator (this CLI) calls `/reset` and `/step` over HTTP. One container =
one episode at a time; scale horizontally by running multiple containers.

```bash
# Build the slim env image (no anthropic, no agent code — env only)
docker build -t sregym-env .

# Run it on :8000
docker run --rm -p 8000:8000 sregym-env

# In another terminal: rollout against the container
python main.py rollout --episodes 5 --env-url http://localhost:8000
```

Endpoints:

| Method | Path     | Body                     | Response                                                      |
|--------|----------|--------------------------|---------------------------------------------------------------|
| GET    | /health  | —                        | `{"status": "ok"}`                                            |
| POST   | /reset   | `{"seed": N}`            | `{"observation": str, "info": {seed, incident_type}}`         |
| POST   | /step    | `{"action": {...}}`      | `{"observation, reward, terminated, truncated, info"}`        |

For local development without docker:

```bash
uvicorn sregym.server:app --reload --port 8000
python main.py rollout --env-url http://localhost:8000
```
