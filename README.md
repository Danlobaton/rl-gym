# Stupid Simple RL Gym

> An RL training gym where the baseline agent's first move is always to page the on-call.
> (Just like real production.)

## The Story of This Repo

I got an email on a Friday night about “a company making AI gyms,” and it immediately caught my interest: heavy infrastructure, backend systems, RL environments, the works. It sounded genuinely cool, so I started digging. I read papers, watched talks, went through examples, and tried to understand what these systems were actually doing. Conceptually, it made sense.

**The problem:** I still had absolutely no clue what an RL gym felt like in practice.

**The Answer:** I did what usually works best for me, built one from scratch and learned by banging my head against the wall until the “ohhh, sh**” moments started turning into actual understanding.

So an embarrassing amount of weekend time later (and work week), I had a working RL training environment.

This repo is an SRE incident-triage environment. An LLM agent gets paged about a `CrashLoopBackOff` or a latency spike, then has to investigate and resolve the incident before its step budget runs out.
The agent has access to read-only diagnostic tools like logs, metrics, and `kubectl describe`, as well as mutating tools like restart, rollback, and page. To finish successfully, it must call:

```python
resolve(root_cause, action)
```

I picked SRE because I was on-call the week before, it sucked, and it was the freshest thing on my mind.

**What I learned:** This is hard

## What's actually under the hood

- **Procedurally generated incidents.** Each seed yields a different incident type, service, and root cause, plus a per-episode tmpdir of synthetic logs and metrics consistent with the hidden ground truth. The OOMKilled crashloop has cut-off logs; the bad-deploy error spike correlates with a build SHA change. The agent has to reverse-engineer the cause from the evidence — same as a real on-call.
- **A real subprocess sandbox.** `tail_logs` literally runs `tail -n 50`. Tools shell out to the per-episode tmpdir with timeouts. The agent is interacting with software, not Python pretending to be software, which means swapping the fake filesystem for a real container is mechanical, not a rewrite.
- **Async parallel rollouts** with a concurrency semaphore. EnvPool's job in 60 lines.
- **Pluggable inference and trace storage.** Anthropic API on a laptop, Bedrock in a VPC. LocalTracer to disk, S3Tracer to a Hive-partitioned prefix. Same wire format both ways. One env var flips it.
- **AWS-deployable from the start.** The container image runs identically on a laptop and on Fargate. CDK stack lands next.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design rationale.

## The baseline is approximately how every LLM agent ships to production

Without reading any logs, the `ScriptedAgent` immediately calls `resolve(root_cause="unknown", action="page_oncall")`. It scores -1.0 on average and incurs no safety penalty. Cowardice is, technically, safe.

The reward shaping is decomposed into separately-tunable terms specifically so that when an agent learns to cheese one of them you can see *which one* and nerf it. Reward hacking is a sport; the verifier is the goalkeeper.

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

## Things this isn't

- A production training environment. The reward function is loud, the task distribution is five hand-tuned templates, and I've trained exactly zero models against this.
- Original research. The patterns here are stolen, with affection, from Gymnasium, EnvPool, `verifiers`, and the public posts you've all written.
- Subtle.

## Roadmap (a.k.a. things I would ship if I had more time/sleep/caffeine)

- CDK stack: ECS Fargate Spot, SQS dispatch, S3 traces, DynamoDB scoreboard. Same Docker image, different env vars.
- LLM-as-judge for the diagnosis-quality term, async, on a sampled subset.
- A second env (terminal-bench-flavored) sharing the same base classes. The whole point of getting the contracts right is that the second one is cheap.
