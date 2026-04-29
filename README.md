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

**What I learned:** TL;DR this is hard. See [RESEARCH_LOG.md](./RESEARCH_LOG.md) for a running log of my findings, learnings, progress.

## What's actually under the hood

- **Procedurally generated incidents.** Each seed yields a different incident type, service, and root cause, plus a per-episode workdir of synthetic logs (`logs/<service>.log`), metric CSVs (`metrics/<metric>.csv`), and a `kubectl_describe.txt` blob — all causally consistent with the hidden ground truth. OOMKilled crashloops have cut-off logs; bad-deploy error spikes correlate with a deploy event in the log. The agent has to reverse-engineer the cause from the evidence.
- **A real subprocess sandbox.** `tail_logs` literally runs `tail -n 50` (with optional `| grep PATTERN`). `query_metrics` shells out to `tail | awk` against the metric CSVs. Every agent-controlled string passes through `shlex.quote`; tools have a 5-second timeout. Swapping the fake filesystem for a real container is mechanical, not a rewrite.
- **Containerized env behind an HTTP API.** A FastAPI server (`sregym/server.py`) exposes `/reset` and `/step` per OpenEnv. One container = one episode at a time, enforced by a session-token mechanism that rejects stale or post-terminate `/step` calls with a clean 409. Scale by running multiple containers; the rollout CLI reaches any of them via `--env-url`.
- **Trace persistence with replay-based determinism checks.** Every episode writes a JSONL trace under nested Hive-style partitions (`dt=YYYY-MM-DD/ts=HH-MM-SSZ/`). `python main.py replay <trace>` re-runs the env from the same seed, feeds the recorded actions back, and verifies byte-identical observations + rewards. `replay audit` does this across every trace under `traces/` and exits non-zero on any divergence.
- **Pluggable inference behind a Protocol.** Two backends today: `inference.anthropic_client.AnthropicClient` (text only — closed API, not RL-trainable) and `inference.vllm.VllmClient` (token IDs + per-token logprobs from vLLM's native `/generate` endpoint, the trace-vs-trajectory pivot for RL training).
- **Episode scoring as a separate `Rubric` class.** Weighted scoring terms (step penalty, completion, root-cause match, action match) plus an LLM-as-judge `rationale_quality` term that fires on a sampled subset (deterministic on `run_id`) and records its reasoning in trace diagnostics.
- **AWS deployment via CDK** (under `infra/`). Four stacks — VPC + ECS cluster; S3 + SQS + DDB + KMS + Glue + ECR; env-worker Fargate behind an internal ALB; coordinator Fargate with no inbound surface. `cdk synth` is verified clean.

### What's *not* in the code yet (despite earlier drafts of this README implying otherwise)

- **Concurrent rollouts from a single CLI invocation.** `python main.py rollout` is a sequential `for` loop today. Horizontal scaling via more containers is supported by the architecture, but there's no in-CLI `--concurrency N` flag yet.
- **`kubectl describe`, `restart`, `rollback`, `page` tools.** A `kubectl_describe.txt` file is generated into each workdir, but no agent tool currently reads it. The mutating tools don't exist.
- **Bedrock inference, LocalTracer/S3Tracer pluggable storage, ScriptedAgent baseline.** Not implemented; on the roadmap.
- **Trace upload to the CDK-provisioned S3 bucket.** The CDK stack ships the bucket + Glue catalog; the Python writer puts JSONL on disk locally. Wiring the writer to S3 inside container deployments is roadmap work.

The reward shaping is decomposed into separately-tunable terms specifically so that when an agent learns to cheese one of them you can see *which one* and nerf it. Reward hacking is a sport; the verifier is the goalkeeper.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the (separately maintained) design rationale.

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

- **A Rust-based execution path that skips the orchestrator entirely** and drives kernel container primitives directly. Trade Fargate's 5+ second task startup for sub-second cold starts at high concurrency.
- **In-CLI `--concurrency N` rollout** that fans out across N containers via `asyncio.gather`. The horizontal-scaling path exists (multiple env containers + session tokens) but there's no single-command entry point yet.
- **Trainer-side integration** for token-fidelity trajectories: load the `inference.vllm` outputs (token_ids + logprobs already captured) into a PPO/GRPO trainer. The trace schema is ready; the consuming pipeline isn't built.
- **Trace upload to S3** from the coordinator container so the CDK-provisioned bucket actually receives traces in production deployments.
- **Mutating tools** (`restart`, `rollback`, `page`) and the `kubectl_describe` reader, plus a `ScriptedAgent` baseline so the floor is measurable without burning LLM tokens.
- **A second env** (terminal-bench-flavored) sharing the same base classes. The whole point of getting the contracts right is that the second one is cheap.
