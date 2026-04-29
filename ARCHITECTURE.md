# Architecture

This repo is an SRE RL gym built around one core constraint:

- Keep environment dynamics deterministic and replayable.
- Keep scoring (rubric/judge) separate from environment transition logic.
- Keep rollout artifacts first-class (trace -> replay -> audit).

## What Matters

- `Env` is pure episode dynamics (`reset`, `step`) with reward always `0.0`.
- `Rubric` scores completed trajectories out-of-band (including optional LLM judge).
- `Inference` backend is pluggable (`AnthropicClient` for eval/demo, `VllmClient` for trainable rollouts with token/logprob capture).
- `Trace` is the contract between rollout, replay, and downstream training.
- Same rollout path runs in-process or over HTTP (`EnvClient` / `server.py`).

---

## 1) Cloud Architecture

Derived from the CDK app composition in `infra/bin/app.ts` (`NetworkStack`, `DataStack`, `EnvStack`, `CoordinatorStack`).

```text
                         ┌───────────────────────────────────────┐
                         │             Internet / CI             │
                         └───────────────────┬───────────────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │ Coordinator ECS    │
                                   │ (rollout runner)   │
                                   └───────┬────────────┘
                                           │ dispatch / control plane
                    ┌──────────────────────┼──────────────────────────┐
                    │                      │                          │
          ┌─────────▼─────────┐  ┌────────▼────────┐      ┌─────────▼─────────┐
          │   SQS Job Queue    │  │   Runs Table     │      │   Traces Bucket    │
          │ (episode dispatch) │  │ (run metadata)   │      │ (JSONL artifacts)  │
          └────────────────────┘  └──────────────────┘      └────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │ Env ALB + Env ECS Fleet │
                              │ (/reset, /step OpenEnv) │
                              └────────────┬────────────┘
                                           │
                               ┌───────────▼───────────┐
                               │ Per-episode workdirs  │
                               │ logs/metrics/tmp state│
                               └───────────────────────┘

Cross-cutting: VPC + SG boundaries, KMS encryption, ECR images, secret for Anthropic key.
```

### Why this shape

- Coordinator and Env workers scale independently (control plane vs step execution).
- Queue decouples rollout submission from worker elasticity.
- Trace/object storage is immutable-by-convention and replay-friendly.
- Env fleet can be replaced (container runtime, infra substrate) without changing coordinator contracts.

---

## 2) Gym Runtime Architecture (Single Episode Dataflow)

```text
         ┌──────────────────────────────────────────────────────────────┐
         │                      sregym.cli rollout                     │
         └──────────────────────────────┬───────────────────────────────┘
                                        │
                              run_episode(env, inference, rubric)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
          ┌─────────▼──────────┐                  ┌─────────▼──────────┐
          │ Inference backend   │                  │   IncidentEnv       │
          │ Anthropic / vLLM    │                  │ reset/step dynamics │
          └─────────┬──────────┘                  └─────────┬──────────┘
                    │ model text/action JSON                │ obs/info
                    └───────────────┬───────────────────────┘
                                    │
                           ┌────────▼────────┐
                           │   TraceStep[]   │
                           │ action, obs,    │
                           │ latency, tokens │
                           └────────┬────────┘
                                    │ terminal trajectory
                           ┌────────▼────────┐
                           │ Rubric.score()  │
                           │ deterministic + │
                           │ optional judge  │
                           └────────┬────────┘
                                    │
                           ┌────────▼───────────┐
                           │ Trace (schema 1.x) │
                           │ write_trace()      │
                           └────────┬───────────┘
                                    │
                           replay/show/audit tooling
```

### Important contracts

- Env emits ground truth at terminal step; rubric consumes it.
- Rubric owns reward; env reward stays neutral to avoid mixing concerns.
- vLLM path captures token IDs + logprobs for RL-trainable traces; closed APIs leave token fields empty by design.

---

## 3) Software Architecture (Repo-Level Components)

```text
                    ┌────────────────────────────┐
                    │          main.py           │
                    │      -> sregym.cli         │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
 ┌───────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
 │  sregym.agent   │      │   sregym.env    │      │  sregym.rubric  │
 │ episode runner  │<---->│ dynamics/tools  │      │ episode scoring │
 └───────┬─────────┘      └────────┬────────┘      └────────┬────────┘
         │                          │                        │
         │                   ┌──────▼──────┐                 │
         │                   │ sregym.tools│                 │
         │                   │ log/metric  │                 │
         │                   └─────────────┘                 │
         │                                                   │
 ┌───────▼──────────────┐                          ┌─────────▼───────────┐
 │      inference/      │                          │    sregym.trace      │
 │ Anthropic / vLLM impl│                          │ trace schema + IO    │
 └───────┬──────────────┘                          └─────────┬───────────┘
         │                                                   │
         └──────────────┬────────────────────────────────────┘
                        │
                ┌───────▼────────┐
                │ sregym.replay  │
                │ determinism/audit
                └────────────────┘

Optional remote mode:
sregym.server (FastAPI OpenEnv endpoints) <-> sregym.client (HTTP env client)
```
