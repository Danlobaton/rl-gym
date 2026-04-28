# Intro

This doc serves as both an intro to the repo architecture as well as learning doc on what I learned about RL environments.

## Domain model

Tasks

A task is a (problem, hidden ground truth) pair. For SREGym, a task is one synthetic incident.
Each task has:

- Public fields the agent sees on `reset()`: incident type, affected service, related services, alert text, available tools, step budget.
- Hidden fields only the verifier sees: root cause slug, correct remediation action, expected investigation pattern, list of known safety-violation predicates.
- Materialized state written to a per-episode tmpdir: synthetic logs, metrics CSVs, kubectl describe blobs. Generated deterministically from the seed. Causally consistent with the hidden ground truth.

## High-Level Architecture of RL Environments

```text
                    ┌────────────────────┐
                    │  Trainer cluster   │  ← FSDP, GPU-heavy, its own SLA
                    └─────────┬──────────┘
                              │ trajectories (token-in/token-out)
                    ┌─────────▼──────────┐
                    │  Rollout service   │  ← vLLM, GPU pool, async
                    │    (inference)     │
                    └─────────┬──────────┘
                              │ HTTP (OpenEnv) or in-proc (verifiers)
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐     ┌─────────┐     ┌─────────┐
         │   Env   │     │   Env   │     │   Env   │  ← Container fleet
         │ replica │     │ replica │     │ replica │     CPU-heavy, cheap
         └────┬────┘     └────┬────┘     └────┬────┘     scales horizontally
              │               │               │
              └────── traces (S3 / DB) ───────┘  ← observability path
```
