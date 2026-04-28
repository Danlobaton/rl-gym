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
# No deps, no API key — establish the floor:
python run.py rollout --agent scripted --episodes 5

# Real LLM, real sandbox, real traces:
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python run.py rollout --agent llm --episodes 20 --concurrency 8

# Inspect a trace:
python run.py show ./traces/.../seed000003.jsonl.gz
```

## Things this isn't

- A production training environment. The reward function is loud, the task distribution is five hand-tuned templates, and I've trained exactly zero models against this.
- Original research. The patterns here are stolen, with affection, from Gymnasium, EnvPool, `verifiers`, and the public posts you've all written.
- Subtle.

## Roadmap (a.k.a. things I would ship if I had more time/sleep/caffeine)

- CDK stack: ECS Fargate Spot, SQS dispatch, S3 traces, DynamoDB scoreboard. Same Docker image, different env vars.
- A `replay` command that re-runs a trace from its seed and verifies byte-identical actions. "Any failed rollout is reproducible" is the actual selling point of an env platform; everything else is decoration.
- LLM-as-judge for the diagnosis-quality term, async, on a sampled subset.
- A second env (terminal-bench-flavored) sharing the same base classes. The whole point of getting the contracts right is that the second one is cheap.
