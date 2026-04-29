# Research Log

This is meant to serve as a runnin log of my research and learnings in the space as I developer the repo. I will do my best to keep it updated.

## First - What even is an RL Gym?

An RL gym (or training environment) is a piece of software that lets an AI agent practice a task. It does four things:

Presents a problem to the agent

1. Lets the agent take actions and see results
2. Scores how well the agent did
3. Records what happened so you can train the model to do better next time

The classic example is OpenAI's CartPole environment: a pole balanced on a cart, the agent moves the cart left or right, the reward is "stayed up." That's an RL gym.

For LLM agents the same idea applies but the surface looks different. Instead of a cart with a pole, the "world" is a fake software system — logs, files, APIs. Instead of "move left/right," the agent calls tools. Instead of "stayed up," in my case the reward is "did you diagnose the incident correctly without breaking anything."

Terminology:

- Episode: One start-to-finish run. One incident, from alert to resolution.
- Rollout: The act of running an episode. "I ran 200 rollouts" = "I did 200 episodes."
- Trajectory: the sequence of (state, action, reward) tuples produced by an episode. This is what the trainer eventually learns from.
- Trace: the human-readable log of an episode. Same content as a trajectory, different audience.
- Reward: the score the agent gets. Higher = better. Can be a single number or a vector of separate concerns. I like to call these brownie points.
- Verifier (or rubric): the function that computes the reward.

## Phase 1: The naive in-process gym

**The wall I hit:** "I can't tell what my agent is doing."

I built the smallest possible in-process gym: one Python file, one hard-coded task, one synchronous agent loop, and no supporting systems. No traces, no logs, no sandbox. Tools were just Python functions returning strings, and the agent called Anthropic in a `while not done` loop.

The LLM was completing real episodes, rewards printing, and everything was alive. The main problem is that I had no ideas of what was actually happening turn by turn. I could see outcomes, but not whether tools returned the right thing or whether the model reasoned correctly about those returns.

Learnings:

- The contract is the easiest part of the system
- The work comes in literally everything around it

The mental model: the Gymnasium contract is really just `reset` and `step`. The real differences between gyms are not in the contract itself, but the logic around it.

## Phase 2: Tracing Layer

**The wall I hit:** "I have no idea what is going on and it is a pain to find out why an agent scored so low."

A trace is a structured record of one episode. I needed to move from ad-hoc prints to something I could actually interrogate. Every episode now emits an append-only JSONL record containing the seed, each `(action, observation, reward)` tuple, ground truth, and timing metadata. I also built a `show` command to pretty-print a trace so I could inspect an episode quickly.

Then I built a replay command. Given a trace, I re-ran the environment from the same seed, fed the same actions back in, and checked for byte-identical observations and rewards. If replay diverged, the environment was not deterministic and I had a real bug. The root causes were consistently one of three things: hidden `random.Random()` usage, dictionary iteration order assumptions, or wall-clock dependencies.

Once I had traces, the next problem was querying them. I added Hive-style partitioning (`agent=X/dt=Y/`) so I could filter without scanning everything. At scale this matters enormously — if you have 50 million trace files spread across two years and ten agents, a query that filters by agent and date might touch 0.1% of them instead of all of them. The cost difference is about three orders of magnitude, both in wall-clock time and in dollars (Athena charges per byte scanned). The convention works because it requires zero metadata infrastructure. No database, no index, no catalog. The folder structure is the index. I made this decision on the fly and it held up.

I also read [Han Lee's taxonomy paper](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/), especially the trajectory-vs-trace distinction. What I built was a trace, not a trajectory. A trajectory is what the trainer needs at token level. At this stage, trace quality was the priority. In production systems both are shipped, and ProRL Agent's token-in/token-out design exists because re-tokenizing from string traces introduces drift.

My mental model now: episode records have two first-class consumers — the trainer (token fidelity, optimization correctness) and the human/observability stack (debuggability). I can defer trainer-side sophistication early, but I cannot defer observability and still call the gym usable.

## Phase 3: File System Materialization

**The wall I hit:** "I can't ship a real subprocess sandbox without files on disk."

The previous naive gym kept incidents in-memory: I hardcoded `INCIDENTS` list of dataclass instances that the env picked from on `reset()`. That worked for the in-process loop, but it blocked the next thing on the roadmap — having tools shell out to real Unix utilities (`tail`, `cat`, `grep`) against synthetic logs and metrics. You can't `tail -n 50 logs/payments-api.log` if there's no file.

So this phase moves materialized state from Python lists to actual files in a per-episode workdir.

**The contract.** Every seed produces a workdir at `/tmp/sregym/ep-<seed>/` containing:

- `logs/<service>.log` — plain text, one log line per entry
- `metrics/<metric>.csv` — `t,value` time series
- `kubectl_describe.txt` — pod-status blob

Two design considerations:

1. **Deterministic from seed.** Same seed → byte-identical workdir contents. This is what keeps `replay` working: env outputs are still a pure function of (seed, action sequence). The replay test suite caught one non-deterministic-looking moment during development (a `random.shuffle` on a list whose iteration order I'd assumed was stable); it surfaced as a single test failure with a clear MISMATCH line, which is exactly what that infrastructure exists for.
2. **Causally consistent with the hidden root cause.** If the root cause is `oom_killed`, the memory metric climbs to its limit and the logs cut off mid-write. If it's `bad_deploy`, errors spike right after a deploy event in the logs. The agent has to reason from real evidence, not pattern-match on canned strings.

**Determinism via sub-rng composition.** Naive determinism is "one seed, threaded through all generation calls in order." It works until you reorder generation steps (or add a new one) and every downstream draw shifts. Refactor-fragile, exactly the kind of trap that makes determinism feel like luck rather than a property.

Instead, generation uses a `subrng(parent, label)` helper that derives a child RNG from a parent (seed-or-RNG), namespaced by a string label. It hashes `(parent_state, label) → SHA-256 → child seed`. Three properties make this safe:

**Why the workdir is per-episode and seed-named.** Two reasons.

First, when episodes run concurrently later, each one needs its own filesystem so they don't trample each other's state. Naming the dir after the seed gives that for free — concurrent episodes have different seeds, therefore different dirs.

Second, materialized state is a debugging surface. If you ever want to know "what did the agent see in episode 47?", you can `ls /tmp/sregym/ep-47/` and inspect the actual files. No need to instrument the env or replay anything.

