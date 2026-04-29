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

Tools need to shell out to `tail`, `grep`, etc., so incidents moved from in-memory dataclasses to a per-episode workdir (`/tmp/sregym/ep-<seed>/`) containing logs, metric CSVs, and a kubectl-describe blob. Two invariants: (1) same seed → byte-identical workdir (keeps replay working), and (2) file contents are causally consistent with the hidden root cause so the agent reasons from evidence, not canned strings.

Determinism uses `subrng(parent, label)` — SHA-256-derived child RNGs namespaced by label — so adding or reordering generation steps doesn't shift downstream draws. Per-seed workdir naming gives free isolation for concurrent episodes and a convenient debugging surface (`ls /tmp/sregym/ep-47/`).

## Phase 5: (Stupidly) Scaling

This is where my laptop officially gave out. I tried running (stupidly, because I was curious on what would happen)

```bash
for i in $(seq 0 79); do
    docker run --rm --memory 512m --cpus 1 -e ANTHROPIC_API_KEY \
        -v $(pwd)/traces:/app/traces rl_gym:dev \
        rollout --agent llm --episodes 1 --seed-start $i &
done
```

Care to say... my laptop did not like that.

**The wall I hit:** "I am spending more resources managing and bookkeping my containers than actually running those containers"

But granted, here are some of the issues I saw that I think could really bite when scaling in prod workloads:

- **Control-plane serialization.** Docker daemon and K8s API server are the same bottleneck shape — one serialization point processing every container request. Cloud raises the ceiling (50 → 1000 containers/sec) but doesn't remove it.
- **Per-container creation cost.** cgroups, namespaces, mounts — that's kernel work. Same cost per container on a laptop or a cloud node; cloud just gives you more nodes to parallelize across.
- **Image pull & warm-up.** Cold pull + Python interpreter startup takes 1-2s whether it's local or Fargate. No free lunch.

So if you have a frontier lab running GPUs that cost $2-4/hour each, running by the 1,000s. The training pipeline looks like `env containers → trajectories → trainer GPUs consume them`. If env containers can't keep up, the trainer GPUs sit idle, which is essentially lighting money on fire. So the real value is "how many trajectories per second can you deliver to a trainer."

So I looked into how others solved this and found [Prime's write up for Intellect-3](https://www.primeintellect.ai/blog/intellect-3):

> While Kubernetes provides the primitives for container management, standard architectural patterns are insufficient for the throughput required by high-velocity training. To overcome these limitations, we built Prime Sandboxes: a fully redesigned, high-performance execution layer that bypasses the Kubernetes control plane, delivers near–local-process latency through a direct Rust-to-pod execution path, achieves sub-10-second startup at massive concurrency.

Interesting.

