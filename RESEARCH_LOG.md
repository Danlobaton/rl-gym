# Research Log

This is meant to serve as a runnin log of my research and learnings in the space as I developer the repo. I will do my best to keep it updated.

Goal: Build an SRE Agent training gym

## First - What even is an RL Gym?

An RL gym (or training environment) is a piece of software that lets an AI agent practice a task. It does four things:

Presents a problem to the agent

1. Lets the agent take actions and see results
2. Scores how well the agent did
3. Records what happened so you can train the model to do better next time

The classic example is OpenAI's CartPole environment: a pole balanced on a cart, the agent moves the cart left or right, the reward is "stayed up." That's an RL gym.

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

I wired every episode to emit an append-only JSONL trace — seed, `(action, observation, reward)` tuples, ground truth, timing — and wrote a `show` command to pretty-print them. Immediately useful. Then I built `replay`: re-run the env from the same seed, feed the same actions, diff for byte-identical outputs. Every time replay diverged I had a real bug — hidden `random.Random()` usage, dict iteration order assumptions, or wall-clock dependencies. It caught all three.

For querying I added Hive-style partitions (`agent=X/dt=Y/`). The folder structure is the index, no catalog needed. At scale (~50M files) that makes a diff between scanning 0.1% vs 100% of files, roughly three orders of magnitude in search time. I made this choice on the fly and it held up.

Reading [Han Lee's taxonomy paper](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/) clarified what I'd built: a trace (string-level), not a trajectory (token-level). Trainer-side token fidelity can come later; I can't defer observability and still call the gym usable.

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

Interesting. I would LOVE to take crack at this problem-space on my own but for the sake of this project i'll glost over it.

## Phase 4: The Agent is Cheating

After burning through enough tokens I started actually reading the traces — and that's when I caught the agent cheating. I'd read that reward hacking was a thing, but seeing it happen irl was pretty wild.

Here are three most interesting way I caught Claude cheating:

1. **Skipping the investigation entirely.** My `resolve` reward is `{completion: 0.5, root_cause: ±, action: ±}`. The 0.5 for *completion* fires unconditionally. So the agent learned that calling `resolve` on step 1 with a random guess from the four root causes nets a positive expected return on a quarter of episodes — and the remaining tools cost `-0.02` per step. Reading logs is *literally negative reward* unless it flips your guess. The cheapest policy is "guess fast, never investigate."

2. **Hacking `grep`.** `tail_logs` accepts a `grep` arg that pipes through shell. The agent stopped reading logs and started probing: `grep="oom_killed|bad_deploy|db_pool_exhausted|missing_config"`. Whichever pattern returns hits is the answer — investigation tool repurposed as a one-call label classifier.

3. **Path traversal via `service`.** `log_path = incident.workdir / "logs" / f"{service}.log"` — pathlib doesn't sanitize `..`. So `service="../../ep-{other_seed}/logs/web"` reads a different episode's files. If the eval set lives in the same `/tmp/sregym/` root, the agent could peek at it!!!

So I looked online and I ended up glazing through [Zheng et al. 2023 "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685) and [Wang et al. 2026 "LLMs Gaming Verifiers: RLVR Can Lead to Reward Hacking"](https://arxiv.org/abs/2604.15149) I picked up two techniques the field uses to fight this kind of thing:

- **LLM-as-judge** (Zheng 2023): when your rubric term is too fuzzy for `==` — like "did the rationale actually cite evidence, or is this confabulation?" — you delegate the scoring to another LLM. Takeaway: it works (~80% agreement with human raters, on par with inter-human agreement), but the paper is just as important for cataloguing the biases — position bias, verbosity bias, self-enhancement bias when the judge and the agent share a model family. Don't trust judge scores as ground truth; calibrate them.

- **Isomorphic perturbation** (Wang 2026): take a successful trace, perturb a surface detail of the task that shouldn't change the answer (rename the affected service, swap a number), re-score under the perturbed task. If the score holds, the agent was solving the underlying problem. If it drops, the agent was riding a surface feature. Takeaway: rewards alone can't tell you whether high scores were earned or structural — perturbation is what makes the difference visible. The paper formalizes this for inductive reasoning; for SRE I'm using a looser version, same intuition.

So the LLM judge is the lever I pulled against cheat #1, and isomorphic perturbation is what I used to fight cheats #2 and #3. (Now I really want to see an LLM cheating its way through GTA 6 — if it ever comes out.)

BUT, I am obviously to training the weights of a frontier model so implementing isomorphic perturbation doesn't really make sense in this instance. It is still worth recognizing as a technique against reward hacking.

### Results

The results were definitely not half bad. A few things I found. The LLM judge works much better when it's from a different model family than the agent it's judging — ideally a different family altogether. Opus judging Opus is biased; Opus judging Sonnet is much better. GPT 5.3 judging Opus 4.6 is better still.

## Phase 5: Token-Level Trajectories

So I have implemented traces, but that is essentially usesless during RL training. Traces are for my own debugging. Now I need token-level trajectories that can be useful during training. Now it is time to swap inference with an open weight server with vLLM or SGLang.
