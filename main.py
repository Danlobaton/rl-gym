from agent import run_episode
from gym import IncidentEnv

if __name__ == "__main__":
    env = IncidentEnv()
    for seed in range(10):
        reward = run_episode(env, seed)
        print(f"seed={seed} reward={reward:+.2f}")
