import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 4x4 GridWorld
# States 0..15 laid out row-major:
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
# Start: S0 (top-left), Goal/Terminal: S15 (bottom-right)
# Actions: 0=Up, 1=Down, 2=Left, 3=Right
# Reward: -1 for every step until the goal is reached (0 at goal, absorbing)
# ---------------------------------------------------------

N = 4
NUM_STATES = N * N
GOAL = NUM_STATES - 1
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
GAMMA = 0.9


def rc(s):
    return divmod(s, N)


def to_s(r, c):
    return r * N + c


def step(s, a):
    """Deterministic transition. Returns (next_state, reward)."""
    if s == GOAL:
        return s, 0
    r, c = rc(s)
    if a == 0:
        r2, c2 = r - 1, c
    elif a == 1:
        r2, c2 = r + 1, c
    elif a == 2:
        r2, c2 = r, c - 1
    elif a == 3:
        r2, c2 = r, c + 1
    if r2 < 0 or r2 >= N or c2 < 0 or c2 >= N:
        r2, c2 = r, c
    s2 = to_s(r2, c2)
    reward = 0 if s2 == GOAL else -1
    return s2, reward


# ---------------------------------------------------------
# Task 2: Policy Evaluation for a given policy (uniform random policy)
# ---------------------------------------------------------
def policy_evaluation(policy, theta=1e-4, max_iter=1000, gamma=GAMMA):
    V = np.zeros(NUM_STATES)
    history = []
    for _ in range(1, max_iter + 1):
        delta = 0
        V_new = np.copy(V)
        for s in range(NUM_STATES):
            if s == GOAL:
                V_new[s] = 0
                continue
            v = 0
            for a in ACTIONS:
                s2, r = step(s, a)
                v += policy[s][a] * (r + gamma * V[s2])
            delta = max(delta, abs(v - V[s]))
            V_new[s] = v
        V = V_new
        history.append(delta)
        if delta < theta:
            break
    return V, history


random_policy = {s: {a: 0.25 for a in ACTIONS} for s in range(NUM_STATES)}
V_eval, delta_history = policy_evaluation(random_policy)

print("=== Task 2: Policy Evaluation (uniform random policy) ===")
for i, d in enumerate(delta_history[:10], 1):
    conv = "Converged" if d < 1e-4 else "Not converged"
    print(f"Iter {i}: delta={d:.5f}  {conv}")
print(f"...converged after {len(delta_history)} iterations\n")
print("State-value function under random policy (V_pi), reshaped to grid:")
print(np.round(V_eval.reshape(N, N), 2), "\n")


# ---------------------------------------------------------
# Task 3: Value Iteration
# ---------------------------------------------------------
def value_iteration(theta=1e-4, max_iter=1000, gamma=GAMMA):
    V = np.zeros(NUM_STATES)
    history = []
    for _ in range(1, max_iter + 1):
        delta = 0
        V_new = np.copy(V)
        for s in range(NUM_STATES):
            if s == GOAL:
                V_new[s] = 0
                continue
            q_values = []
            for a in ACTIONS:
                s2, r = step(s, a)
                q_values.append(r + gamma * V[s2])
            best = max(q_values)
            delta = max(delta, abs(best - V[s]))
            V_new[s] = best
        V = V_new
        history.append(delta)
        if delta < theta:
            break

    optimal_policy = {}
    for s in range(NUM_STATES):
        if s == GOAL:
            optimal_policy[s] = None
            continue
        q_values = []
        for a in ACTIONS:
            s2, r = step(s, a)
            q_values.append(r + gamma * V[s2])
        optimal_policy[s] = int(np.argmax(q_values))
    return V, optimal_policy, history


V_star, optimal_policy, vi_history = value_iteration()

print("=== Task 3: Value Iteration ===")
print(f"Converged after {len(vi_history)} iterations\n")
print("Optimal state-value function V*, reshaped to grid:")
print(np.round(V_star.reshape(N, N), 2), "\n")
print("State | Optimal Action | State Value")
for s in range(NUM_STATES):
    a = optimal_policy[s]
    aname = ACTION_NAMES[a] if a is not None else "GOAL"
    print(f"S{s:<4} | {aname:<15} | {V_star[s]:.3f}")
print()

arrow_map = {0: "^", 1: "v", 2: "<", 3: ">"}
grid_arrows = ["G" if s == GOAL else arrow_map[optimal_policy[s]] for s in range(NUM_STATES)]
print("Optimal policy (arrow grid):")
for r in range(N):
    print(" ".join(grid_arrows[r * N:(r + 1) * N]))
print()


def greedy_policy_from_V(V, gamma=GAMMA):
    policy = {}
    for s in range(NUM_STATES):
        if s == GOAL:
            policy[s] = None
            continue
        q_values = []
        for a in ACTIONS:
            s2, r = step(s, a)
            q_values.append(r + gamma * V[s2])
        policy[s] = int(np.argmax(q_values))
    return policy


evaluated_policy = greedy_policy_from_V(V_eval)


# ---------------------------------------------------------
# Task 4: Run policies from S0 and compare paths
# ---------------------------------------------------------
def run_policy(policy_fn, start=0, max_steps=100):
    s = start
    path = [s]
    total_reward = 0
    for _ in range(max_steps):
        if s == GOAL:
            break
        a = policy_fn(s)
        s2, r = step(s, a)
        total_reward += r
        path.append(s2)
        s = s2
    reached = s == GOAL
    return path, len(path) - 1, total_reward, reached


def random_policy_fn(s):
    return random.choice(ACTIONS)


def evaluated_policy_fn(s):
    return evaluated_policy[s]


def optimal_policy_fn(s):
    return optimal_policy[s]


print("=== Task 4: Optimal Path Analysis (start = S0) ===")
results = {}
for name, fn in [("Random Policy", random_policy_fn),
                 ("Evaluated Policy", evaluated_policy_fn),
                 ("Optimal Policy", optimal_policy_fn)]:
    path, length, total_r, reached = run_policy(fn)
    results[name] = (path, length, total_r, reached)
    path_str = " -> ".join(f"S{p}" for p in path)
    print(f"{name}:")
    print(f"  Path: {path_str}")
    print(f"  Path Length: {length}, Total Reward: {total_r}, Goal Reached: {'Yes' if reached else 'No'}")
print()