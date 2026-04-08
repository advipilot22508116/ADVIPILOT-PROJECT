from env.career_env import CareerEnv
from agent.dqn_agent import DQNAgent
from utils.plot import plot_rewards

env = CareerEnv()
agent = DQNAgent(7, 4)

rewards = []

for e in range(200):
    state = env.reset()
    total = 0

    while True:
        action = agent.act(state)
        ns, r, done, _ = env.step(action)

        agent.remember(state, action, r, ns, done)
        agent.replay()

        state = ns
        total += r

        if done:
            print(f"Episode {e+1}: {round(total,2)}")
            rewards.append(total)
            break

plot_rewards(rewards)