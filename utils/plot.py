import matplotlib.pyplot as plt

def plot_rewards(r):
    plt.plot(r)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
