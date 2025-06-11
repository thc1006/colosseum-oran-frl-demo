import matplotlib.pyplot as plt

def plot_training_results(all_round_avg_rewards, all_round_avg_losses):
    """Plot average rewards and losses per communication round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(range(1, len(all_round_avg_rewards) + 1), all_round_avg_rewards, marker='o', linestyle='-', color='b')
    ax1.set_title('Average Reward per Communication Round', fontsize=16)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.grid(True)

    ax2.plot(range(1, len(all_round_avg_losses) + 1), all_round_avg_losses, marker='o', linestyle='-', color='r')
    ax2.set_title('Average Loss per Communication Round', fontsize=16)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Average Loss', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()