import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm


class ExpertDataSet(Dataset):
    def __init__(
        self,
        expert_observations,
        expert_actions,
        expert_rewards,
        expert_dones,
        expert_next_observations,
    ):
        self.observations = expert_observations
        self.actions = expert_actions
        self.rewards = expert_rewards
        self.dones = expert_dones
        self.next_observations = expert_next_observations

    def __getitem__(self, index):
        return (
            self.observations[index],
            self.actions[index],
            self.rewards[index],
            self.dones[index],
            self.next_observations[index],
        )

    def __len__(self):
        return len(self.dones)


def train(
    student,
    batch_size=64,
    epochs=1000,
    log_interval=100,
    no_cuda=False,
    seed=42,
) -> None:
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 12, "pin_memory": True} if use_cuda else {}

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    student.policy.to(device)

    # Switch to train mode (this affects batch norm / dropout)
    student.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    # student._update_learning_rate([student.actor.optimizer, student.critic.optimizer])

    for epoch in range(1, epochs + 1):
        for batch_idx, (
            observations,
            actions,
            rewards,
            dones,
            next_observations,
        ) in enumerate(train_loader):
            observations, actions, rewards, dones, next_observations = (
                observations.to(device),
                actions.to(device),
                rewards.to(device),
                dones.to(device),
                next_observations.to(device),
            )

            with th.no_grad():
                # Select action according to policy and add clipped noise
                next_actions = student.actor_target(next_observations)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(
                    student.critic_target(next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = rewards + (1 - dones) * student.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = student.critic(observations, actions)

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)

            # Optimize the critics
            student.critic.optimizer.zero_grad()
            critic_loss.backward()
            student.critic.optimizer.step()

            # Delayed policy updates
            if student._n_updates % student.policy_delay == 0:
                # Compute actor loss
                actor_loss = -student.critic.q1_forward(
                    observations, student.actor(observations)
                ).mean()

                # Optimize the actor
                student.actor.optimizer.zero_grad()
                actor_loss.backward()
                student.actor.optimizer.step()

                polyak_update(
                    student.critic.parameters(),
                    student.critic_target.parameters(),
                    student.tau,
                )
                polyak_update(
                    student.actor.parameters(),
                    student.actor_target.parameters(),
                    student.tau,
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    student.critic_batch_norm_stats,
                    student.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    student.actor_batch_norm_stats,
                    student.actor_batch_norm_stats_target,
                    1.0,
                )

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tActor Loss: {:.6f}\tCritic Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(dones),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        actor_loss.item(),
                        critic_loss.item(),
                    )
                )


if __name__ == "__main__":
    env_id = "Pendulum-v1"
    env = gym.make(env_id)
    ppo_expert = TD3.load("ppo_expert", env)
    student = TD3("MlpPolicy", env_id, verbose=1, learning_rate=1e-8, device="cpu")

    # Load the data
    data = np.load("expert_data.npz")

    # Access the saved arrays using their keys
    expert_observations = data["expert_observations"]
    expert_actions = data["expert_actions"]
    expert_rewards = data["expert_rewards"]
    expert_dones = data["expert_dones"]
    expert_next_observations = data["expert_next_observations"]

    expert_dataset = ExpertDataSet(
        expert_observations,
        expert_actions,
        expert_rewards,
        expert_dones,
        expert_next_observations,
    )

    train_size = int(1 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )

    mean_reward, std_reward = evaluate_policy(student, env, n_eval_episodes=10)

    train(
        student,
        batch_size=64,
        epochs=100,
        log_interval=100,
        no_cuda=True,
        seed=42,
    )
    student.save("student")

    new_mean_reward, new_std_reward = evaluate_policy(student, env, n_eval_episodes=10)

    print(f"Mean reward = {mean_reward} +/- {std_reward}")
    print(f"Mean reward = {new_mean_reward} +/- {new_std_reward}")
