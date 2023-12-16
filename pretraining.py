import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm
from env import CarlaEnvFusion, CarlaEnvDummy
from create_model import ComplexMultiInputPolicy, PPOComplexMultiInputPolicy


NO_CUDA = False
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
LOG_INTERVAL = 100
TEST_BATCH_SIZE = 64


def batch_indices(indices, batch_size=BATCH_SIZE):
    """Yield successive batches of indices along with the batch number."""
    batch_num = 0
    for start_idx in range(0, len(indices), batch_size):
        end_idx = min(start_idx + batch_size, len(indices))
        yield batch_num, indices[start_idx:end_idx]
        batch_num += 1


if __name__ == "__main__":
    with CarlaEnvDummy(debug=False) as carla_env:
        # student = TD3(
        #     ComplexMultiInputPolicy,
        #     carla_env,
        #     learning_rate=0.001,
        #     buffer_size=1,
        #     learning_starts=6000,
        #     # batch_size=512,
        #     # action_noise=normal_action_noise,
        #     optimize_memory_usage=False,
        #     policy_delay=5,
        #     # policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     device="cuda",
        # )

        student = PPO(
            PPOComplexMultiInputPolicy,
            carla_env,
            learning_rate=0.001,
            verbose=1,
            device="cuda",
        )
        student = PPO.load("ppo_pretrained")

        # Load the data
        upper = np.load("upper.npy")
        lower = np.load("lower.npy")
        info = np.load("info.npy")
        action = np.load("action.npy")

        # Change image to CxHxW
        upper = np.transpose(upper, (0, 3, 1, 2))
        lower = np.transpose(lower, (0, 3, 1, 2))

        # Change to tensor
        upper = th.from_numpy(upper)
        lower = th.from_numpy(lower)
        info = th.from_numpy(info)
        action = th.from_numpy(action)

        # Combine into dict
        dataset = [
            {"upper": up, "lower": low, "info": inf}
            for up, low, inf in zip(upper, lower, info)
        ]

        # Create an array of indices
        data_length = len(dataset)
        indices = np.arange(data_length)

        # Split dataset
        train_size = int(0.8 * data_length)
        test_size = data_length - train_size
        train_dataset, test_dataset = random_split(indices, [train_size, test_size])

        use_cuda = not NO_CUDA and th.cuda.is_available()
        th.manual_seed(SEED)
        device = th.device("cuda" if use_cuda else "cpu")
        criterion = nn.MSELoss()

        # Extract initial policy
        actor = student.policy.to(device)

        # Define an Optimizer and a learning rate schedule.
        actor_optimizer = optim.AdamW(actor.parameters(), lr=LEARNING_RATE)

        total_batches = (len(train_dataset) + BATCH_SIZE - 1) // BATCH_SIZE

        actor.train()
        for epoch in range(1, EPOCHS + 1):
            train_loader = batch_indices(train_dataset)
            for batch_idx, batch in train_loader:
                # Extract the tensors for each key and stack them along the first dimension (batch dimension)
                batch_upper = th.stack(
                    [dataset[idx]["upper"].to(device) for idx in batch]
                )
                batch_lower = th.stack(
                    [dataset[idx]["lower"].to(device) for idx in batch]
                )
                batch_info = th.stack(
                    [dataset[idx]["info"].to(device) for idx in batch]
                )
                target_action = th.stack([action[idx].to(device) for idx in batch])

                # Now create a dictionary with these batched tensors
                data = {"upper": batch_upper, "lower": batch_lower, "info": batch_info}

                predict = actor(data)
                # actor_loss = criterion(predict, target_action)
                actor_loss = criterion(predict[0], target_action)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if batch_idx % LOG_INTERVAL == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tActor Loss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            data_length,
                            100.0 * batch_idx / total_batches,
                            actor_loss.item(),
                        )
                    )

        def test(actor, device, test_loader):
            actor.eval()
            actor_loss = 0
            with th.no_grad():
                for (data,) in test_loader:
                    target_action = th.from_numpy(
                        np.array([1, 0], dtype=np.float32)
                    ).to(device)
                    data = data.to(device)

                    # Repeat target_action to match the batch size
                    target_action = target_action.repeat(data.size(0), 1)

                    action = actor(data)
                    actor_loss = criterion(action, target_action)

                    actor_loss /= len(test_loader.dataset)
            print("\nTest set: Average actor loss: {:.4f}\n".format(actor_loss.item()))

        # Implant the trained policy network back into the RL student agent
        student.policy = actor
        student.save("ppo_pretrained")
