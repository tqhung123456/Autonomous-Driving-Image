import os
import time
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.td3.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

# import create_model
from env import CarlaEnvFusion


class SaveOnIntervalCallback(BaseCallback):
    def __init__(self, save_interval: int, save_path: str, verbose=0):
        super(SaveOnIntervalCallback, self).__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path

    def _on_step(self) -> bool:
        # This method will be called by the model at each call to `model.learn()`.
        if self.n_calls % self.save_interval == 0:
            self.model.save(
                os.path.join(self.save_path, f"model_{datetime.now().hour}")
            )
        return True


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        """
        Returns value between `initial_value` to 0 (included) at `progress_remaining` = 1
        to linearly decreasing value at `progress_remaining` = 0
        """
        return initial_value * progress_remaining

    return func


def linear_schedule_with_min(
    initial_value: float,
    min_value: float = 1e-4,
    keep_fraction=0.5,
    decay_fraction: float = 0.5,
):
    def func(progress_remaining: float) -> float:
        """
        For `decay_fraction` of the training, it returns a value between
        `initial_value` and `min_value` that linearly decreases.
        For the remaining part of the training, it returns `min_value`.
        """
        if progress_remaining > (1 - keep_fraction):
            return initial_value
        elif progress_remaining > (1 - decay_fraction):
            return initial_value - (initial_value - min_value) * (
                (1 - keep_fraction - progress_remaining) / decay_fraction
            )
        else:
            return min_value

    return func


def stepped_schedule(initial_value: float, min_lr: float = 1e-4):
    def func(progress_remaining: float) -> float:
        """
        Decays the learning rate based on the progress_remaining:
        1.0 - 0.9: initial_value
        0.9 - 0.8: decay by 20% of the difference between initial_value and min_lr
        0.8 - 0.7: decay by 40% of the difference
        ...
        0.5 - 0.0: use min_lr
        ...
        """
        if 1.0 >= progress_remaining > 0.9:
            return initial_value
        elif 0.9 >= progress_remaining > 0.8:
            return initial_value - 0.2 * (initial_value - min_lr)
        elif 0.8 >= progress_remaining > 0.7:
            return initial_value - 0.4 * (initial_value - min_lr)
        elif 0.7 >= progress_remaining > 0.6:
            return initial_value - 0.6 * (initial_value - min_lr)
        elif 0.6 >= progress_remaining > 0.5:
            return initial_value - 0.8 * (initial_value - min_lr)
        else:
            return min_lr

    return func


class TimeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TimeCallback, self).__init__(verbose)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        done = self.locals.get("done", False)
        if done:  # This checks if the episode is done
            end_time = time.time()
            episode_duration = end_time - self.start_time
            print(f"Duration of episode: {episode_duration:.2f} seconds")
            self.start_time = time.time()
        return True


if __name__ == "__main__":
    with CarlaEnvFusion() as carla_env:
        # # Check compatibility
        # check_env(carla_env, warn=True)

        mean = [0, 0]  # mean values for throttle/brake and steer
        sigma = [0.1, 0.05]  # standard deviations for throttle/brake and steer
        normal_action_noise = NormalActionNoise(mean=mean, sigma=sigma)

        theta = 0.15
        sigma_ou = 0.2
        orn_action_noise = OrnsteinUhlenbeckActionNoise(
            mean=mean, sigma=sigma_ou, theta=theta
        )

        # Register the custom network
        policy_kwargs = dict(
            net_arch=dict(qf=[128, 256, 128, 64], pi=[128, 256, 128, 64])
        )

        initial_learning_rate = 1e-4
        # lr_schedule = linear_schedule_with_min(
        #     initial_learning_rate, min_value=1e-5, keep_fraction=0.1, decay_fraction=0.2
        # )
        lr_schedule = stepped_schedule(initial_learning_rate, min_lr=1e-5)

        # # Define the TD3 model
        # model = TD3(
        #     MultiInputPolicy,
        #     carla_env,
        #     learning_rate=lr_schedule,
        #     # buffer_size=1000000,
        #     learning_starts=6000,
        #     # batch_size=512,
        #     # action_noise=normal_action_noise,
        #     optimize_memory_usage=False,
        #     policy_delay=5,
        #     # policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     device="cuda",
        # )

        # # Define the TD3 model
        # model = TD3(
        #     "TD3DensePolicy",
        #     carla_env,
        #     learning_rate=lr_schedule,
        #     buffer_size=10000000,
        #     learning_starts=6000,
        #     batch_size=512,
        #     # action_noise=normal_action_noise,
        #     optimize_memory_usage=False,
        #     policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     device="cuda",
        # )

        # model = TD3.load(
        #     "checkpoints/baseline",
        #     carla_env,
        #     learning_rate=lr_schedule,
        #     buffer_size=100000,
        #     learning_starts=600,
        #     # batch_size=16,
        #     # action_noise=normal_action_noise,
        #     optimize_memory_usage=False,
        #     policy_delay=2,
        #     # policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     device="cuda",
        # )

        model = PPO.load(
            "checkpoints/ppo_baseline",
            carla_env,
            learning_rate=lr_schedule,
            # buffer_size=100000,
            # learning_starts=600,
            # batch_size=16,
            # action_noise=normal_action_noise,
            # optimize_memory_usage=False,
            # policy_delay=2,
            # policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda",
        )

        save_interval = 1000
        save_path = "checkpoints"

        checkpoint_callback = SaveOnIntervalCallback(save_interval, save_path)

        # Train the model
        try:
            model.learn(
                total_timesteps=4000000,
                callback=checkpoint_callback,
                # log_interval=1,
                progress_bar=True,
            )
        except Exception as e:
            print(e)
