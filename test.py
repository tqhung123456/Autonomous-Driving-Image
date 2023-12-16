import PIL
from stable_baselines3 import PPO, TD3

from env import CarlaEnvFusion

with CarlaEnvFusion() as carla_env:
    # Load the trained agent
    model = PPO.load("checkpoints/v1/model_14", carla_env)

    obs, _ = carla_env.reset()
    try:
        for _ in range(1000000):
            try:
                action, _ = model.predict(obs.copy(), deterministic=True)
            except Exception as e:
                print(e)
            obs, _, terminated, truncated, _ = carla_env.step(action)
            if terminated or truncated:
                obs, _ = carla_env.reset()
            # Save image
            PIL.Image.fromarray(carla_env.img_tracking).save(f"test/{carla_env.frame}.jpg")
    except Exception as e:
        print(e)
