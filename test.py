from stable_baselines3 import TD3

from env import CarlaEnvFusion

with CarlaEnvFusion() as carla_env:
    # Load the trained agent
    model = TD3.load("checkpoints/pretrained", carla_env)

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
    except Exception as e:
        print(e)
