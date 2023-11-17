import time

import numpy as np
from tqdm import tqdm

from env import CarlaEnvContinuous

if __name__ == "__main__":
    with CarlaEnvContinuous(debug=True) as carla_env:
        num_images = 100000
        data = np.empty((num_images, carla_env.img_height, carla_env.img_width, 3), dtype=np.uint8)
        try:
            carla_env.reset()
            # Get the start time
            start_time = time.time()
            # while time.time() - start_time < 1e10:
            for i in tqdm(range(num_images)):
                # obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                obs, reward, terminated, truncated, _ = carla_env.step(
                    # [(random.random() ** 0.3) * 2 - 1, random.uniform(-1, 1)]
                    # [1.0, random.uniform(-1, 1)]
                    [1.0, 0.0]
                )
                # cv2.imshow("image", obs["rbg"])
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
                # print(env.imu_data[0])
                # print(SPEED_THRESHOLD * obs[2])
                # print(obs.shape)
                # print(obs[1])
                # print(reward)
                data[i] = obs["rgb"]
                if terminated or truncated:
                    carla_env.reset()
                    # print(len(env.spawn_points))
                # time.sleep(0.05)
            # print(f"Time elapsed: {time.time() - start_time}s")
        except Exception as e:
            print(e)
        np.savez_compressed("data.npz", data=data)
