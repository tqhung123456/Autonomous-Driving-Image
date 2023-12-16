import time

import numpy as np
from tqdm import tqdm

from env import CarlaEnvFusion

if __name__ == "__main__":
    with CarlaEnvFusion(debug=True) as carla_env:
        num = 100000
        # data = np.empty((num, 3, carla_env.img_height, carla_env.img_width), dtype=np.uint8)
        upper = np.empty((num, carla_env.img_height, carla_env.img_width, 1), dtype=np.float64)
        lower = np.empty((num, carla_env.img_height, carla_env.img_width, 1), dtype=np.float64)
        info = np.empty((num, 3), dtype=np.float32)
        action = np.empty((num, 2), dtype=np.float32)
        reward = np.empty((num, 1), dtype=np.float32)
        done = np.empty((num, 1), dtype=bool)
        try:
            obs, _ = carla_env.reset()
            # Get the start time
            start_time = time.time()
            # while time.time() - start_time < 1e10:
            for i in tqdm(range(num)):
                upper[i] = obs["upper"]
                lower[i] = obs["lower"]
                info[i] = obs["info"]
                vehicle_control = carla_env.ego_vehicle.get_control()
                action[i] = np.array([vehicle_control.throttle, vehicle_control.steer])
                # obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                obs, rw, terminated, truncated, _ = carla_env.step(
                    # [(random.random() ** 0.3) * 2 - 1, random.uniform(-1, 1)]
                    # [1.0, random.uniform(-1, 1)]
                    [1.0, 0.0]
                )
                reward[i] = rw
                done[i] = terminated * (1 - truncated)
                # cv2.imshow("image", obs["rbg"])
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
                # print(env.imu_data[0])
                # print(SPEED_THRESHOLD * obs[2])
                # print(obs.shape)
                # print(obs[1])
                # print(reward)
                if terminated or truncated:
                    obs, _ = carla_env.reset()
                    # print(len(env.spawn_points))
                # time.sleep(0.05)
            # print(f"Time elapsed: {time.time() - start_time}s")
        except Exception as e:
            print(e)
        # np.savez_compressed("images.npz", data=data)
        np.save("upper.npy", upper)
        np.save("lower.npy", lower)
        np.save("info.npy", info)
        np.save("action.npy", action)
        np.save("reward.npy", reward)
        np.save("done.npy", done)
