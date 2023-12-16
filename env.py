import math
import random
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt

import carla
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces

sem = threading.Semaphore(1)
events = [threading.Event() for _ in range(6)]
imu_event = threading.Event()

SECTIONS = 18
PART = 2
MAX_DISTANCE = 50  # same as radar range
SPEED_NORMALIZATION = 100
OPTIMAL_SPEED = 60  # km/h
SPEED_LIMIT = 80  # km/h
SPEED_LOWER_BOUND = 20  # km/h
LIDAR_RANGE = 20  # meters
LIDAR_CHANNELS = 64

DELTA_SECONDS = 0.05


class CarlaEnvDummy(gymnasium.Env):
    metadata = {"render_modes": ["true", "false"], "render_fps": 20}

    def __init__(self, debug=False, render_mode=True, num_npc=20):
        # Set image size
        self.img_width = 64
        self.img_height = 64

        # Additional information
        self.ego_vehicle = None
        self.frame = 0
        self.actors = []
        self.img_captured = None
        self.collision = []
        self.lane_invasion = []
        self.lidar_data = None
        self.radar_data_dict = {}
        self.imu_data = None
        self.npc = []

        # Define speed thresholds
        self.optimal_speed = OPTIMAL_SPEED  # km/h
        self.speed_limit = SPEED_LIMIT  # km/h
        self.speed_lower_bound = SPEED_LOWER_BOUND  # km/h
        self.speed_lower_slope = 1 / self.speed_lower_bound
        self.speed_upper_slope = 1 / (self.optimal_speed - self.speed_lower_bound)

        # Define reward and penalty factors
        self.speed_reward_factor = 1.0
        self.overspeed_penalty_factor = 5.0
        self.steering_penalty_factor = 2.0

        # Observations are dictionaries with the sensor data
        self.observation_space = spaces.Dict(
            {
                "upper": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, self.img_height, self.img_width),
                    dtype=np.float64,
                ),
                "lower": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, self.img_height, self.img_width),
                    dtype=np.float64,
                ),
                "info": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # self.observation_space = spaces.Box(
        #     0, 255, shape=(3, self.img_height, self.img_width), dtype=np.uint8
        # )

        # low = np.full(39, 0.0)
        # low[0] = -1.0
        # low[1] = -1.0
        # high = np.full(39, 1.0)
        # high[2] = 3.0
        # self.observation_space = spaces.Box(
        #     low=low, high=high, shape=(39,), dtype=np.float32
        # )

        # Action corresponding to throtle, steer, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        # # Get the current time
        # now = datetime.now()
        # with open("log.txt", "a") as file:
        #     file.write(f"Reset environment at {now.hour}:{now.minute}\n")

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the data
        self.frame = 0
        self.images = []
        self.img_captured = None
        self.collision = []
        self.lane_invasion = []
        self.lidar_data = None
        self.radar_data_dict = {}
        self.imu_data = None

        result = self._set_up_env()
        while not result:
            for _ in range(10):
                self.world.tick()
            result = self._set_up_env()

        for _ in range(6):
            self.world.tick()

        # Wait for image to be captured
        while self.img_captured is None:
            time.sleep(0.1)
            continue

        # # Wait for imu data
        # while self.imu_data is None:
        #     time.sleep(0.1)
        #     continue

        # # Wait for lidar data
        # while self.lidar_data is None:
        #     pass

        # Get initial observation
        # observation = self._get_obs()

        self._follow_agent()

        return self.img_captured, {}

    def step(self, action):
        # print(action)
        # self._follow_agent()
        # Set vehicle control
        if action[0] > 0:
            control = carla.VehicleControl(
                throttle=float(action[0]), steer=float(action[1]), brake=0.0
            )
        else:
            control = carla.VehicleControl(
                throttle=0.0, steer=float(action[1]), brake=float(-action[0])
            )
        self.ego_vehicle.apply_control(control)

        # Tick the world
        self._tick()

        # Calculate the reward
        terminated = False
        truncated = False

        obs = self._get_obs()

        if len(self.collision) != 0 or len(self.lane_invasion) != 0:
            terminated = True
            reward = -100.0
            print("Collision!")
            return self.img_captured, reward, terminated, truncated, {}

        if self.ego_vehicle.get_location().distance(self.goal_location) < 10:
            terminated = True
            reward = 100.0
            print("Goal reached!")
            # self._follow_agent()
            # self.goal_location = self._set_goal(location=self.goal_location)
            return self.img_captured, reward, terminated, truncated, {}

        speed = SPEED_NORMALIZATION * obs[2]
        # reward = float(action[0] - abs(action[1]) - abs(obs[0]))
        # angle_to_goal = obs[0]
        # reward += 1 - abs(angle_to_goal)
        reward = self._calculate_reward(speed)

        # if speed >= self.speed_limit:
        #     reward = -self.overspeed_penalty_factor * (
        #         (speed - self.speed_limit) / self.speed_limit
        #     )
        # else:
        #     reward = float(action[0] - 2 * abs(action[1]) - abs(obs[0]))
        # if speed <= self.optimal_speed:

        # if self.lane_invasion:
        #     print("Lane invasion")
        #     reward -= 1
        #     self.lane_invasion = None

        if self.frame >= 600:
            truncated = True
            print("Time out!")
            return self.img_captured, reward, terminated, truncated, {}

        return self.img_captured, reward, terminated, truncated, {}

    def _set_up_env(self):
        # self._destroy()

        # # Setup ego vehicle
        # vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        # vehicle_bp.set_attribute("color", "0,0,0")
        # spawn_points = self.ego_spawn_points.copy()
        # chosen_spawn_point = spawn_points.pop(random.randint(0, len(spawn_points) - 1))
        # # chosen_spawn_point = spawn_points.pop(random.randint(345, 352))
        # self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, chosen_spawn_point)
        # while self.ego_vehicle is None:
        #     chosen_spawn_point = spawn_points.pop(
        #         random.randint(0, len(spawn_points) - 1)
        #     )
        #     self.ego_vehicle = self.world.try_spawn_actor(
        #         vehicle_bp, chosen_spawn_point
        #     )
        # if self.ego_vehicle is None:
        #     return False

        # # Setup RGB camera
        # camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        # camera_bp.set_attribute("image_size_x", f"{self.img_width}")
        # camera_bp.set_attribute("image_size_y", f"{self.img_height}")
        # camera_bp.set_attribute("fov", "100")
        # camera_transform = carla.Transform(
        #     carla.Location(x=2.8, z=2.3), carla.Rotation(pitch=-50)
        # )
        # self.camera = self.world.spawn_actor(
        #     camera_bp, camera_transform, attach_to=self.ego_vehicle
        # )
        # self.camera.listen(self._process_image)
        # # self.camera = None

        # # Setup collision and lane invasion sensors
        # collision_bp = self.blueprint_library.find("sensor.other.collision")
        # self.collision_sensor = self.world.spawn_actor(
        #     collision_bp, carla.Transform(), attach_to=self.ego_vehicle
        # )
        # self.collision_sensor.listen(self._on_collision)

        # lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        # self.lane_invasion_sensor = self.world.spawn_actor(
        #     lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
        # )
        # self.lane_invasion_sensor.listen(self._on_lane_invasion)

        # # # Attach LiDAR sensor to the ego vehicle
        # # lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        # # # lidar_bp.set_attribute('range', '100')  # 100 meters range
        # # # lidar_bp.set_attribute("channels", "32")  # 32 channels
        # # # lidar_bp.set_attribute("rotation_frequency", "10")  # 10 Hz
        # # # lidar_bp.set_attribute("upper_fov", "0")  # Upper field of view
        # # # lidar_bp.set_attribute("lower_fov", "-30")  # Lower field of view
        # # # lidar_bp.set_attribute("points_per_second", "100000")
        # # lidar_bp.set_attribute("dropoff_general_rate", "0.0")  # No points dropoff
        # # lidar_bp.set_attribute("dropoff_intensity_limit", "0.0")
        # # lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
        # # lidar_bp.set_attribute("horizontal_fov", "180.0")
        # # self.lidar_sensor = self.world.spawn_actor(
        # #     lidar_bp,
        # #     carla.Transform(carla.Location(x=0, z=2.4)),
        # #     attach_to=self.ego_vehicle,
        # # )
        # # self.lidar_sensor.listen(self._process_lidar)
        # # self.lidar_sensor = None

        # # # Attach radars
        # # self.rad_cam = []
        # # self.rad_num = 6
        # # self.rad_section = SECTIONS // self.rad_num
        # # for i in range(self.rad_num):
        # #     rad_bp = self.world.get_blueprint_library().find("sensor.other.radar")
        # #     rad_bp.set_attribute("horizontal_fov", str(30))
        # #     rad_bp.set_attribute("vertical_fov", str(30))
        # #     rad_bp.set_attribute("range", str(50))
        # #     rad_location = carla.Location(x=2.0, z=1.0)
        # #     rad_rotation = carla.Rotation(pitch=0, yaw=(-75 + 30 * i))
        # #     rad_transform = carla.Transform(rad_location, rad_rotation)
        # #     rad_ego = self.world.spawn_actor(
        # #         rad_bp,
        # #         rad_transform,
        # #         attach_to=self.ego_vehicle,
        # #         attachment_type=carla.AttachmentType.Rigid,
        # #     )
        # #     rad_ego.listen(
        # #         lambda radar_data, idx=i: self._process_radar(radar_data, idx)
        # #     )
        # #     self.rad_cam.append(rad_ego)
        # self.rad_cam = []

        # # # Setup IMU sensor
        # # imu_bp = self.blueprint_library.find("sensor.other.imu")
        # # self.imu_sensor = self.world.spawn_actor(
        # #     imu_bp, carla.Transform(), attach_to=self.ego_vehicle
        # # )
        # # self.imu_sensor.listen(self._process_imu)
        # self.imu_sensor = None

        # # Save all actors
        # self.actors = [
        #     self.ego_vehicle,
        #     self.camera,
        #     self.collision_sensor,
        #     self.lane_invasion_sensor,
        #     self.imu_sensor,
        # ]
        # self.actors.extend(self.rad_cam)

        spawn_points = self.ego_spawn_points.copy()
        chosen_spawn_point = spawn_points.pop(random.randint(0, len(spawn_points) - 1))
        # chosen_spawn_point = spawn_points.pop(random.randint(345, 352))
        if self.ego_vehicle is None:
            vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
            vehicle_bp.set_attribute("color", "0,0,0")
            self.ego_vehicle = self.world.try_spawn_actor(
                vehicle_bp, chosen_spawn_point
            )
            while self.ego_vehicle is None:
                chosen_spawn_point = spawn_points.pop(
                    random.randint(0, len(spawn_points) - 1)
                )
                self.ego_vehicle = self.world.try_spawn_actor(
                    vehicle_bp, chosen_spawn_point
                )
            if self.ego_vehicle is None:
                return False

            # Setup RGB camera
            camera_bp = self.blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", f"{self.img_width}")
            camera_bp.set_attribute("image_size_y", f"{self.img_height}")
            camera_bp.set_attribute("fov", "100")
            camera_transform = carla.Transform(
                carla.Location(x=2.8, z=2.3), carla.Rotation(pitch=-50)
            )
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.ego_vehicle
            )
            self.camera.listen(self._process_image)
            # self.camera = None

            # Setup collision and lane invasion sensors
            collision_bp = self.blueprint_library.find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.ego_vehicle
            )
            self.collision_sensor.listen(self._on_collision)

            lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
            self.lane_invasion_sensor = self.world.spawn_actor(
                lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
            )
            self.lane_invasion_sensor.listen(self._on_lane_invasion)

            # Save all actors
            self.actors = [
                self.ego_vehicle,
                self.camera,
                self.collision_sensor,
                self.lane_invasion_sensor,
            ]
        else:
            try:
                self.ego_vehicle.set_target_angular_velocity(carla.Vector3D())
                self.ego_vehicle.set_target_velocity(carla.Vector3D())
                self.ego_vehicle.set_transform(chosen_spawn_point)
            except Exception as e:
                print(e)
                return False

        # If in debug mode, enable autopilot
        if self.debug:
            if self.ego_vehicle is not None:
                self.ego_vehicle.set_autopilot(True)

        # Tick the world
        self.world.tick()

        # # Get a list of spawn points and filter for points near the ego vehicle
        # nearby_spawn_points = [
        #     sp
        #     for sp in spawn_points
        #     if sp.location.distance(self.ego_vehicle.get_location()) < 50
        # ]

        # sampled_spawn_points = random.sample(
        #     nearby_spawn_points, len(nearby_spawn_points) // 2
        # )

        # # Spawn NPC vehicles
        # self.num_npc = num_npc
        # for sp in sampled_spawn_points:
        #     npc = self._spawn_npc(sp)
        #     self.npc.append(npc)

        # Setup goal location
        self.goal_location = self._set_goal()

        return True

    def _set_goal(self, location=None, spacing=1.0):
        if location:
            initial_waypoint = self.map.get_waypoint(location)
        else:
            ego_location = self.ego_vehicle.get_location()
            initial_waypoint = self.map.get_waypoint(ego_location)

        distance_to_travel = 40
        distance_traveled = 0
        current_waypoint = initial_waypoint

        while distance_traveled < distance_to_travel:
            # Get the next waypoint along the road
            next_waypoints = current_waypoint.next(
                spacing
            )  # Adjust the distance to the next waypoint as needed
            if not next_waypoints:
                # No more waypoints, end the loop
                break
            next_waypoint = next_waypoints[0]

            # Update the distance traveled
            distance_traveled += spacing

            # Update the current waypoint for the next iteration
            current_waypoint = next_waypoint

        # waypoints = self._get_waypoints_across_lanes(current_waypoint)

        # # Initialize sums
        # sum_x = 0
        # sum_y = 0
        # sum_z = 0

        # # Sum up the coordinates
        # for waypoint in waypoints:
        #     sum_x += waypoint.transform.location.x
        #     sum_y += waypoint.transform.location.y
        #     sum_z += waypoint.transform.location.z

        # # Compute the average
        # center_x = sum_x / len(waypoints)
        # center_y = sum_y / len(waypoints)
        # center_z = sum_z / len(waypoints)

        # # The center location
        # goal_location = carla.Location(x=center_x, y=center_y, z=center_z)

        goal_location = current_waypoint.transform.location

        # self.world.debug.draw_point(
        #     goal_location,
        #     size=0.1,
        #     life_time=1,
        #     persistent_lines=False,
        #     color=carla.Color(255, 0, 0),
        # )

        return goal_location

    def _get_obs(self):
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        waypoint = self.map.get_waypoint(ego_location)

        # Speed
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = 3.6 * math.sqrt(
            ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2
        )  # speed in km/h
        ego_speed = ego_speed / SPEED_NORMALIZATION  # normalize to [0, 1]

        # Angle to the Goal
        ego_orientation = ego_rotation.yaw  # in degrees
        # print(f"Ego orientation: {ego_orientation}")

        # Calculate the direction to the goal in degrees
        direction_to_goal_deg = math.degrees(
            math.atan2(
                self.goal_location.y - ego_location.y,
                self.goal_location.x - ego_location.x,
            )
        )
        # print(f"Direction to goal: {direction_to_goal_deg}")
        # Compute the angle difference
        angle_to_goal = direction_to_goal_deg - ego_orientation
        # Normalize the angle difference to the range [-180, 180]
        angle_to_goal = (angle_to_goal + 180) % 360 - 180
        # Normalize further to the range [-1, 1]
        angle_to_goal_normalized = angle_to_goal / 180.0
        # print(f"Angle to goal: {angle_to_goal_normalized}")

        # Angle to the road
        road_orientation = waypoint.transform.rotation.yaw
        # Calculate the angle difference
        angle_to_road = ego_orientation - road_orientation
        # Normalize to range [-180, 180]
        angle_to_road %= 360
        if angle_to_road > 180:
            angle_to_road -= 360
        # Normalize further to the range [-1, 1]
        angle_to_road_normalized = angle_to_road / 180.0
        # print(angle_to_road_normalized)

        return np.array(
            [
                angle_to_goal_normalized,
                angle_to_road_normalized,
                ego_speed,
            ],
            dtype=np.float32,
        )

        # return self.lidar_data

        # for event in events:
        #     event.wait()

        # combined_data = np.full((SECTIONS, PART), MAX_DISTANCE)
        # all_points = []

        # for idx, radar_data in self.radar_data_dict.items():
        #     points = []
        #     for detect in radar_data:
        #         azi = math.degrees(detect.azimuth)
        #         alt = math.degrees(detect.altitude)
        #         section = int(
        #             (azi + 15) % self.rad_section
        #         )  # +15 to shift from [-15, 15] to [0, 30]
        #         section = section + self.rad_section * idx
        #         # part_idx = 1 if alt >= 0 else 0  # 1 for up, 0 for down
        #         part_idx = int(PART * ((alt + 15) / 30))
        #         distance = detect.depth / MAX_DISTANCE
        #         if distance < combined_data[section, part_idx]:
        #             combined_data[section, part_idx] = distance
        #             points.append(detect)
        #     all_points.append((radar_data.transform, points))
        # combined_data = combined_data.flatten() / 50

        # # self._draw_radar_points(all_points)

        # for event in events:
        #     event.clear()

        # return np.concatenate(
        #     [
        #         [angle_to_goal_normalized, angle_to_road_normalized, ego_speed],
        #         combined_data,
        #     ],
        #     dtype=np.float32,
        # )

    def _calculate_reward(self, speed):
        # Calculate speed-based reward/penalty
        if speed <= self.speed_lower_bound:
            speed_reward = (speed - self.speed_lower_bound) * self.speed_lower_slope
        elif speed <= self.optimal_speed:
            speed_reward = (speed - self.speed_lower_bound) * self.speed_upper_slope
        elif speed <= self.speed_limit:
            speed_reward = self.speed_reward_factor
        else:
            speed_reward = (
                -speed * self.speed_lower_slope + self.overspeed_penalty_factor
            )
        # if speed <= self.optimal_speed:
        #     speed_reward = self.speed_reward_factor * (speed / self.optimal_speed)
        # elif self.optimal_speed < speed <= self.speed_limit:
        #     # Linearly decrease reward from optimal_speed to speed_limit
        #     speed_reward = self.speed_reward_factor
        # else:
        #     # Apply penalty for speeding above speed_limit
        #     speed_reward = -self.overspeed_penalty_factor * (
        #         (speed - self.speed_limit) / self.speed_limit
        #     )

        # # Calculate steering penalty (increase with speed)
        # steering_penalty = (
        #     self.steering_penalty_factor * abs(steering) * (speed / self.speed_limit)
        # )

        # # Combine rewards and penalties
        # total_reward = speed_reward - steering_penalty

        return speed_reward

    # Function to calculate the angle between two vectors
    def _calculate_angle(v1, v2):
        dot_product = v1.x * v2.x + v1.y * v2.y
        v1_length = math.sqrt(v1.x**2 + v1.y**2)
        v2_length = math.sqrt(v2.x**2 + v2.y**2)
        angle = math.acos(dot_product / (v1_length * v2_length))
        return angle

    def _get_waypoints_across_lanes(self, waypoint):
        """
        Get waypoints across different lanes given an initial waypoint.

        Args:
        - waypoint: The initial CARLA waypoint.

        Returns:
        - A list of waypoints across different lanes.
        """
        waypoints_across_lanes = []

        # Add the initial waypoint to the list
        waypoints_across_lanes.append(waypoint)

        # Get waypoints on the left lanes
        left_waypoint = waypoint.get_left_lane()
        while (
            left_waypoint is not None
            and left_waypoint.lane_type == carla.LaneType.Driving
        ):
            waypoints_across_lanes.append(left_waypoint)
            left_waypoint = left_waypoint.get_left_lane()

        # Get waypoints on the right lanes
        right_waypoint = waypoint.get_right_lane()
        while (
            right_waypoint is not None
            and right_waypoint.lane_type == carla.LaneType.Driving
        ):
            waypoints_across_lanes.append(right_waypoint)
            right_waypoint = right_waypoint.get_right_lane()

        return waypoints_across_lanes

    def _tick(self):
        self.world.tick()
        self.frame += 1
        # print(f"Time: {self.frame*self.fixed_delta_seconds}s")

    def _process_image(self, image):
        img_captured = np.array(image.raw_data, dtype=np.dtype("uint8"))
        img_captured = np.reshape(img_captured, (self.img_height, self.img_width, 4))
        img_captured = img_captured[:, :, :3]
        img_captured = img_captured[:, :, ::-1]
        # Transpose the image to have channels as the first dimension
        # Changing from (height, width, num_channels) to (num_channels, height, width)
        self.img_captured = np.transpose(img_captured, (2, 0, 1))

    def _process_lidar(self, data):
        # Convert the raw data to numpy array
        self.lidar_data = np.array(data.raw_data, dtype=np.dtype("f4"))
        self.lidar_data = np.reshape(
            self.lidar_data, (int(self.lidar_data.shape[0] / 4), 4)
        )

        self.lidar_data = self.lidar_data[:, :3]

    def _process_radar(self, data, idx):
        self.radar_data_dict[idx] = data
        events[idx].set()

    def _process_imu(self, data):
        self.imu_data = (data.accelerometer, data.gyroscope)
        imu_event.set()

    def _draw_radar_points(self, all_points):
        for current_trans, points in all_points:
            if points:
                current_rot = current_trans.rotation
                for point in points:
                    azi = math.degrees(point.azimuth)
                    alt = math.degrees(point.altitude)
                    # The 0.25 adjusts a bit the distance so the dots can
                    # be properly seen
                    fw_vec = carla.Vector3D(x=point.depth - 0.25)
                    carla.Transform(
                        carla.Location(),
                        carla.Rotation(
                            pitch=current_rot.pitch + alt,
                            yaw=current_rot.yaw + azi,
                            roll=current_rot.roll,
                        ),
                    ).transform(fw_vec)

                    self.world.debug.draw_point(
                        current_trans.location + fw_vec,
                        size=0.075,
                        life_time=0.06,
                        persistent_lines=False,
                        color=carla.Color(255, 0, 0),
                    )

    def _on_collision(self, event):
        self.collision.append(event)

    def _on_lane_invasion(self, event):
        self.lane_invasion.append(event)

    def _follow_agent(self):
        # Get the spectator from the world
        spectator = self.world.get_spectator()

        # Get the car's current transform
        car_transform = self.ego_vehicle.get_transform()

        # Modify the transform to move the spectator
        car_transform.location.z = 75
        car_transform.rotation.pitch = -60

        # Set the spectator's transform
        spectator.set_transform(car_transform)
        # spectator.set_transform(self.camera.get_transform())

    def _spawn_npc(self):
        sampled_spawn_points = random.sample(
            self.spawn_points, len(self.spawn_points) // 2
        )

        # Spawn NPC vehicles
        for sp in sampled_spawn_points:
            # Get a random blueprint.
            blueprint = random.choice(
                self.world.get_blueprint_library().filter("vehicle.*")
            )

            # Some vehicles do not support autopilot, so we need to check and possibly choose again.
            while (
                blueprint.has_attribute("number_of_wheels")
                and int(blueprint.get_attribute("number_of_wheels")) < 4
            ):
                blueprint = random.choice(
                    self.world.get_blueprint_library().filter("vehicle.*")
                )

            # Spawn the vehicle
            vehicle = self.world.try_spawn_actor(blueprint, sp)
            if vehicle:
                vehicle.set_autopilot(True)

            self.npc.append(vehicle)

    def _destroy(self):
        for actor in self.actors:
            if actor is not None:
                if actor.is_alive:
                    actor.destroy()

        # for npc in self.npc:
        #     if npc is not None:
        #         if npc.is_alive:
        #             npc.destroy()

        self.actors = []
        # self.npc = []

    def _close(self):
        # # Save images and controls
        # data = np.array(self.images)
        # np.save("images.npy", data)
        # # np.savez_compressed('images.npz', array=data)
        # data = np.array(self.controls)
        # np.save("controls.npy", data)
        # # np.savez_compressed('controls.npz', array=data)

        # # Ensure synchronous mode is turned off
        # self.traffic_manager.set_synchronous_mode(False)
        # settings = self.world.get_settings()
        # settings.synchronous_mode = False
        # self.world.apply_settings(settings)

        # for actor in self.actors:
        #     if actor is not None:
        #         if actor.is_alive:
        #             actor.destroy()

        # for npc in self.npc:
        #     if npc is not None:
        #         if npc.is_alive:
        #             npc.destroy()

        self.actors = []
        self.npc = []

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()


class CarlaEnvFusion(gymnasium.Env):
    metadata = {"render_modes": ["true", "false"], "render_fps": 20}

    def __init__(self, debug=False, render_mode=True, num_npc=20):
        # Connect to the server
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(30.0)
        self.client.load_world("Town04")

        # Get the world
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Set up spawn points
        self.spawn_points = self.map.get_spawn_points()
        self.ego_spawn_points = self.spawn_points.copy()
        # Read excluded spawn points from file
        excluded_spawn_points = []
        with open("excluded_spawn_points.txt", "r") as file:
            for line in file:
                # Convert each line to an integer and append to the list
                excluded_spawn_points.append(int(line.strip()))
        excluded_spawn_points.reverse()  # Reverse the list to remove from the end
        for excluded_spawn_point in excluded_spawn_points:
            self.ego_spawn_points.pop(excluded_spawn_point)

        # Initialize Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DELTA_SECONDS  # FPS
        settings.no_rendering_mode = not render_mode
        self.world.apply_settings(settings)
        self.fixed_delta_seconds = DELTA_SECONDS

        # Set debug mode
        self.debug = debug

        # Set image size
        self.img_width = 64
        self.img_height = 64

        # Additional information
        self.ego_vehicle = None
        self.frame = 0
        self.actors = []
        self.img_captured = None
        self.collision = []
        self.lane_invasion = []
        self.lidar_data = None
        self.radar_data_dict = {}
        self.imu_data = None
        self.img_tracking = None
        self.npc = []
        self.xbins = np.linspace(-LIDAR_RANGE, LIDAR_RANGE, 128 + 1)
        self.ybins = np.linspace(-LIDAR_RANGE, LIDAR_RANGE, 128 + 1)

        # Set actors
        self.blueprint_library = self.world.get_blueprint_library()

        self._set_up_env()
        # self._spawn_npc()

        # Define speed thresholds
        self.optimal_speed = OPTIMAL_SPEED  # km/h
        self.speed_limit = SPEED_LIMIT  # km/h
        self.speed_lower_bound = SPEED_LOWER_BOUND  # km/h
        self.speed_lower_slope = 1 / self.speed_lower_bound
        self.speed_upper_slope = 1 / (self.optimal_speed - self.speed_lower_bound)

        # Define reward and penalty factors
        self.speed_reward_factor = 1.0
        self.overspeed_penalty_factor = 5.0
        self.steering_penalty_factor = 2.0

        # Observations are dictionaries with the sensor data
        self.observation_space = spaces.Dict(
            {
                "upper": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, self.img_height, self.img_width),
                    dtype=np.float64,
                ),
                "lower": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1, self.img_height, self.img_width),
                    dtype=np.float64,
                ),
                "info": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # self.observation_space = spaces.Box(
        #     0, 255, shape=(3, self.img_height, self.img_width), dtype=np.uint8
        # )

        # low = np.full(39, 0.0)
        # low[0] = -1.0
        # low[1] = -1.0
        # high = np.full(39, 1.0)
        # high[2] = 3.0
        # self.observation_space = spaces.Box(
        #     low=low, high=high, shape=(39,), dtype=np.float32
        # )

        # Action corresponding to throtle, steer, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        # # Get the current time
        # now = datetime.now()
        # with open("log.txt", "a") as file:
        #     file.write(f"Reset environment at {now.hour}:{now.minute}\n")

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the data
        self.frame = 0
        self.images = []
        self.img_captured = None
        self.collision = []
        self.lane_invasion = []
        self.lidar_data = None
        self.radar_data_dict = {}
        self.imu_data = None
        self.img_tracking = None

        result = self._set_up_env()
        while not result:
            for _ in range(10):
                self.world.tick()
            result = self._set_up_env()

        for _ in range(6):
            self.world.tick()

        # # Wait for image to be captured
        # while self.img_captured is None:
        #     time.sleep(0.1)
        #     continue

        # # Wait for imu data
        # while self.imu_data is None:
        #     time.sleep(0.1)
        #     continue

        # Wait for lidar data
        while self.upper_lidar_data is None or self.lower_lidar_data is None:
            time.sleep(0.1)
            continue

        # Get initial observation
        observation = self._get_obs()

        self._follow_agent()

        return observation, {}

    def step(self, action):
        # print(action)
        # self._follow_agent()
        # Set vehicle control
        if action[0] > 0:
            control = carla.VehicleControl(
                throttle=float(action[0]), steer=float(action[1]), brake=0.0
            )
        else:
            control = carla.VehicleControl(
                throttle=0.0, steer=float(action[1]), brake=float(-action[0])
            )
        self.ego_vehicle.apply_control(control)

        # Tick the world
        self._tick()

        # Calculate the reward
        terminated = False
        truncated = False

        obs = self._get_obs()

        if len(self.collision) != 0 or len(self.lane_invasion) != 0:
            terminated = True
            reward = -100.0
            print("Collision!")
            return obs, reward, terminated, truncated, {}

        if self.ego_vehicle.get_location().distance(self.goal_location) < 10:
            # terminated = True
            reward = 100.0
            # print("Goal reached!")
            self._follow_agent()
            self.goal_location = self._set_goal(location=self.goal_location)
            return obs, reward, terminated, truncated, {}

        speed = SPEED_NORMALIZATION * obs["info"][2]
        # print(speed)
        # reward = float(action[0] - abs(action[1]) - abs(obs[0]))
        # angle_to_goal = obs[0]
        # reward += 1 - abs(angle_to_goal)
        reward = self._calculate_reward(speed)

        # if speed >= self.speed_limit:
        #     reward = -self.overspeed_penalty_factor * (
        #         (speed - self.speed_limit) / self.speed_limit
        #     )
        # else:
        #     reward = float(action[0] - 2 * abs(action[1]) - abs(obs[0]))
        # if speed <= self.optimal_speed:

        # if self.lane_invasion:
        #     print("Lane invasion")
        #     reward -= 1
        #     self.lane_invasion = None

        if self.frame >= 6000:
            truncated = True
            print("Time out!")
            return obs, reward, terminated, truncated, {}

        return obs, reward, terminated, truncated, {}

    def _set_up_env(self):
        spawn_points = self.ego_spawn_points.copy()
        chosen_spawn_point = spawn_points.pop(random.randint(0, len(spawn_points) - 1))
        if self.ego_vehicle is None:
            vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
            vehicle_bp.set_attribute("color", "0,0,0")
            self.ego_vehicle = self.world.try_spawn_actor(
                vehicle_bp, chosen_spawn_point
            )
            while self.ego_vehicle is None:
                chosen_spawn_point = spawn_points.pop(
                    random.randint(0, len(spawn_points) - 1)
                )
                self.ego_vehicle = self.world.try_spawn_actor(
                    vehicle_bp, chosen_spawn_point
                )
            if self.ego_vehicle is None:
                return False

            # # Setup RGB camera
            # camera_bp = self.blueprint_library.find("sensor.camera.rgb")
            # camera_bp.set_attribute("image_size_x", f"{self.img_width}")
            # camera_bp.set_attribute("image_size_y", f"{self.img_height}")
            # camera_bp.set_attribute("fov", "100")
            # camera_transform = carla.Transform(
            #     carla.Location(x=1.3, y=0.0, z=2.3),
            #     carla.Rotation(pitch=-50),
            # )
            # self.camera = self.world.spawn_actor(
            #     camera_bp, camera_transform, attach_to=self.ego_vehicle
            # )
            # self.camera.listen(self._process_image)
            self.camera = None

            # Setup tracking camera
            tracking_camera_bp = self.blueprint_library.find("sensor.camera.rgb")
            tracking_camera_bp.set_attribute("image_size_x", "640")
            tracking_camera_bp.set_attribute("image_size_y", "640")
            tracking_camera_bp.set_attribute("fov", "100")
            tracking_camera_transform = carla.Transform(
                carla.Location(x=-10.0, y=0.0, z=10.0),
                carla.Rotation(pitch=-35.0, yaw=0.0, roll=0.0),
            )
            self.tracking_camera = self.world.spawn_actor(
                tracking_camera_bp,
                tracking_camera_transform,
                attach_to=self.ego_vehicle,
            )
            self.tracking_camera.listen(self._process_tracking_image)

            # Setup lidar sensor
            lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("range", f"{LIDAR_RANGE}")
            # lidar_bp.set_attribute("channels", f"{LIDAR_CHANNELS}")
            lidar_bp.set_attribute(
                "rotation_frequency", f"{1/self.fixed_delta_seconds}"
            )
            lidar_transform = carla.Transform(
                carla.Location(x=1.3, y=0.0, z=2.3),
                carla.Rotation(roll=0.0, pitch=0.0, yaw=180.0),
            )
            self.lidar = self.world.spawn_actor(
                lidar_bp, lidar_transform, attach_to=self.ego_vehicle
            )
            self.lidar.listen(self._process_lidar)

            # Setup collision and lane invasion sensors
            collision_bp = self.blueprint_library.find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.ego_vehicle
            )
            self.collision_sensor.listen(self._on_collision)

            lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
            self.lane_invasion_sensor = self.world.spawn_actor(
                lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
            )
            self.lane_invasion_sensor.listen(self._on_lane_invasion)

            # Save all actors
            self.actors = [
                self.ego_vehicle,
                self.camera,
                self.lidar,
                self.collision_sensor,
                self.lane_invasion_sensor,
            ]
        else:
            try:
                self.ego_vehicle.set_target_angular_velocity(carla.Vector3D())
                self.ego_vehicle.set_target_velocity(carla.Vector3D())
                self.ego_vehicle.set_transform(chosen_spawn_point)
            except Exception as e:
                print(e)
                return False

        # car_bp = self.blueprint_library.find("vehicle.tesla.model3")

        # ego_waypoint = self.map.get_waypoint(chosen_spawn_point.location)

        # # Find the waypoint in the next lane
        # next_lane_waypoint = (
        #     ego_waypoint.get_left_lane()
        #     if ego_waypoint.lane_type == carla.LaneType.Driving
        #     else ego_waypoint.get_right_lane()
        # )

        # # Adjust the transform for the new vehicle
        # spawn_transform = carla.Transform(
        #     next_lane_waypoint.transform.location,
        #     next_lane_waypoint.transform.rotation,
        # )
        # spawn_transform.location.z = 0.3

        # # Spawn the vehicle
        # self.vehicle = self.world.spawn_actor(car_bp, spawn_transform)

        # If in debug mode, enable autopilot
        if self.debug:
            if self.ego_vehicle is not None:
                self.ego_vehicle.set_autopilot(True)

        # Tick the world
        self.world.tick()

        # # Get a list of spawn points and filter for points near the ego vehicle
        # nearby_spawn_points = [
        #     sp
        #     for sp in spawn_points
        #     if sp.location.distance(self.ego_vehicle.get_location()) < 50
        # ]

        # sampled_spawn_points = random.sample(
        #     nearby_spawn_points, len(nearby_spawn_points) // 2
        # )

        # # Spawn NPC vehicles
        # self.num_npc = num_npc
        # for sp in sampled_spawn_points:
        #     npc = self._spawn_npc(sp)
        #     self.npc.append(npc)

        # Setup goal location
        self.goal_location = self._set_goal()

        return True

    def _set_goal(self, location=None, spacing=1.0):
        if location:
            initial_waypoint = self.map.get_waypoint(location)
        else:
            ego_location = self.ego_vehicle.get_location()
            initial_waypoint = self.map.get_waypoint(ego_location)

        distance_to_travel = 40
        distance_traveled = 0
        current_waypoint = initial_waypoint

        while distance_traveled < distance_to_travel:
            # Get the next waypoint along the road
            next_waypoints = current_waypoint.next(
                spacing
            )  # Adjust the distance to the next waypoint as needed
            if not next_waypoints:
                # No more waypoints, end the loop
                break
            next_waypoint = next_waypoints[0]

            # Update the distance traveled
            distance_traveled += spacing

            # Update the current waypoint for the next iteration
            current_waypoint = next_waypoint

        # waypoints = self._get_waypoints_across_lanes(current_waypoint)

        # # Initialize sums
        # sum_x = 0
        # sum_y = 0
        # sum_z = 0

        # # Sum up the coordinates
        # for waypoint in waypoints:
        #     sum_x += waypoint.transform.location.x
        #     sum_y += waypoint.transform.location.y
        #     sum_z += waypoint.transform.location.z

        # # Compute the average
        # center_x = sum_x / len(waypoints)
        # center_y = sum_y / len(waypoints)
        # center_z = sum_z / len(waypoints)

        # # The center location
        # goal_location = carla.Location(x=center_x, y=center_y, z=center_z)

        goal_location = current_waypoint.transform.location

        # self.world.debug.draw_point(
        #     goal_location,
        #     size=0.1,
        #     life_time=1,
        #     persistent_lines=False,
        #     color=carla.Color(255, 0, 0),
        # )

        return goal_location

    def _get_obs(self):
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        waypoint = self.map.get_waypoint(ego_location)

        # Speed
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = 3.6 * math.sqrt(
            ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2
        )  # speed in km/h
        ego_speed = ego_speed / SPEED_NORMALIZATION  # normalize to [0, 1]

        # Angle to the Goal
        ego_orientation = ego_rotation.yaw  # in degrees
        # print(f"Ego orientation: {ego_orientation}")

        # Calculate the direction to the goal in degrees
        direction_to_goal_deg = math.degrees(
            math.atan2(
                self.goal_location.y - ego_location.y,
                self.goal_location.x - ego_location.x,
            )
        )
        # print(f"Direction to goal: {direction_to_goal_deg}")
        # Compute the angle difference
        angle_to_goal = direction_to_goal_deg - ego_orientation
        # Normalize the angle difference to the range [-180, 180]
        angle_to_goal = (angle_to_goal + 180) % 360 - 180
        # Normalize further to the range [-1, 1]
        angle_to_goal_normalized = angle_to_goal / 180.0
        # print(f"Angle to goal: {angle_to_goal_normalized}")

        # Angle to the road
        road_orientation = waypoint.transform.rotation.yaw
        # Calculate the angle difference
        angle_to_road = ego_orientation - road_orientation
        # Normalize to range [-180, 180]
        angle_to_road %= 360
        if angle_to_road > 180:
            angle_to_road -= 360
        # Normalize further to the range [-1, 1]
        angle_to_road_normalized = angle_to_road / 180.0
        # print(angle_to_road_normalized)

        info = np.array(
            [
                angle_to_goal_normalized,
                angle_to_road_normalized,
                ego_speed,
            ],
            dtype=np.float32,
        )

        obs = {
            "upper": self.upper_lidar_data,
            "lower": self.lower_lidar_data,
            "info": info,
        }

        return obs

        # return np.array(
        #     [
        #         angle_to_goal_normalized,
        #         angle_to_road_normalized,
        #         ego_speed,
        #     ],
        #     dtype=np.float32,
        # )

        # return self.lidar_data

        # for event in events:
        #     event.wait()

        # combined_data = np.full((SECTIONS, PART), MAX_DISTANCE)
        # all_points = []

        # for idx, radar_data in self.radar_data_dict.items():
        #     points = []
        #     for detect in radar_data:
        #         azi = math.degrees(detect.azimuth)
        #         alt = math.degrees(detect.altitude)
        #         section = int(
        #             (azi + 15) % self.rad_section
        #         )  # +15 to shift from [-15, 15] to [0, 30]
        #         section = section + self.rad_section * idx
        #         # part_idx = 1 if alt >= 0 else 0  # 1 for up, 0 for down
        #         part_idx = int(PART * ((alt + 15) / 30))
        #         distance = detect.depth / MAX_DISTANCE
        #         if distance < combined_data[section, part_idx]:
        #             combined_data[section, part_idx] = distance
        #             points.append(detect)
        #     all_points.append((radar_data.transform, points))
        # combined_data = combined_data.flatten() / 50

        # # self._draw_radar_points(all_points)

        # for event in events:
        #     event.clear()

        # return np.concatenate(
        #     [
        #         [angle_to_goal_normalized, angle_to_road_normalized, ego_speed],
        #         combined_data,
        #     ],
        #     dtype=np.float32,
        # )

    def _calculate_reward(self, speed):
        # Calculate speed-based reward/penalty
        if speed <= self.speed_lower_bound:
            speed_reward = (speed - self.speed_lower_bound) * self.speed_lower_slope
        elif speed <= self.optimal_speed:
            speed_reward = (speed - self.speed_lower_bound) * self.speed_upper_slope
        elif speed <= self.speed_limit:
            speed_reward = self.speed_reward_factor
        else:
            speed_reward = (
                -speed * self.speed_lower_slope + self.overspeed_penalty_factor
            )
        # if speed <= self.optimal_speed:
        #     speed_reward = self.speed_reward_factor * (speed / self.optimal_speed)
        # elif self.optimal_speed < speed <= self.speed_limit:
        #     # Linearly decrease reward from optimal_speed to speed_limit
        #     speed_reward = self.speed_reward_factor
        # else:
        #     # Apply penalty for speeding above speed_limit
        #     speed_reward = -self.overspeed_penalty_factor * (
        #         (speed - self.speed_limit) / self.speed_limit
        #     )

        # # Calculate steering penalty (increase with speed)
        # steering_penalty = (
        #     self.steering_penalty_factor * abs(steering) * (speed / self.speed_limit)
        # )

        # # Combine rewards and penalties
        # total_reward = speed_reward - steering_penalty

        return speed_reward

    # Function to calculate the angle between two vectors
    def _calculate_angle(v1, v2):
        dot_product = v1.x * v2.x + v1.y * v2.y
        v1_length = math.sqrt(v1.x**2 + v1.y**2)
        v2_length = math.sqrt(v2.x**2 + v2.y**2)
        angle = math.acos(dot_product / (v1_length * v2_length))
        return angle

    def _get_waypoints_across_lanes(self, waypoint):
        """
        Get waypoints across different lanes given an initial waypoint.

        Args:
        - waypoint: The initial CARLA waypoint.

        Returns:
        - A list of waypoints across different lanes.
        """
        waypoints_across_lanes = []

        # Add the initial waypoint to the list
        waypoints_across_lanes.append(waypoint)

        # Get waypoints on the left lanes
        left_waypoint = waypoint.get_left_lane()
        while (
            left_waypoint is not None
            and left_waypoint.lane_type == carla.LaneType.Driving
        ):
            waypoints_across_lanes.append(left_waypoint)
            left_waypoint = left_waypoint.get_left_lane()

        # Get waypoints on the right lanes
        right_waypoint = waypoint.get_right_lane()
        while (
            right_waypoint is not None
            and right_waypoint.lane_type == carla.LaneType.Driving
        ):
            waypoints_across_lanes.append(right_waypoint)
            right_waypoint = right_waypoint.get_right_lane()

        return waypoints_across_lanes

    def _tick(self):
        self.world.tick()
        self.frame += 1
        # print(f"Time: {self.frame*self.fixed_delta_seconds}s")

    def _process_image(self, image):
        img_captured = np.array(image.raw_data, dtype=np.dtype("uint8"))
        img_captured = np.reshape(img_captured, (self.img_height, self.img_width, 4))
        img_captured = img_captured[:, :, :3]
        img_captured = img_captured[:, :, ::-1]
        # Transpose the image to have channels as the first dimension
        # Changing from (height, width, num_channels) to (num_channels, height, width)
        self.img_captured = np.transpose(img_captured, (2, 0, 1))

    def _process_tracking_image(self, image):
        img_tracking = np.array(image.raw_data, dtype=np.dtype("uint8"))
        img_tracking = np.reshape(img_tracking, (640, 640, 4))
        img_tracking = img_tracking[:, :, :3]
        self.img_tracking = img_tracking[:, :, ::-1]

    def _process_lidar(self, data):
        # Convert the raw data to numpy array
        points = np.frombuffer(data.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        # lidar = np.array(points[:, :3])

        # Upper part
        upper_lidar = points[points[:, 2] >= -2]
        upper_lidar = upper_lidar[:, :2]

        # Create a 2D histogram
        upper_hist = np.histogramdd(upper_lidar, bins=[self.xbins, self.ybins])[0]

        # Clipping histogram values
        upper_hist = np.clip(upper_hist, 0, 5)

        # Crop the front center of the histogram
        upper_hist = upper_hist[:64, 32:96]

        # Normalize the histogram
        upper_hist = upper_hist / 5

        # # Scale the histogram to 0-255
        # upper_hist = (upper_hist / 5) * 255

        # # Convert to int8
        # upper_hist = upper_hist.astype(np.int8)

        # Add channel dimension
        self.upper_lidar_data = upper_hist.reshape(1, 64, 64)

        # Lower part
        lower_lidar = points[points[:, 2] < -2]
        lower_lidar = lower_lidar[:, :2]

        # Create a 2D histogram
        lower_hist = np.histogramdd(lower_lidar, bins=[self.xbins, self.ybins])[0]

        # Clipping histogram values
        lower_hist = np.clip(lower_hist, 0, 5)

        # Crop the front center of the histogram
        lower_hist = lower_hist[:64, 32:96]

        # Normalize the histogram
        lower_hist = lower_hist / 5

        # # Scale the histogram to 0-255
        # lower_hist = (lower_hist / 5) * 255

        # # Convert to int8
        # lower_hist = lower_hist.astype(np.int8)

        # Add channel dimension
        self.lower_lidar_data = lower_hist.reshape(1, 64, 64)

    def _process_radar(self, data, idx):
        self.radar_data_dict[idx] = data
        events[idx].set()

    def _process_imu(self, data):
        self.imu_data = (data.accelerometer, data.gyroscope)
        imu_event.set()

    def _draw_radar_points(self, all_points):
        for current_trans, points in all_points:
            if points:
                current_rot = current_trans.rotation
                for point in points:
                    azi = math.degrees(point.azimuth)
                    alt = math.degrees(point.altitude)
                    # The 0.25 adjusts a bit the distance so the dots can
                    # be properly seen
                    fw_vec = carla.Vector3D(x=point.depth - 0.25)
                    carla.Transform(
                        carla.Location(),
                        carla.Rotation(
                            pitch=current_rot.pitch + alt,
                            yaw=current_rot.yaw + azi,
                            roll=current_rot.roll,
                        ),
                    ).transform(fw_vec)

                    self.world.debug.draw_point(
                        current_trans.location + fw_vec,
                        size=0.075,
                        life_time=0.06,
                        persistent_lines=False,
                        color=carla.Color(255, 0, 0),
                    )

    def _on_collision(self, event):
        self.collision.append(event)

    def _on_lane_invasion(self, event):
        pass
        # self.lane_invasion.append(event)

    def _follow_agent(self):
        # Get the spectator from the world
        spectator = self.world.get_spectator()

        # Get the car's current transform
        car_transform = self.ego_vehicle.get_transform()

        # Modify the transform to move the spectator
        car_transform.location.z = 75
        car_transform.rotation.pitch = -60

        # Set the spectator's transform
        spectator.set_transform(car_transform)
        # spectator.set_transform(self.camera.get_transform())

    def _spawn_npc(self):
        sampled_spawn_points = random.sample(
            self.spawn_points, len(self.spawn_points) // 2
        )

        # Spawn NPC vehicles
        for sp in sampled_spawn_points:
            # Get a random blueprint.
            blueprint = random.choice(
                self.world.get_blueprint_library().filter("vehicle.*")
            )

            # Some vehicles do not support autopilot, so we need to check and possibly choose again.
            while (
                blueprint.has_attribute("number_of_wheels")
                and int(blueprint.get_attribute("number_of_wheels")) < 4
            ):
                blueprint = random.choice(
                    self.world.get_blueprint_library().filter("vehicle.*")
                )

            # Spawn the vehicle
            vehicle = self.world.try_spawn_actor(blueprint, sp)
            if vehicle:
                vehicle.set_autopilot(True)

            self.npc.append(vehicle)

    def _destroy(self):
        for actor in self.actors:
            if actor is not None:
                if actor.is_alive:
                    actor.destroy()

        # for npc in self.npc:
        #     if npc is not None:
        #         if npc.is_alive:
        #             npc.destroy()

        self.actors = []
        # self.npc = []

    def _close(self):
        # # Save images and controls
        # data = np.array(self.images)
        # np.save("images.npy", data)
        # # np.savez_compressed('images.npz', array=data)
        # data = np.array(self.controls)
        # np.save("controls.npy", data)
        # # np.savez_compressed('controls.npz', array=data)

        # Ensure synchronous mode is turned off
        self.traffic_manager.set_synchronous_mode(False)
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        for actor in self.actors:
            if actor is not None:
                if actor.is_alive:
                    actor.destroy()

        for npc in self.npc:
            if npc is not None:
                if npc.is_alive:
                    npc.destroy()

        self.actors = []
        self.npc = []

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# class CarlaEnvOnlyImage(gymnasium.Env):
#     metadata = {"render_modes": ["true", "false"], "render_fps": 20}

#     def __init__(self, debug=False, render_mode=True, num_npc=20):
#         # Connect to the server
#         self.client = carla.Client("127.0.0.1", 2000)
#         self.client.set_timeout(30.0)
#         self.client.load_world("Town04")

#         # Get the world
#         self.world = self.client.get_world()
#         self.map = self.world.get_map()

#         # Set up spawn points
#         self.spawn_points = self.map.get_spawn_points()
#         self.ego_spawn_points = self.spawn_points.copy()
#         # Read excluded spawn points from file
#         excluded_spawn_points = []
#         with open("excluded_spawn_points.txt", "r") as file:
#             for line in file:
#                 # Convert each line to an integer and append to the list
#                 excluded_spawn_points.append(int(line.strip()))
#         excluded_spawn_points.reverse()  # Reverse the list to remove from the end
#         for excluded_spawn_point in excluded_spawn_points:
#             self.ego_spawn_points.pop(excluded_spawn_point)

#         # Initialize Traffic Manager
#         self.traffic_manager = self.client.get_trafficmanager(8000)
#         self.traffic_manager.set_synchronous_mode(True)

#         # Set synchronous mode
#         settings = self.world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = DELTA_SECONDS  # FPS
#         settings.no_rendering_mode = not render_mode
#         self.world.apply_settings(settings)
#         self.fixed_delta_seconds = DELTA_SECONDS

#         # Set debug mode
#         self.debug = debug

#         # Set image size
#         self.img_width = 128
#         self.img_height = 128

#         # Additional information
#         self.ego_vehicle = None
#         self.frame = 0
#         self.actors = []
#         self.img_captured = None
#         self.collision = []
#         self.lane_invasion = []
#         self.lidar_data = None
#         self.radar_data_dict = {}
#         self.imu_data = None
#         self.npc = []

#         # Set actors
#         self.blueprint_library = self.world.get_blueprint_library()

#         self._set_up_env()
#         # self._spawn_npc()

#         # Define speed thresholds
#         self.optimal_speed = OPTIMAL_SPEED  # km/h
#         self.speed_limit = SPEED_LIMIT  # km/h
#         self.speed_lower_bound = SPEED_LOWER_BOUND  # km/h
#         self.speed_lower_slope = 1 / self.speed_lower_bound
#         self.speed_upper_slope = 1 / (self.optimal_speed - self.speed_lower_bound)

#         # Define reward and penalty factors
#         self.speed_reward_factor = 1.0
#         self.overspeed_penalty_factor = 5.0
#         self.steering_penalty_factor = 2.0

#         # # Observations are dictionaries with the sensor data
#         # self.observation_space = spaces.Dict(
#         #     {
#         #         "rgb": spaces.Box(
#         #             0, 255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
#         #         ),
#         #         "info": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
#         #     }
#         # )

#         self.observation_space = spaces.Box(
#             0, 255, shape=(3, self.img_height, self.img_width), dtype=np.uint8
#         )

#         # low = np.full(39, 0.0)
#         # low[0] = -1.0
#         # low[1] = -1.0
#         # high = np.full(39, 1.0)
#         # high[2] = 3.0
#         # self.observation_space = spaces.Box(
#         #     low=low, high=high, shape=(39,), dtype=np.float32
#         # )

#         # Action corresponding to throtle, steer, brake
#         self.action_space = spaces.Box(
#             low=np.array([-1.0, -1.0]),
#             high=np.array([1.0, 1.0]),
#             shape=(2,),
#             dtype=np.float32,
#         )

#     def reset(self, seed=None):
#         # # Get the current time
#         # now = datetime.now()
#         # with open("log.txt", "a") as file:
#         #     file.write(f"Reset environment at {now.hour}:{now.minute}\n")

#         # We need the following line to seed self.np_random
#         super().reset(seed=seed)

#         # Reset the data
#         self.frame = 0
#         self.images = []
#         self.img_captured = None
#         self.collision = []
#         self.lane_invasion = []
#         self.lidar_data = None
#         self.radar_data_dict = {}
#         self.imu_data = None

#         result = self._set_up_env()
#         while not result:
#             for _ in range(10):
#                 self.world.tick()
#             result = self._set_up_env()

#         for _ in range(6):
#             self.world.tick()

#         # Wait for image to be captured
#         while self.img_captured is None:
#             time.sleep(0.1)
#             continue

#         # # Wait for imu data
#         # while self.imu_data is None:
#         #     time.sleep(0.1)
#         #     continue

#         # # Wait for lidar data
#         # while self.lidar_data is None:
#         #     pass

#         # Get initial observation
#         # observation = self._get_obs()

#         self._follow_agent()

#         with sem:
#             return self.img_captured, {}

#     def step(self, action):
#         # print(action)
#         # self._follow_agent()
#         # Set vehicle control
#         if action[0] > 0:
#             control = carla.VehicleControl(
#                 throttle=float(action[0]), steer=float(action[1]), brake=0.0
#             )
#         else:
#             control = carla.VehicleControl(
#                 throttle=0.0, steer=float(action[1]), brake=float(-action[0])
#             )
#         self.ego_vehicle.apply_control(control)

#         # Tick the world
#         self._tick()

#         # Calculate the reward
#         terminated = False
#         truncated = False

#         obs = self._get_obs()

#         if len(self.collision) != 0 or len(self.lane_invasion) != 0:
#             terminated = True
#             reward = -100.0
#             print("Collision!")
#             with sem:
#                 return self.img_captured, reward, terminated, truncated, {}

#         if self.ego_vehicle.get_location().distance(self.goal_location) < 10:
#             # terminated = True
#             reward = 100.0
#             print("Goal reached!")
#             self._follow_agent()
#             self.goal_location = self._set_goal(location=self.goal_location)
#             with sem:
#                 return self.img_captured, reward, terminated, truncated, {}

#         speed = SPEED_NORMALIZATION * obs[2]
#         # reward = float(action[0] - abs(action[1]) - abs(obs[0]))
#         # angle_to_goal = obs[0]
#         # reward += 1 - abs(angle_to_goal)
#         reward = self._calculate_reward(speed)

#         # if speed >= self.speed_limit:
#         #     reward = -self.overspeed_penalty_factor * (
#         #         (speed - self.speed_limit) / self.speed_limit
#         #     )
#         # else:
#         #     reward = float(action[0] - 2 * abs(action[1]) - abs(obs[0]))
#         # if speed <= self.optimal_speed:

#         # if self.lane_invasion:
#         #     print("Lane invasion")
#         #     reward -= 1
#         #     self.lane_invasion = None

#         if self.frame >= 6000:
#             truncated = True
#             print("Time out!")
#             with sem:
#                 return self.img_captured, reward, terminated, truncated, {}

#         with sem:
#             return self.img_captured, reward, terminated, truncated, {}

#     def _set_up_env(self):
#         # self._destroy()

#         # # Setup ego vehicle
#         # vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
#         # vehicle_bp.set_attribute("color", "0,0,0")
#         # spawn_points = self.ego_spawn_points.copy()
#         # chosen_spawn_point = spawn_points.pop(random.randint(0, len(spawn_points) - 1))
#         # # chosen_spawn_point = spawn_points.pop(random.randint(345, 352))
#         # self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, chosen_spawn_point)
#         # while self.ego_vehicle is None:
#         #     chosen_spawn_point = spawn_points.pop(
#         #         random.randint(0, len(spawn_points) - 1)
#         #     )
#         #     self.ego_vehicle = self.world.try_spawn_actor(
#         #         vehicle_bp, chosen_spawn_point
#         #     )
#         # if self.ego_vehicle is None:
#         #     return False

#         # # Setup RGB camera
#         # camera_bp = self.blueprint_library.find("sensor.camera.rgb")
#         # camera_bp.set_attribute("image_size_x", f"{self.img_width}")
#         # camera_bp.set_attribute("image_size_y", f"{self.img_height}")
#         # camera_bp.set_attribute("fov", "100")
#         # camera_transform = carla.Transform(
#         #     carla.Location(x=2.8, z=2.3), carla.Rotation(pitch=-50)
#         # )
#         # self.camera = self.world.spawn_actor(
#         #     camera_bp, camera_transform, attach_to=self.ego_vehicle
#         # )
#         # self.camera.listen(self._process_image)
#         # # self.camera = None

#         # # Setup collision and lane invasion sensors
#         # collision_bp = self.blueprint_library.find("sensor.other.collision")
#         # self.collision_sensor = self.world.spawn_actor(
#         #     collision_bp, carla.Transform(), attach_to=self.ego_vehicle
#         # )
#         # self.collision_sensor.listen(self._on_collision)

#         # lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
#         # self.lane_invasion_sensor = self.world.spawn_actor(
#         #     lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
#         # )
#         # self.lane_invasion_sensor.listen(self._on_lane_invasion)

#         # # # Attach LiDAR sensor to the ego vehicle
#         # # lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
#         # # # lidar_bp.set_attribute('range', '100')  # 100 meters range
#         # # # lidar_bp.set_attribute("channels", "32")  # 32 channels
#         # # # lidar_bp.set_attribute("rotation_frequency", "10")  # 10 Hz
#         # # # lidar_bp.set_attribute("upper_fov", "0")  # Upper field of view
#         # # # lidar_bp.set_attribute("lower_fov", "-30")  # Lower field of view
#         # # # lidar_bp.set_attribute("points_per_second", "100000")
#         # # lidar_bp.set_attribute("dropoff_general_rate", "0.0")  # No points dropoff
#         # # lidar_bp.set_attribute("dropoff_intensity_limit", "0.0")
#         # # lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
#         # # lidar_bp.set_attribute("horizontal_fov", "180.0")
#         # # self.lidar_sensor = self.world.spawn_actor(
#         # #     lidar_bp,
#         # #     carla.Transform(carla.Location(x=0, z=2.4)),
#         # #     attach_to=self.ego_vehicle,
#         # # )
#         # # self.lidar_sensor.listen(self._process_lidar)
#         # # self.lidar_sensor = None

#         # # # Attach radars
#         # # self.rad_cam = []
#         # # self.rad_num = 6
#         # # self.rad_section = SECTIONS // self.rad_num
#         # # for i in range(self.rad_num):
#         # #     rad_bp = self.world.get_blueprint_library().find("sensor.other.radar")
#         # #     rad_bp.set_attribute("horizontal_fov", str(30))
#         # #     rad_bp.set_attribute("vertical_fov", str(30))
#         # #     rad_bp.set_attribute("range", str(50))
#         # #     rad_location = carla.Location(x=2.0, z=1.0)
#         # #     rad_rotation = carla.Rotation(pitch=0, yaw=(-75 + 30 * i))
#         # #     rad_transform = carla.Transform(rad_location, rad_rotation)
#         # #     rad_ego = self.world.spawn_actor(
#         # #         rad_bp,
#         # #         rad_transform,
#         # #         attach_to=self.ego_vehicle,
#         # #         attachment_type=carla.AttachmentType.Rigid,
#         # #     )
#         # #     rad_ego.listen(
#         # #         lambda radar_data, idx=i: self._process_radar(radar_data, idx)
#         # #     )
#         # #     self.rad_cam.append(rad_ego)
#         # self.rad_cam = []

#         # # # Setup IMU sensor
#         # # imu_bp = self.blueprint_library.find("sensor.other.imu")
#         # # self.imu_sensor = self.world.spawn_actor(
#         # #     imu_bp, carla.Transform(), attach_to=self.ego_vehicle
#         # # )
#         # # self.imu_sensor.listen(self._process_imu)
#         # self.imu_sensor = None

#         # # Save all actors
#         # self.actors = [
#         #     self.ego_vehicle,
#         #     self.camera,
#         #     self.collision_sensor,
#         #     self.lane_invasion_sensor,
#         #     self.imu_sensor,
#         # ]
#         # self.actors.extend(self.rad_cam)

#         spawn_points = self.ego_spawn_points.copy()
#         chosen_spawn_point = spawn_points.pop(random.randint(0, len(spawn_points) - 1))
#         # chosen_spawn_point = spawn_points.pop(random.randint(345, 352))
#         if self.ego_vehicle is None:
#             vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
#             vehicle_bp.set_attribute("color", "0,0,0")
#             self.ego_vehicle = self.world.try_spawn_actor(
#                 vehicle_bp, chosen_spawn_point
#             )
#             while self.ego_vehicle is None:
#                 chosen_spawn_point = spawn_points.pop(
#                     random.randint(0, len(spawn_points) - 1)
#                 )
#                 self.ego_vehicle = self.world.try_spawn_actor(
#                     vehicle_bp, chosen_spawn_point
#                 )
#             if self.ego_vehicle is None:
#                 return False

#             # Setup RGB camera
#             camera_bp = self.blueprint_library.find("sensor.camera.rgb")
#             camera_bp.set_attribute("image_size_x", f"{self.img_width}")
#             camera_bp.set_attribute("image_size_y", f"{self.img_height}")
#             camera_bp.set_attribute("fov", "100")
#             camera_transform = carla.Transform(
#                 carla.Location(x=2.8, z=2.3), carla.Rotation(pitch=-50)
#             )
#             self.camera = self.world.spawn_actor(
#                 camera_bp, camera_transform, attach_to=self.ego_vehicle
#             )
#             self.camera.listen(self._process_image)
#             # self.camera = None

#             # Setup collision and lane invasion sensors
#             collision_bp = self.blueprint_library.find("sensor.other.collision")
#             self.collision_sensor = self.world.spawn_actor(
#                 collision_bp, carla.Transform(), attach_to=self.ego_vehicle
#             )
#             self.collision_sensor.listen(self._on_collision)

#             lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
#             self.lane_invasion_sensor = self.world.spawn_actor(
#                 lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
#             )
#             self.lane_invasion_sensor.listen(self._on_lane_invasion)

#             # Save all actors
#             self.actors = [
#                 self.ego_vehicle,
#                 self.camera,
#                 self.collision_sensor,
#                 self.lane_invasion_sensor,
#             ]
#         else:
#             try:
#                 self.ego_vehicle.set_target_angular_velocity(carla.Vector3D())
#                 self.ego_vehicle.set_target_velocity(carla.Vector3D())
#                 self.ego_vehicle.set_transform(chosen_spawn_point)
#             except Exception as e:
#                 print(e)
#                 return False

#         # If in debug mode, enable autopilot
#         if self.debug:
#             if self.ego_vehicle is not None:
#                 self.ego_vehicle.set_autopilot(True)

#         # Tick the world
#         self.world.tick()

#         # # Get a list of spawn points and filter for points near the ego vehicle
#         # nearby_spawn_points = [
#         #     sp
#         #     for sp in spawn_points
#         #     if sp.location.distance(self.ego_vehicle.get_location()) < 50
#         # ]

#         # sampled_spawn_points = random.sample(
#         #     nearby_spawn_points, len(nearby_spawn_points) // 2
#         # )

#         # # Spawn NPC vehicles
#         # self.num_npc = num_npc
#         # for sp in sampled_spawn_points:
#         #     npc = self._spawn_npc(sp)
#         #     self.npc.append(npc)

#         # Setup goal location
#         self.goal_location = self._set_goal()

#         return True

#     def _set_goal(self, location=None, spacing=1.0):
#         if location:
#             initial_waypoint = self.map.get_waypoint(location)
#         else:
#             ego_location = self.ego_vehicle.get_location()
#             initial_waypoint = self.map.get_waypoint(ego_location)

#         distance_to_travel = 40
#         distance_traveled = 0
#         current_waypoint = initial_waypoint

#         while distance_traveled < distance_to_travel:
#             # Get the next waypoint along the road
#             next_waypoints = current_waypoint.next(
#                 spacing
#             )  # Adjust the distance to the next waypoint as needed
#             if not next_waypoints:
#                 # No more waypoints, end the loop
#                 break
#             next_waypoint = next_waypoints[0]

#             # Update the distance traveled
#             distance_traveled += spacing

#             # Update the current waypoint for the next iteration
#             current_waypoint = next_waypoint

#         # waypoints = self._get_waypoints_across_lanes(current_waypoint)

#         # # Initialize sums
#         # sum_x = 0
#         # sum_y = 0
#         # sum_z = 0

#         # # Sum up the coordinates
#         # for waypoint in waypoints:
#         #     sum_x += waypoint.transform.location.x
#         #     sum_y += waypoint.transform.location.y
#         #     sum_z += waypoint.transform.location.z

#         # # Compute the average
#         # center_x = sum_x / len(waypoints)
#         # center_y = sum_y / len(waypoints)
#         # center_z = sum_z / len(waypoints)

#         # # The center location
#         # goal_location = carla.Location(x=center_x, y=center_y, z=center_z)

#         goal_location = current_waypoint.transform.location

#         # self.world.debug.draw_point(
#         #     goal_location,
#         #     size=0.1,
#         #     life_time=1,
#         #     persistent_lines=False,
#         #     color=carla.Color(255, 0, 0),
#         # )

#         return goal_location

#     def _get_obs(self):
#         ego_transform = self.ego_vehicle.get_transform()
#         ego_location = ego_transform.location
#         ego_rotation = ego_transform.rotation
#         waypoint = self.map.get_waypoint(ego_location)

#         # Speed
#         ego_velocity = self.ego_vehicle.get_velocity()
#         ego_speed = 3.6 * math.sqrt(
#             ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2
#         )  # speed in km/h
#         ego_speed = ego_speed / SPEED_NORMALIZATION  # normalize to [0, 1]

#         # Angle to the Goal
#         ego_orientation = ego_rotation.yaw  # in degrees
#         # print(f"Ego orientation: {ego_orientation}")

#         # Calculate the direction to the goal in degrees
#         direction_to_goal_deg = math.degrees(
#             math.atan2(
#                 self.goal_location.y - ego_location.y,
#                 self.goal_location.x - ego_location.x,
#             )
#         )
#         # print(f"Direction to goal: {direction_to_goal_deg}")
#         # Compute the angle difference
#         angle_to_goal = direction_to_goal_deg - ego_orientation
#         # Normalize the angle difference to the range [-180, 180]
#         angle_to_goal = (angle_to_goal + 180) % 360 - 180
#         # Normalize further to the range [-1, 1]
#         angle_to_goal_normalized = angle_to_goal / 180.0
#         # print(f"Angle to goal: {angle_to_goal_normalized}")

#         # Angle to the road
#         road_orientation = waypoint.transform.rotation.yaw
#         # Calculate the angle difference
#         angle_to_road = ego_orientation - road_orientation
#         # Normalize to range [-180, 180]
#         angle_to_road %= 360
#         if angle_to_road > 180:
#             angle_to_road -= 360
#         # Normalize further to the range [-1, 1]
#         angle_to_road_normalized = angle_to_road / 180.0
#         # print(angle_to_road_normalized)

#         return np.array(
#             [
#                 angle_to_goal_normalized,
#                 angle_to_road_normalized,
#                 ego_speed,
#             ],
#             dtype=np.float32,
#         )

#         # return self.lidar_data

#         # for event in events:
#         #     event.wait()

#         # combined_data = np.full((SECTIONS, PART), MAX_DISTANCE)
#         # all_points = []

#         # for idx, radar_data in self.radar_data_dict.items():
#         #     points = []
#         #     for detect in radar_data:
#         #         azi = math.degrees(detect.azimuth)
#         #         alt = math.degrees(detect.altitude)
#         #         section = int(
#         #             (azi + 15) % self.rad_section
#         #         )  # +15 to shift from [-15, 15] to [0, 30]
#         #         section = section + self.rad_section * idx
#         #         # part_idx = 1 if alt >= 0 else 0  # 1 for up, 0 for down
#         #         part_idx = int(PART * ((alt + 15) / 30))
#         #         distance = detect.depth / MAX_DISTANCE
#         #         if distance < combined_data[section, part_idx]:
#         #             combined_data[section, part_idx] = distance
#         #             points.append(detect)
#         #     all_points.append((radar_data.transform, points))
#         # combined_data = combined_data.flatten() / 50

#         # # self._draw_radar_points(all_points)

#         # for event in events:
#         #     event.clear()

#         # return np.concatenate(
#         #     [
#         #         [angle_to_goal_normalized, angle_to_road_normalized, ego_speed],
#         #         combined_data,
#         #     ],
#         #     dtype=np.float32,
#         # )

#     def _calculate_reward(self, speed):
#         # Calculate speed-based reward/penalty
#         if speed <= self.speed_lower_bound:
#             speed_reward = (speed - self.speed_lower_bound) * self.speed_lower_slope
#         elif speed <= self.optimal_speed:
#             speed_reward = (speed - self.speed_lower_bound) * self.speed_upper_slope
#         elif speed <= self.speed_limit:
#             speed_reward = self.speed_reward_factor
#         else:
#             speed_reward = (
#                 -speed * self.speed_lower_slope + self.overspeed_penalty_factor
#             )
#         # if speed <= self.optimal_speed:
#         #     speed_reward = self.speed_reward_factor * (speed / self.optimal_speed)
#         # elif self.optimal_speed < speed <= self.speed_limit:
#         #     # Linearly decrease reward from optimal_speed to speed_limit
#         #     speed_reward = self.speed_reward_factor
#         # else:
#         #     # Apply penalty for speeding above speed_limit
#         #     speed_reward = -self.overspeed_penalty_factor * (
#         #         (speed - self.speed_limit) / self.speed_limit
#         #     )

#         # # Calculate steering penalty (increase with speed)
#         # steering_penalty = (
#         #     self.steering_penalty_factor * abs(steering) * (speed / self.speed_limit)
#         # )

#         # # Combine rewards and penalties
#         # total_reward = speed_reward - steering_penalty

#         return speed_reward

#     # Function to calculate the angle between two vectors
#     def _calculate_angle(v1, v2):
#         dot_product = v1.x * v2.x + v1.y * v2.y
#         v1_length = math.sqrt(v1.x**2 + v1.y**2)
#         v2_length = math.sqrt(v2.x**2 + v2.y**2)
#         angle = math.acos(dot_product / (v1_length * v2_length))
#         return angle

#     def _get_waypoints_across_lanes(self, waypoint):
#         """
#         Get waypoints across different lanes given an initial waypoint.

#         Args:
#         - waypoint: The initial CARLA waypoint.

#         Returns:
#         - A list of waypoints across different lanes.
#         """
#         waypoints_across_lanes = []

#         # Add the initial waypoint to the list
#         waypoints_across_lanes.append(waypoint)

#         # Get waypoints on the left lanes
#         left_waypoint = waypoint.get_left_lane()
#         while (
#             left_waypoint is not None
#             and left_waypoint.lane_type == carla.LaneType.Driving
#         ):
#             waypoints_across_lanes.append(left_waypoint)
#             left_waypoint = left_waypoint.get_left_lane()

#         # Get waypoints on the right lanes
#         right_waypoint = waypoint.get_right_lane()
#         while (
#             right_waypoint is not None
#             and right_waypoint.lane_type == carla.LaneType.Driving
#         ):
#             waypoints_across_lanes.append(right_waypoint)
#             right_waypoint = right_waypoint.get_right_lane()

#         return waypoints_across_lanes

#     def _tick(self):
#         self.world.tick()
#         self.frame += 1
#         # print(f"Time: {self.frame*self.fixed_delta_seconds}s")

#     def _process_image(self, image):
#         with sem:
#             self.img_captured = np.array(image.raw_data, dtype=np.dtype("uint8"))
#             self.img_captured = np.reshape(
#                 self.img_captured, (self.img_height, self.img_width, 4)
#             )
#             self.img_captured = self.img_captured[:, :, :3]
#             self.img_captured = self.img_captured[:, :, ::-1]
#             # Transpose the image to have channels as the first dimension
#             # Changing from (height, width, num_channels) to (num_channels, height, width)
#             self.img_captured = np.transpose(self.img_captured, (2, 0, 1))

#     def _process_lidar(self, data):
#         with sem:
#             # Convert the raw data to numpy array
#             self.lidar_data = np.array(data.raw_data, dtype=np.dtype("f4"))
#             self.lidar_data = np.reshape(
#                 self.lidar_data, (int(self.lidar_data.shape[0] / 4), 4)
#             )

#             self.lidar_data = self.lidar_data[:, :3]

#     def _process_radar(self, data, idx):
#         self.radar_data_dict[idx] = data
#         events[idx].set()

#     def _process_imu(self, data):
#         self.imu_data = (data.accelerometer, data.gyroscope)
#         imu_event.set()

#     def _draw_radar_points(self, all_points):
#         for current_trans, points in all_points:
#             if points:
#                 current_rot = current_trans.rotation
#                 for point in points:
#                     azi = math.degrees(point.azimuth)
#                     alt = math.degrees(point.altitude)
#                     # The 0.25 adjusts a bit the distance so the dots can
#                     # be properly seen
#                     fw_vec = carla.Vector3D(x=point.depth - 0.25)
#                     carla.Transform(
#                         carla.Location(),
#                         carla.Rotation(
#                             pitch=current_rot.pitch + alt,
#                             yaw=current_rot.yaw + azi,
#                             roll=current_rot.roll,
#                         ),
#                     ).transform(fw_vec)

#                     self.world.debug.draw_point(
#                         current_trans.location + fw_vec,
#                         size=0.075,
#                         life_time=0.06,
#                         persistent_lines=False,
#                         color=carla.Color(255, 0, 0),
#                     )

#     def _on_collision(self, event):
#         self.collision.append(event)

#     def _on_lane_invasion(self, event):
#         self.lane_invasion.append(event)

#     def _follow_agent(self):
#         # Get the spectator from the world
#         spectator = self.world.get_spectator()

#         # Get the car's current transform
#         car_transform = self.ego_vehicle.get_transform()

#         # Modify the transform to move the spectator
#         car_transform.location.z = 75
#         car_transform.rotation.pitch = -60

#         # Set the spectator's transform
#         spectator.set_transform(car_transform)
#         # spectator.set_transform(self.camera.get_transform())

#     def _spawn_npc(self):
#         sampled_spawn_points = random.sample(
#             self.spawn_points, len(self.spawn_points) // 2
#         )

#         # Spawn NPC vehicles
#         for sp in sampled_spawn_points:
#             # Get a random blueprint.
#             blueprint = random.choice(
#                 self.world.get_blueprint_library().filter("vehicle.*")
#             )

#             # Some vehicles do not support autopilot, so we need to check and possibly choose again.
#             while (
#                 blueprint.has_attribute("number_of_wheels")
#                 and int(blueprint.get_attribute("number_of_wheels")) < 4
#             ):
#                 blueprint = random.choice(
#                     self.world.get_blueprint_library().filter("vehicle.*")
#                 )

#             # Spawn the vehicle
#             vehicle = self.world.try_spawn_actor(blueprint, sp)
#             if vehicle:
#                 vehicle.set_autopilot(True)

#             self.npc.append(vehicle)

#     def _destroy(self):
#         for actor in self.actors:
#             if actor is not None:
#                 if actor.is_alive:
#                     actor.destroy()

#         # for npc in self.npc:
#         #     if npc is not None:
#         #         if npc.is_alive:
#         #             npc.destroy()

#         self.actors = []
#         # self.npc = []

#     def _close(self):
#         # # Save images and controls
#         # data = np.array(self.images)
#         # np.save("images.npy", data)
#         # # np.savez_compressed('images.npz', array=data)
#         # data = np.array(self.controls)
#         # np.save("controls.npy", data)
#         # # np.savez_compressed('controls.npz', array=data)

#         # Ensure synchronous mode is turned off
#         self.traffic_manager.set_synchronous_mode(False)
#         settings = self.world.get_settings()
#         settings.synchronous_mode = False
#         self.world.apply_settings(settings)

#         for actor in self.actors:
#             if actor is not None:
#                 if actor.is_alive:
#                     actor.destroy()

#         for npc in self.npc:
#             if npc is not None:
#                 if npc.is_alive:
#                     npc.destroy()

#         self.actors = []
#         self.npc = []

#     def __exit__(self, exc_type, exc_value, traceback):
#         pass


def splat_lidar(lidar):
    # Create a 2D histogram
    xbins = np.linspace(-LIDAR_RANGE, LIDAR_RANGE, 128 + 1)
    ybins = np.linspace(-LIDAR_RANGE, LIDAR_RANGE, 128 + 1)
    hist = np.histogramdd(lidar, bins=[xbins, ybins])[0]

    # Clipping histogram values
    hist = np.clip(hist, 0, 5)

    # Crop the front center of the histogram
    hist_croped = hist[:64, 32:96]

    # Display the image
    cv2.imshow("LiDAR Histogram", hist_croped)
    cv2.waitKey(1)


def lidar_to_2d(lidar):
    xbins = np.linspace(-LIDAR_RANGE, LIDAR_RANGE, 600)
    ybins = np.linspace(-LIDAR_RANGE, LIDAR_RANGE, 600)
    hist = np.histogramdd(lidar, bins=[xbins, ybins])[0]

    # Normalize the histogram to fit in the 0-255 range
    hist_normalized = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit image (grayscale)
    hist_image = np.uint8(hist_normalized)

    # Resize the image for better visibility (optional)
    hist_image_resized = cv2.resize(
        hist_image, (256, 256), interpolation=cv2.INTER_NEAREST
    )

    # Display the image
    cv2.imshow("LiDAR Histogram", hist_image_resized)
    cv2.waitKey(1)

    # # Plotting the histogram
    # plt.figure(figsize=(8, 6))
    # plt.imshow(
    #     hist.T,
    #     extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
    #     origin="lower",
    # )
    # plt.colorbar(label="Count")
    # plt.xlabel("X coordinate")
    # plt.ylabel("Y coordinate")
    # plt.title("2D Histogram of LiDAR Data")
    # plt.show()


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


def visualize_lidar(points, name):
    lidar_range = LIDAR_RANGE
    disp_size = (1000, 1000)

    lidar_data = np.array(points[:, :2])
    lidar_data[:, 0] *= -1
    lidar_data += (lidar_range, lidar_range)
    lidar_data *= (disp_size[0] / (lidar_range * 2), disp_size[1] / (lidar_range * 2))

    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (disp_size[0], disp_size[1])
    lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

    lidar_img[tuple(lidar_data.T)] = 255
    lidar_img = np.stack((lidar_img,) * 3, axis=-1)

    # Plotting
    cv2.imshow(name, lidar_img)
    cv2.waitKey(1)


if __name__ == "__main__":
    import cv2
    import PIL

    env = CarlaEnvFusion(debug=True)
    env.reset()
    # print(len(env.spawn_points))
    try:
        # Get the start time
        start_time = time.time()
        while time.time() - start_time < 1e10:
            # obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            obs, reward, terminated, truncated, _ = env.step(
                # [(random.random() ** 0.3) * 2 - 1, random.uniform(-1, 1)]
                # [1.0, random.uniform(-1, 1)]
                [1.0, 0.0]
            )
            # lidar_data = [data[2] for data in env.lidar_data]
            # print("Min: ", np.min(lidar_data))
            # print("Max: ", np.max(lidar_data))
            # lidar_to_2d(env.lidar_data)
            # splat_lidar(env.lidar_data)
            # lidar_data = env.lidar_data
            # transformed_lidar_data = transform_2d_points(
            #     lidar_data,
            #     math.radians(90),
            #     0,
            #     0,
            #     math.radians(90),
            #     0,
            #     0,
            # )
            # visualize_lidar(lidar_data)
            # visualize_lidar(transformed_lidar_data)
            # print("Ego location: ", env.ego_vehicle.get_location())
            # print("NPC location: ", env.vehicle.get_location())
            # cv2.imshow("Lower LIDAR", env.lower_lidar_data)
            # cv2.imshow("Upper LIDAR", env.upper_lidar_data)
            cv2.imshow("RGB", env.img_tracking)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # # Save image
            # PIL.Image.fromarray(obs["rgb"]).save(f"test/{env.frame}.png")
            # break
            # print(env.imu_data[0])
            # print(SPEED_THRESHOLD * obs[2])
            # print(obs.shape)
            # print(obs[1])
            # print(reward)
            if terminated or truncated:
                env.reset()
                # print(len(env.world.get_actors()))
                # print(len(env.spawn_points))
            # time.sleep(0.1)
        print(f"Time elapsed: {time.time() - start_time}s")
    except Exception as e:
        print(e)
    finally:
        env._close()
