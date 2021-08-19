

import glob
import os
import sys
import math
import time
import numpy as np
import cv2
import random
from collections import deque
from random import randrange
# from platform import python_version

# print(python_version())
# Global Setting
ROUNDING_FACTOR = 3
SECONDS_PER_EPISODE = 80
LIMIT_RADAR = 600
STATE_SPACE = 602
ACTION_SPACE =1
np.random.seed(32)
random.seed(32)
MAX_LEN = 600

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
'''
[start_x,start_y,start_z,ed_x,end_y,end_z]
'''
# way_points = [[128.0,206.9,3.0,-22.1,204.2,3.0,180],[7.8,-35.6,3,9.7,-108.0,3.0,90],[-8.6,-123.1,3.0,-88.0,-36.1,3.0,90],[5.3,112.2,3.0,4.6,50.2,3.0,90],[59.9,7.4,3.0,190.0,9.9,3.0,270]]
#way_points = [[128.0,206.9,3.0,-22.1,204.2,3.0],[18.1,-134.4,3.0,63.7,-133.5,3.0],[-10.0,34.4,3.0,-9.1,105.4,3.0],[5.6,119.1,3.0,4.9,52.1,3.0],[-77.2,93.6,3.0,-77.3,30.2,3.0]]
# way_points = [[128.0,206.9,3.0,-22.1,204.2,3.0],[-72.3,81.4,3,-74.1,-118.6,3],[-62.5,-133.3,3,-16.2,-132.7,3],[261.4,199.1,3,322.3,198.4,3],[335.1,140.4,3,334.7,182.5,3]]
way_points = [[128.0,206.9,3.0,-22.1,204.2,3.0,180],[7.2,-33.4,3.0,10.0,-106.3,3.0,270],[155.4,-12.3,3.0,157.9,-105.8,3.0,270],[-152.4,-99.3,3.0,-152.5,-7.7,3.0,90]]


class CarlaVehicle(object):
    """
    class responsable of:
            -spawning the ego vehicle
            -destroy the created objects
            -providing environment for RL training
    """

    # Class Variables
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.radar_data = deque(maxlen=MAX_LEN)
        self.observation_space = np.array([STATE_SPACE])
        self.action_space = np.array([ACTION_SPACE])

        

    def reset(self, Norender=False):
        '''reset function to reset the environment before 
        the begining of each episode
        :params Norender: to be set true during training
        '''
        self.collision_hist = []
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.idx = randrange(len(way_points)) #random way point selction on every episode
        self.apply_brake = 0 #after reaching endpoint apply brakes
        self.brake_power = 0.5  #brake power after reaching endpoint
        # print("this is way_point ",way_points[self.idx])
        '''target_location: Orientation details of the target loacation
		(To be obtained from route planner)'''
        # self.target_waypoint = carla.Transform(carla.Location(x = 1.89, y = 117.06, z=0), carla.Rotation(yaw=269.63))

        # Code for setting no rendering mode
        if Norender:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        self.actor_list = []
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = self.blueprint_library.filter("model3")[0]

        # create ego vehicle the reason for adding offset to z is to avoid collision
        init_pos = carla.Transform(carla.Location(
            x=way_points[self.idx][0], y=way_points[self.idx][1], z=way_points[self.idx][2]), carla.Rotation(yaw=way_points[self.idx][6]))
        self.vehicle = self.world.spawn_actor(self.bp, init_pos)
        self.actor_list.append(self.vehicle)
        self.end_pos = carla.Transform(carla.Location(
            x=way_points[self.idx][3], y=way_points[self.idx][4], z=way_points[self.idx][5]), carla.Rotation(yaw=180))
        self.end_pos = np.array([way_points[self.idx][3],way_points[self.idx][4], way_points[self.idx][5]])   
        # Create location to spawn sensors
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        # Create Collision Sensors
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.episode_start = time.time()

        # Create location to spawn sensors
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        # Radar Data Collectiom
        self.radar = self.blueprint_library.find('sensor.other.radar')
        self.radar.set_attribute("range", f"100")
        self.radar.set_attribute("horizontal_fov", f"35")
        self.radar.set_attribute("vertical_fov", f"25")

        # We will initialise Radar Data
        self.resetRadarData(100, 35, 25)

        self.sensor = self.world.spawn_actor(
            self.radar, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_radar(data))
        vehicle_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        current_location = vehicle_waypoint.transform.location

        current_position = []
        current_position.append(current_location.x)
        current_position.append(current_location.y)
        current_position.append(current_location.z)

        current_position = np.array(current_position)
        data = np.array(self.radar_data)
        data = data[-LIMIT_RADAR:]
        data = np.append(data, 0.0)
        data = np.append(data,  (current_location.x - self.end_pos[0])**2+(current_location.y - self.end_pos[1])**2 + (current_location.z - self.end_pos[2])**2)
        # print("data shape ",data.shape)
        return data

    def resetRadarData(self, dist, hfov, vfov):
        # [Altitude, Azimuth, Dist, Velocity]
        alt = 2*math.pi/vfov
        azi = 2*math.pi/hfov

        vel = 0
        deque_list = []
        for _ in range(MAX_LEN//4):
            altitude = random.uniform(-alt, alt)
            deque_list.append(altitude)
            azimuth = random.uniform(-azi, azi)
            deque_list.append(azimuth)
            distance = random.uniform(10, dist)
            deque_list.append(distance)
            deque_list.append(vel)

        self.radar_data.extend(deque_list)

    # Process Camera Image
    def process_radar(self, radar):
        # To plot the radar data into the simulator
        self._Radar_callback_plot(radar)

        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # Parameters :: frombuffer(data_input, data_type, count, offset)
        # count : Number of items to read. -1 means all data in the buffer.
        # offset : Start reading the buffer from this offset (in bytes); default: 0.
        points = np.frombuffer(buffer=radar.raw_data, dtype='f4')
        points = np.reshape(points, (len(radar), 4))
        for i in range(len(radar)):
            self.radar_data.append(points[i, 0])
            self.radar_data.append(points[i, 1])
            self.radar_data.append(points[i, 2])
            self.radar_data.append(points[i, 3])

    # Taken from manual_control.py

    def _Radar_callback_plot(self, radar_data):
        current_rot = radar_data.transform.rotation
        velocity_range = 7.5  # m/s
        world = self.world
        debug = world.debug

        def clamp(min_v, max_v, value):
            return max(min_v, min(value, max_v))

        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)

            debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

    # record a collision event

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        # Apply Vehicle Action
        if(self.apply_brake):
             self.vehicle.apply_control(carla.VehicleControl(
            throttle=0.0, steer=action[1], brake=action[2]+ self.brake_power, reverse=action[3]))
        else:
              self.vehicle.apply_control(carla.VehicleControl(
            throttle=action[0], steer=action[1], brake=action[2], reverse=action[3]))

    # Method to take action by the DQN Agent for straight drive

    def step_straight(self, action, p):
        # print("apply brake ",self.apply_brake)
        done = False
        self.step(action)
        # Calculate vehicle speed
        kmh = self.get_speed()

        vehicle_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        current_location = vehicle_waypoint.transform.location

        current_position = []
        current_position.append(current_location.x)
        current_position.append(current_location.y)
        current_position.append(current_location.z)

        current_position = np.array(current_position)
        norm_dis = math.sqrt((current_location.x - self.end_pos[0])**2+(current_location.y - self.end_pos[1])**2 + (current_location.z - self.end_pos[2])**2)
        # print("current x location")
        # print(obs)
        # if(norm_dis<20):
        #     self.apply_brake=1
        # print(norm_dis)
        if p:
            print(
                f'collision_hist----{self.collision_hist}------kmh----{kmh}------light----{self.vehicle.is_at_traffic_light()}')
        reward = 0
        if(self.apply_brake and kmh>0): #decrease the reward if the ego vehicle stops far away from start point
            reward-=10
        reward = 0
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 1000
            # self.destroy()
        # incorporated that the ego should slow down near end point
        elif norm_dis < 10 and int(kmh)<= 5:
            self.apply_brake=1
            done = False
            reward += 200
        elif norm_dis < 10 and int(kmh)> 5:
            self.apply_brake=1
            done = False
            reward -= 200
        elif kmh < 2:
            done = False
            reward += -1
        elif kmh < 10:
            done = False
            reward += float(kmh/10)
        elif kmh < 25:
            done = False
            reward += 1
            reward += float(kmh/25)
        elif kmh < 50:
            done = False
            reward += 2 - float(kmh/25)
        else:
            done = False
            reward += -1

        # Build in function of Carla
        if self.vehicle.is_at_traffic_light() and kmh < 20:
            done = True
            reward = reward+200
            # self.destroy()
        elif self.vehicle.is_at_traffic_light() and kmh > 20:
            done = True
            reward = reward-100
            # self.destroy()
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            reward = reward-200
            # self.destroy()
        data = np.array(self.radar_data)

        return data[-LIMIT_RADAR:], int(kmh), reward, done, norm_dis

    def destroy(self):
        """
                destroy all the actors
                :param self
                :return None
        """
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')

    def get_speed(self):
        """
                Compute speed of a vehicle in Kmh
                :param vehicle: the vehicle for which speed is calculated
                :return: speed as a float in Kmh
        """
        vel = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
