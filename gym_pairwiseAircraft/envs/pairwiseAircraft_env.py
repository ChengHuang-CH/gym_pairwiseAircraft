from pkg_resources import parse_version
import pybullet_utils.bullet_client as bc
import pybullet_data
import pybullet as p2
import subprocess
import time
import numpy as np
from gym.utils import seeding
from gym import spaces
import gym
import math
import logging
import os
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

logger = logging.getLogger(__name__)


class pairwiseAircraftEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, renders=False, actionRepeat=1500, discrete_actions=False):
        # start the bullet physics server
        self._renders = renders
        self._discrete_actions = discrete_actions
        self._render_height = 400
        self._render_width = 420
        self._actionRepeat = actionRepeat
        self._physics_client_id = -1
        self._cam_dist = 40
        self._cam_pitch = -21  # degree
        self._cam_yaw = 200#200  # degree

        if self._renders:
            self._p = bc.BulletClient(connection_mode=p2.GUI)
        else:
            self._p = bc.BulletClient()

        self.v = 0.1111  # nmi/s

        # action space
        self.min_t1 = 100
        self.max_t1 = 400
        self.min_h = 30
        self.max_h = 40
        self.min_t2 = 200
        self.max_t2 = 400

        # max area:  Isosceles Right Triangle

        self.max_area = (self.max_t2 * self.v) ** 2 / 4 * 2

        self.low_action = np.array([self.min_t1, self.min_h, self.min_t2],
                                   dtype=np.float32)  # t1, h, t2
        self.high_action = np.array([self.max_t1, self.max_h, self.max_t2],
                                    dtype=np.float32)

        self.action_space = spaces.Box(self.low_action,
                                       self.high_action,
                                       dtype=np.float32)
        print('high_action', self.high_action)

        # observation space
        self.seperationStatus_min = 3
        self.seperationStatus_max = 100

        self.low_state = np.array([self.seperationStatus_min],
                                  dtype=np.float32)
        self.high_state = np.array([self.seperationStatus_max],
                                   dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state,
                                            dtype=np.float32)

        self.seed()
        # self.reset()
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p = self._p
        r1, r2, r3 = 0, 0, 0
        storeClosestPoints = []
        storeOwnerPosition = []
        storeIntruderPosition = []

        t1, h, t2 = action
        print("t1, h, t2", [t1, h, t2])

        maneuver = self.v * t2 / h
        if maneuver > 1:
            area = np.sqrt((self.v * t2)**2 - h**2) * h
            r3 = - area / self.max_area * (h/np.sqrt((self.v * t2)**2 - h**2))
        else:
            r2 = -3

        for i in range(self._actionRepeat):

            p.resetBaseVelocity(self.intruder,
                                linearVelocity=[-self.v * np.cos(61 * np.pi / 180), self.v * np.sin(61 * np.pi / 180),
                                                0],
                                angularVelocity=[0, 0, 0])
            p.resetBaseVelocity(self.owner,
                                linearVelocity=[self.v, 0, 0],
                                angularVelocity=[0, 0, 0])
            if t1 <= i < t1 + t2:
                p.resetBaseVelocity(self.owner,
                                    linearVelocity=[self.v * np.cos(np.arcsin(h / (self.v * t2))), 0,
                                                    self.v * (h / (self.v * t2))],
                                    angularVelocity=[0, 0, 0])
            if t1 + t2 < i < t1 + 2 * t2:
                p.resetBaseVelocity(self.owner,
                                    linearVelocity=[self.v * np.cos(np.arcsin(h / (self.v * t2))), 0,
                                                    -self.v * (h / (self.v * t2))],
                                    angularVelocity=[0, 0, 0])
            if i > t1 + 2 * t2:
                p.resetBaseVelocity(self.owner,
                                    linearVelocity=[self.v, 0, 0],
                                    angularVelocity=[0, 0, 0])

            closetPoints = p.getClosestPoints(self.owner, self.intruder, 100)  # must be put before stepSimulation
            # print("closetPoints", closetPoints)
            ownerPos, ownerOrn = p.getBasePositionAndOrientation(self.owner)
            intruderPos, intruderOrn = p.getBasePositionAndOrientation(self.intruder)
            storeOwnerPosition.append(ownerPos)
            storeIntruderPosition.append(intruderPos)

            if closetPoints:
                closeDistance = closetPoints[0][8]
                storeClosestPoints.append(closeDistance)

            if storeClosestPoints:
                self.state = min(storeClosestPoints)
            else:
                self.state = 95.23294384

            done = False
            contactPoints = p.getContactPoints(self.owner, self.intruder)
            if bool(contactPoints):
                done = bool(contactPoints)
                endType = "failedResolution"
                r1 = -3
                r3 = 0
                print("!!---- End this loop beacuse of collision  -------- ")

            if min(storeClosestPoints) < 5:
                done = min(storeClosestPoints) < 5
                endType = "failedResolution"
                r1 = -3
                r3 = 0
                print("!!---- End this loop beacuse of unsafe distance  -------- ")
                break
            if i == self._actionRepeat - 1:
                done = (i == self._actionRepeat - 1)
                endType = "successfulResolution"
                print("!!---- End this loop beacuse of ended simulation time -------- ")
                break
            if maneuver <= 1:
                done = True
                endType = "failedManuever"
                r2 = -3
                print("!!----End this loop because of the failed manuever ----")
                break
            p.stepSimulation()
            # time.sleep(0.01)

        # 写包含起始冲突的飞行动作过程
        reward = 2 + r1 + r2 + r3
        print('env done+ r1 r2 r3 endType', [done, r1, r2, r3, endType])
        return np.array([self.state]), [reward, r1, r2, r3], done, endType, storeOwnerPosition, storeIntruderPosition

    def reset(self):
        print("-----------reset simulation---------------")

        p = self._p
        p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                     self._cam_pitch, [0, 0, 0])
        p.resetSimulation()
        # print('---reach here---')

        ownerX = -50 #+ np.random.randn(1)
        ownerY = 0
        ownerZ = 0
        ownerQuatenion = p.getQuaternionFromEuler([0, 0,
                                                   90 / 180 * 3.14])  # rad
        self.owner = p.loadURDF(
            "H:/LocalGitFIles/gym_ATM/gym_ATM/urdfFile/owner.urdf",
            basePosition=[ownerX, ownerY, ownerZ],
            baseOrientation=ownerQuatenion,
            globalScaling=2)
        p.resetBaseVelocity(self.owner,
                            linearVelocity=[0.1111, 0, 0],
                            angularVelocity=[0, 0, 0])

        intruderX = 35
        intruderY = -50
        intruderZ = 0
        intruderQuaternion = p.getQuaternionFromEuler(
            [0, 0, -151 / 180 * 3.14])  # rad
        self.intruder = p.loadURDF(
            "H:/LocalGitFIles/gym_ATM/gym_ATM/urdfFile/intruder.urdf",
            basePosition=[intruderX, intruderY, intruderZ],
            baseOrientation=intruderQuaternion,
            globalScaling=2)

        p.resetBaseVelocity(self.intruder,
                            linearVelocity=[-self.v * np.cos(61 * np.pi / 180), self.v * np.sin(61 * np.pi / 180),
                                            0],
                            angularVelocity=[0, 0, 0])

        self.timeStep = 1

        p.setTimeStep(self.timeStep)
        # 0 to disable real-time simulation, 1 to enable
        p.setRealTimeSimulation(0)
        closetPoints = p.getClosestPoints(self.owner, self.intruder, 100)
        # print('closetPoints', closetPoints)
        if closetPoints:
            self.state = closetPoints[0][8]
        else:
            self.state = 95.23294384

        return np.array([self.state])

    def render(self, mode='human', close=False):
        if mode == "human":
            self._renders = True
        if mode != "rgb_array":
            return np.array([])
        base_pos = [0, 0, 0]

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(self._render_width) /
                                                                self._render_height,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def configure(self, args):
        pass



