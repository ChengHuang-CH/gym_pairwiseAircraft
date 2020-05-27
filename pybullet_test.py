import numpy as np
import pybullet as p
import time

DURATION = 10000

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

ownerX = -50
ownerY = 0
ownerZ = 0
ownerQuatenion = p.getQuaternionFromEuler([0, 0, 90 / 180 * 3.14])  # rad
owner = p.loadURDF("./gym_pairwiseAircraft/urdfFile/owner.urdf",
                   basePosition=[ownerX, ownerY, ownerZ],
                   baseOrientation=ownerQuatenion,
                   globalScaling=2)
# p.resetBaseVelocity(owner,
#                     linearVelocity=[1.111, 0, 0],
#                     angularVelocity=[0, 0, 0])

# print("getBaseVelocity", p.getBaseVelocity(owner))

intruderX = 35
intruderY = -50
intruderZ = 0
intruderQuaternion = p.getQuaternionFromEuler([0, 0, -151 / 180 * 3.14])  # rad
intruder = p.loadURDF(
    "./gym_pairwiseAircraft/urdfFile/intruder.urdf",
    basePosition=[intruderX, intruderY, intruderZ],
    baseOrientation=intruderQuaternion,
    globalScaling=2)
timeStep = 1
p.setTimeStep(timeStep)
v = 0.1111  # nmi/s
t1 = 470
h = 5
t2 = 100
for i in range(DURATION):
    p.resetBaseVelocity(intruder,
                        linearVelocity=[-v * np.cos(61 * np.pi / 180), v * np.sin(61 * np.pi / 180), 0],
                        angularVelocity=[0, 0, 0])
    p.resetBaseVelocity(owner,
                        linearVelocity=[v, 0, 0],
                        angularVelocity=[0, 0, 0])
    # if i < t1:
    #     p.resetBaseVelocity(owner,
    #                         linearVelocity=[v, 0, 0],
    #                         angularVelocity=[0, 0, 0])
    if t1 <= i < t1 + t2:
        p.resetBaseVelocity(owner,
                            linearVelocity=[v * np.cos(np.arcsin(h / (v * t2))), 0, v * (h / (v * t2))],
                            angularVelocity=[0, 0, 0])
    if t1 + t2 < i < t1 + 2 * t2:
        p.resetBaseVelocity(owner,
                            linearVelocity=[v * np.cos(np.arcsin(h / (v * t2))), 0, -v * (h / (v * t2))],
                            angularVelocity=[0, 0, 0])
    if i > t1 + 2 * t2:
        p.resetBaseVelocity(owner,
                            linearVelocity=[v, 0, 0],
                            angularVelocity=[0, 0, 0])

    ownerPos, ownerOrn = p.getBasePositionAndOrientation(owner)
    intruderPos, intruderOrn = p.getBasePositionAndOrientation(intruder)
    # print("ownerPos, intruderPos", [ownerPos, intruderPos])
    # print("getBaseVelocity", [p.getBaseVelocity(owner), p.getBaseVelocity(intruder)])
    contactPoints = p.getContactPoints(owner, intruder)
    closetPoints = p.getClosestPoints(owner, intruder, 50) #
    '''output format:
    ((0, 0, 1, -1, -1, (-21.730744373009323, -0.04453755058001718, 0.07431134903918696),
      (21.26892218134371, -25.28949702961894, 0.07431134903918696), (-0.8623630313458847, 0.506290432625323, 0.0),
      49.8626042529255, 0.0, 0.0, (5.111670427567e-312, 6.95165968059067e-310, 0.0), 0.0,
      (0.0, 1.0, 1.3234071696055e-311)),)
    '''

    if contactPoints:
        ownerContactPosition, intruderContactPosition, contactDistance = contactPoints[0][5], contactPoints[0][6], \
                                                                         contactPoints[0][8]
        print("Warning!! contact diantace:", contactDistance)
    if closetPoints:
        ownerClosePosition, intruderClosePosition, closeDistance = closetPoints[0][5], closetPoints[0][6], \
                                                                   closetPoints[0][8]
        print("closest distance:", closeDistance)

    p.stepSimulation()
    time.sleep(0.1)

p.disconnect()
