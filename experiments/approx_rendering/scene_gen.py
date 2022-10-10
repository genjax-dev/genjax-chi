import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


p.resetSimulation()
physicsClient = p.connect(p.DIRECT)
p.setGravity(0, 0, -10)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

cubeStartPos = [0, 0, 5]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
brick_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 1.0, 1.0])
brick = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=brick_coll,
    basePosition=cubeStartPos,
    baseOrientation=cubeStartOrientation,
)


viewMatrix = p.computeViewMatrix(
    cameraEyePosition=np.array([20.0, 0.0, 1.0]),
    cameraTargetPosition=np.array([0.0, 0.0, 1.0]),
    cameraUpVector=np.array([0.0, 0.0, 1.0]),
)

fov = 30.0
width = 640  # 3200
height = 480  # 2400
aspect_ratio = width / height
near = 0.0001
far = 25.0
projMatrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)

depth = far * near / (far - (far - near) * depth)
cx, cy = width / 2.0, height / 2.0
fov_y = np.deg2rad(fov)
fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
fx = cx / np.tan(fov_x / 2.0)
fy = cy / np.tan(fov_y / 2.0)


for _ in range(100):
    p.stepSimulation()

w, h, rgb, depth, segmentation = p.getCameraImage(
    width, height, viewMatrix, projMatrix
)

np.savez(
    "data.npz",
    depth=depth,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    width=width,
    height=height,
)

plt.clf()
plt.imshow(rgb)
plt.savefig("out.png")
