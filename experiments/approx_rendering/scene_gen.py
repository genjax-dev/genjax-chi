import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


p.resetSimulation()
p.setGravity(0, 0, -5)

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

cubeStartPos = [0, -5, 5]
cubeStartOrientation = p.getQuaternionFromEuler([np.pi/10,np.pi/3,np.pi/5])
brick_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 1.0, 1.0])
brick = p.createMultiBody(baseMass=0.1,
                            baseCollisionShapeIndex=brick_coll,
                            basePosition=cubeStartPos,
                            baseOrientation=cubeStartOrientation)
p.changeDynamics(planeId, -1, restitution=1.0)
p.changeDynamics(brick, -1, restitution=1.05)

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
far = 200.0
projMatrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)


cx, cy = width / 2.0, height / 2.0
fov_y = np.deg2rad(fov)
fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
fx = cx / np.tan(fov_x / 2.0)
fy = cy / np.tan(fov_y / 2.0)

rgb_imgs = []
depth_imgs = []
for _ in range(100):
    for _ in range(30):     
        p.stepSimulation()
    w,h, rgb, depth, segmentation = p.getCameraImage(width, height,
        viewMatrix,
        projMatrix
    )
    depth = far * near / (far - (far - near) * depth)
    rgb_imgs.append(rgb)
    depth_imgs.append(depth)

<<<<<<< HEAD
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
=======
images = []
for rgb in rgb_imgs:
    images.append(
        Image.fromarray(
            rgb.astype(np.int8), mode="RGBA"
        )
    )

images[0].save(
    fp="rgb.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)

np.savez("data.npz", rgb_imgs=rgb_imgs, depth_imgs=depth_imgs, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
>>>>>>> save
