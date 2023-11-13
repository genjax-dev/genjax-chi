from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pyro import poutine

from .air import latents_to_tensor


def bounding_box(z_where, x_size):
    """This doesn't take into account interpolation, but it's close enough to
    be usable."""
    w = x_size / z_where.s
    h = x_size / z_where.s
    xtrans = -z_where.x / z_where.s * x_size / 2.0
    ytrans = -z_where.y / z_where.s * x_size / 2.0
    x = (x_size - w) / 2 + xtrans  # origin is top left
    y = (x_size - h) / 2 + ytrans
    return (x, y), w, h


z_obj = namedtuple("z", "s,x,y,pres")


# Map a tensor of latents (as produced by latents_to_tensor) to a list
# of z_obj named tuples.
def tensor_to_objs(latents):
    return [[z_obj._make(step) for step in z] for z in latents]


def visualize_model(examples_to_viz, air):
    trace = poutine.trace(air.guide).get_trace(examples_to_viz, None)
    z, recons = poutine.replay(air.prior, trace=trace)(examples_to_viz.size(0))
    z_wheres = tensor_to_objs(latents_to_tensor(z))
    draw_many(examples_to_viz.reshape(-1, 50, 50), z_wheres, "Original")
    draw_many(recons, z_wheres, "Reconstruction")


def colors(k):
    return ["r", "g", "b"][k % 3]


def draw_one(img, z):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.detach().cpu(), cmap="gray_r")
    for k, z in enumerate(z):
        if z.pres > 0:
            (x, y), w, h = bounding_box(z, img.shape[0])
            plt.gca().add_patch(
                Rectangle(
                    (x, y), w, h, linewidth=1, edgecolor=colors(k), facecolor="none"
                )
            )


def draw_many(imgs, zs, title):
    plt.figure(figsize=(8, 1.9))
    plt.title(title)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.box(False)
    for i, (img, z) in enumerate(zip(imgs, zs)):
        plt.subplot(1, len(imgs), i + 1)
        draw_one(img, z)
    plt.show()
