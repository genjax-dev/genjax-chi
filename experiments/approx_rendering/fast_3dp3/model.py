import genjax
import jax.numpy as jnp
from dataclasses import dataclass

from .likelihood import neural_descriptor_likelihood
from .rendering import render_cloud_at_pose

@dataclass
class _NeuralDescriptorLikelihood(genjax.Distribution):
    def sample(self, key, *args, **kwargs):
        return key, ()

    def logpdf(self, key, image, *args):
        return neural_descriptor_likelihood(image, *args)

NeuralDescriptorLikelihood = _NeuralDescriptorLikelihood()


@genjax.gen
def model(key, object_model_cloud, h, w, fx_fy, cx_cy ,r , outlier_prob):
    key, pos = genjax.trace("pos", genjax.Uniform)(key, (jnp.array([-10.0, -10.0,-100.0]),jnp.array([10.0, 10.0,100.0])))
    pose = jnp.array(
        [
            [1.0, 0.0, 0.0, pos[0]],
            [0.0, 1.0, 0.0, pos[1]],
            [0.0, 0.0, 1.0, pos[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rendered_image = render_cloud_at_pose(
        object_model_cloud, pose, h, w, fx_fy, cx_cy
    )
    key, cloud = genjax.trace("observed", NeuralDescriptorLikelihood)(
        key, (rendered_image, r, outlier_prob)
    )
    return key, rendered_image.shape

def make_scoring_function(object_model_cloud, h, w, fx_fy, cx_cy ,r , outlier_prob):
    def scorer(key, pos, gt_image):
        obs = genjax.ChoiceMap.new({("pos",): pos, ("observed",): gt_image})
        key, (weight, tr) = model.importance(key, obs, (object_model_cloud, h, w, fx_fy, cx_cy ,r , outlier_prob))
        return key, weight, x
    return scorer