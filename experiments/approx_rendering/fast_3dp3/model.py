import genjax
from dataclasses import dataclass

from .likelihood import neural_descriptor_likelihood
from .rendering import render_cloud_at_pose


@dataclass
class _NeuralDescriptorLikelihood(genjax.Distribution):
    def random_weighted(self, key, *args, **kwargs):
        return key, (0.0, None)

    def estimate_logpdf(self, key, image, *args):
        w = neural_descriptor_likelihood(image, *args)
        return key, (w, image)


NeuralDescriptorLikelihood = _NeuralDescriptorLikelihood()


def make_scoring_function(
    object_model_cloud, h, w, fx_fy, cx_cy, r, outlier_prob, pixel_smudge
):
    @genjax.gen
    def model(key, pose, object_model_cloud):
        rendered_image = render_cloud_at_pose(
            object_model_cloud, pose, h, w, fx_fy, cx_cy, pixel_smudge
        )
        key, cloud = genjax.trace("observed", NeuralDescriptorLikelihood)(
            key, (rendered_image, r, outlier_prob)
        )
        return key, rendered_image.shape

    def scorer(key, pose, gt_image):
        obs = genjax.ChoiceMap.new({("observed",): gt_image})
        key, (weight, tr) = model.importance(
            key,
            obs,
            (
                pose,
                object_model_cloud,
            ),
        )
        return key, weight, pose

    return scorer
