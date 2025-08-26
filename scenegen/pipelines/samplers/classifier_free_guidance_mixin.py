from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred

class ClassifierFreeGuidanceSamplerVGGTMixin:
    """
    A mixin class for samplers that apply classifier-free guidance,
    specifically designed for models that return both image prediction and positions.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred_output = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred_output = super()._inference_model(model, x_t, t, neg_cond, **kwargs)

        pred_v, positions = pred_output
        neg_pred_v, neg_positions = neg_pred_output

        guided_pred_v = (1 + cfg_strength) * pred_v - cfg_strength * neg_pred_v
        
        guided_positions = positions

        return guided_pred_v, guided_positions
