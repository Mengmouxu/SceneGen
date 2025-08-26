from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)

class GuidanceIntervalSamplerVGGTMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval,
    specifically designed for models that return both image prediction and positions.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred_output = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred_output = super()._inference_model(model, x_t, t, neg_cond, **kwargs)

            pred_v, positions = pred_output
            neg_pred_v, neg_positions = neg_pred_output

            guided_pred_v = (1 + cfg_strength) * pred_v - cfg_strength * neg_pred_v
            
            # Keep original positions
            guided_positions = positions

            return guided_pred_v, guided_positions
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
    