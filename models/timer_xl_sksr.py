from models.timer_xl import Model as TimerXLModel
from layers.stable_spectral_residual import StableKoopmanSpectralResidual


class Model(TimerXLModel):
    """Timer-XL plus Stable Koopman Spectral Residual.

    This wrapper preserves the original Timer-XL implementation and applies a
    bounded spectral continuation branch only after the base model has produced
    its forecast. Use `--model timer_xl_sksr` to enable it.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.sksr = StableKoopmanSpectralResidual(
            n_vars=getattr(configs, "n_vars", None),
            max_modes=getattr(configs, "sksr_max_modes", 8),
            lowpass_ratio=getattr(configs, "sksr_lowpass_ratio", 0.35),
            max_mix=getattr(configs, "linear_res_scale", 0.35),
            init_gate=getattr(configs, "linear_init_gate", -3.0),
            trend_window=getattr(configs, "sksr_trend_window", 192),
            use_trend=not getattr(configs, "sksr_no_trend", False),
        )

    def forecast(self, x, x_mark=None, y_mark=None):
        ret = super().forecast(x, x_mark, y_mark)
        if isinstance(ret, tuple):
            out, attns = ret
            out = self.sksr(x, out)
            return out, attns
        return self.sksr(x, ret)

    def sksr_metrics(self):
        return {k: float(v.detach().cpu()) for k, v in self.sksr.last_stats.items()}
