# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each current_epoch."""
    if ray.train._internal.session._get_session():  # replacement for deprecated ray.tune.is_session_enabled()
        metrics = trainer.metrics
        metrics["current_epoch"] = trainer.current_epoch
        session.report(metrics)


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
