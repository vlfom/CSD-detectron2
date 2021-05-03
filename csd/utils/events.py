import wandb
from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """Periodically writes all scalars to Wandb.

    See `d2.utils.JSONWriter` as an example.
    """

    def __init__(self, window_size: int = 20, **kwargs):
        """
        Args:
            window_size (int): the scalars will be median-smoothed by this window size
        """
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()

        metrics_dict = {}
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                metrics_dict[k] = v
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # Log to Wandb
        metrics_dict["iter"] = metrics_dict["global_step"] = storage.iter
        wandb.log(metrics_dict, step=storage.iter)

    def close(self):
        pass
