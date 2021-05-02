import detectron2.utils.comm as comm
import wandb
from detectron2.engine import EvalHook
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils.events import get_event_storage


class CSDEvalHook(EvalHook):
    """Adds several lines of code to upload the metrics to wandb.

    See `d2.engine.EvalHook` for all details. Changes are commented with "CSD: ..."."""

    def _do_eval(self):
        """See `d2.engine.hooks.CSDEvalHook._do_eval`."""

        results = self._func()

        if results:
            assert isinstance(results, dict), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)
    
            # CSD: log values to Wandb
            if comm.is_main_process():
                try:  # Get current iteration
                    iter_ = get_event_storage().iter
                except:  # There is no iter when in eval mode - set to 0
                    iter_ = 0
                flattened_results["global_step"] = iter_
                wandb.log(flattened_results, step=iter_)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()
