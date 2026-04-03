import json
import os
from algorithm.config import INSTANCE, configure_runtime
from algorithm.initialization import run_montecarlo


def run(*, m=10000000, instance=None, default_overrides=None, seed=None, output_path=None):
    if instance is not None or default_overrides is not None:
        configure_runtime(instance=instance if instance is not None else INSTANCE, default_overrides=default_overrides)

    mins, maxs = run_montecarlo(m, seed=seed)

    result = {
        "m": int(m),
        "red_obj_mins": [float(x) for x in mins],
        "red_obj_maxs": [float(x) for x in maxs],
    }

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
