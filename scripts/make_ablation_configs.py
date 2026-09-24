import copy
import os
import yaml

BASE = "configs/byol_cifar10.yaml"

with open(BASE, "r") as f:
    base = yaml.safe_load(f)

experiments = {
    "A_byol": {
        "use_ema": True,
        "stop_gradient": True,
        "use_predictor": True,
        "use_sigreg": False,
    },

    "B_byol_sigreg": {
        "use_ema": True,
        "stop_gradient": True,
        "use_predictor": True,
        "use_sigreg": True,
    },

    "C_noema_sigreg": {
        "use_ema": False,
        "stop_gradient": True,
        "use_predictor": True,
        "use_sigreg": True,
    },

    "D_nostop_sigreg": {
        "use_ema": False,
        "stop_gradient": False,
        "use_predictor": True,
        "use_sigreg": True,
    },

    "E_symmetric_sigreg": {
        "use_ema": False,
        "stop_gradient": False,
        "use_predictor": False,
        "use_sigreg": True,
    },

    "F_symmetric_noreg": {
        "use_ema": False,
        "stop_gradient": False,
        "use_predictor": False,
        "use_sigreg": False,
    },
}

for name, method in experiments.items():

    cfg = copy.deepcopy(base)

    cfg["experiment"]["name"] = name

    cfg["method"] = method

    cfg["sigreg"] = {
        "weight": 1.0,
        "num_slices": 256,
        "knots": 17,
    }

    path = f"configs/{name}.yaml"

    with open(path, "w") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    print("Created:", path)
