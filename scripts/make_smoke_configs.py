import os
import copy
import yaml

names = [
    "A_byol",
    "B_byol_sigreg",
    "C_noema_sigreg",
    "D_nostop_sigreg",
    "E_symmetric_sigreg",
    "F_symmetric_noreg",
]

for name in names:

    src = f"configs/{name}.yaml"

    with open(src, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)

    cfg["experiment"]["name"] = name + "_smoke"

    cfg["train"]["epochs"] = 1
    cfg["train"]["max_batches"] = 20
    cfg["train"]["checkpoint_every"] = 1

    cfg["linear_probe"]["epochs"] = 2

    dst = f"configs/{name}_smoke.yaml"

    with open(dst, "w") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    print("Created:", dst)
