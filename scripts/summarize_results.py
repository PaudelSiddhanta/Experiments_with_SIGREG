import os
import csv

experiments = [
    "A_byol",
    "B_byol_sigreg",
    "C_noema_sigreg",
    "D_nostop_sigreg",
    "E_symmetric_sigreg",
    "F_symmetric_noreg",
]

base = "/scratch/sp7007/byol_sigreg/outputs"

rows = []

for exp in experiments:
    run_dir = os.path.join(base, exp, "seed_0")

    train_csv = os.path.join(run_dir, "train.csv")
    probe_csv = os.path.join(run_dir, "linear_probe.csv")

    result = {
        "experiment": exp,
        "final_ssl_loss": None,
        "final_inv_loss": None,
        "final_sigreg_loss": None,
        "best_probe_acc": None,
        "final_probe_acc": None,
    }

    # SSL training results
    if os.path.exists(train_csv):
        with open(train_csv, newline="") as f:
            train_rows = list(csv.DictReader(f))

        if train_rows:
            last = train_rows[-1]

            result["final_ssl_loss"] = float(last["loss"])
            result["final_inv_loss"] = float(last["invariance_loss"])
            result["final_sigreg_loss"] = float(last["sigreg_loss"])

    # Linear probe results
    if os.path.exists(probe_csv):
        with open(probe_csv, newline="") as f:
            probe_rows = list(csv.DictReader(f))

        if probe_rows:
            accuracies = [
                float(r["test_accuracy"])
                for r in probe_rows
            ]

            result["best_probe_acc"] = max(accuracies)
            result["final_probe_acc"] = accuracies[-1]

    rows.append(result)


print()
print(
    f"{'Experiment':24s} "
    f"{'SSL loss':>10s} "
    f"{'Inv':>10s} "
    f"{'SIGReg':>10s} "
    f"{'Best Acc':>10s} "
    f"{'Final Acc':>10s}"
)

print("-" * 82)

for r in rows:

    def fmt(x):
        return "N/A" if x is None else f"{x:.4f}"

    def acc(x):
        return "N/A" if x is None else f"{x:.2f}%"

    print(
        f"{r['experiment']:24s} "
        f"{fmt(r['final_ssl_loss']):>10s} "
        f"{fmt(r['final_inv_loss']):>10s} "
        f"{fmt(r['final_sigreg_loss']):>10s} "
        f"{acc(r['best_probe_acc']):>10s} "
        f"{acc(r['final_probe_acc']):>10s}"
    )


# Also write combined CSV
output_csv = os.path.join(base, "summary.csv")

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print("Saved summary to:", output_csv)
