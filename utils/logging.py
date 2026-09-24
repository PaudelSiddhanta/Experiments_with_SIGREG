import csv
import os


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

    def log(self, values):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )

            writer.writerow(values)
