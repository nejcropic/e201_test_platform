from pathlib import Path
import yaml
import csv


def load_yaml(file_name):
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    return data


def assert_no_paths(obj, where="root"):
    if isinstance(obj, Path):
        raise TypeError(f"Path leaked into config at {where}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_paths(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_paths(v, f"{where}[{i}]")


def save_to_yaml(data, file_name):
    assert_no_paths(data)
    with open(file_name, "w", encoding="utf-8") as savefile:
        yaml.dump(data, savefile)


def save_csv(path: Path, data: list[dict]):
    """
    data = self.recorded_data (your existing structure)
    """

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # HEADER
        writer.writerow(
            [
                "sample_idx",
                "timestamp_ns",
                "ref_counts",
                "dut_counts",
                "ref_deg",
                "dut_deg",
            ]
        )

        # DATA
        for row in data:
            writer.writerow(
                [
                    row["x"],
                    row["ts"],
                    row["ref_counts"],
                    row["dut_counts"],
                    row["ref_deg"],
                    row["dut_deg"],
                ]
            )
