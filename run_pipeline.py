"""Reproduce the Paper 1 OULAD experiments from a CSV directory or ZIP."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
CSV_FILES = (
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
)


def prepare_raw(source):
    if not source.is_file():
        return source

    destination = ROOT / "data_oulad"
    with ZipFile(source) as archive:
        for name in CSV_FILES:
            matches = [item for item in archive.namelist() if Path(item).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in {source}.")
        destination.mkdir(exist_ok=True)
        for name in CSV_FILES:
            member = next(item for item in archive.namelist() if Path(item).name == name)
            with archive.open(member) as input_file, (destination / name).open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data_oulad",
                        help="Path to the extracted OULAD CSV directory or download ZIP.")
    parser.add_argument("--stage", choices=("all", "prepare", "benchmark", "explain", "report"),
                        default="all", help="Run the full workflow or one stage.")
    args = parser.parse_args()
    raw = args.raw.resolve()

    if args.stage in ("all", "prepare"):
        if not raw.exists():
            parser.error(f"OULAD input does not exist: {raw}")
        raw = prepare_raw(raw)
        missing = [name for name in CSV_FILES if not (raw / name).is_file()]
        if missing:
            parser.error(f"Missing files in {raw}: {', '.join(missing)}")
    elif args.stage == "explain" and not (raw / "studentInfo.csv").is_file():
        parser.error("The explain stage requires an extracted OULAD directory with studentInfo.csv.")

    environment = {**os.environ, "DATA": "code", "RAW": str(raw), "MPLBACKEND": "Agg"}
    (ROOT / "results").mkdir(exist_ok=True)

    def run(script, *arguments, **settings):
        print(f"Running {script} {' '.join(arguments)}", flush=True)
        subprocess.run([sys.executable, f"code/{script}.py", *arguments], cwd=ROOT,
                       env={**environment, **settings}, check=True)

    if args.stage in ("all", "prepare"):
        run("build_features", str(raw), "code")
        run("shift")
        run("eda")
    if args.stage in ("all", "benchmark"):
        run("run_exp", "main", "5", MSET="classical")
        run("run_exp", "main", "5", MSET="hybrid")
        run("run_exp", "main", "3", MSET="extra")
        run("run_exp", "ablation", "5")
        run("run_exp", "stream")
        run("run_exp", "anytime", AMETH="LightGBM-T,LightGBM-P,Hybrid-FT-L2SP",
            AOUT="results/anytime.csv")
        run("run_exp", "anytime", AMETH="Hybrid-Head", AOUT="results/anytime_head.csv")
    if args.stage in ("all", "explain"):
        run("xai")
    if args.stage in ("all", "report"):
        run("analyze")


if __name__ == "__main__":
    main()
