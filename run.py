"""Offline geodesic animation lab. Run `python run.py --help`."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "2203fb21e332f038cdf7f554d255656e5d8340a1"


def main():
    parser = argparse.ArgumentParser(description="Geodesic robot motion lab / 机械臂测地线动画")
    parser.add_argument("--build", action="store_true", help="Recompute all trajectories and validations")
    parser.add_argument("--no-open", action="store_true", help="Do not launch the ImGui/OpenGL desktop viewer")
    parser.add_argument("--robot", type=Path, help="Build one custom robot from JSON instead of preset cases")
    parser.add_argument("--samples", type=int, default=241, help="Trajectory samples (minimum 61)")
    args = parser.parse_args()
    if args.samples < 61:
        parser.error("--samples must be at least 61")
    if args.build or args.robot or not (ROOT / "output" / "trajectories.json").exists():
        from geodesic_lab.scenarios import scenarios, custom_scenario
        from geodesic_lab.export import build_case, write_dataset
        selected = [custom_scenario(json.loads(args.robot.read_text(encoding="utf-8-sig")))] if args.robot else scenarios()
        built, comparisons = [], []
        for case in selected:
            print(f"Computing {case.id} ({case.robot.dof} DOF)...", flush=True)
            data = build_case(case, args.samples, task_mode=True)
            built.append(data)
            print(f"  energy improvement = {data['metrics']['energy_improvement_pct']:.2f}%, "
                  f"energy drift = {data['metrics']['max_energy_drift']:.2e}, "
                  f"time = {data['metrics']['solve_seconds']:.2f}s", flush=True)
            print(f"  Cartesian straight-line error = {data['metrics']['cartesian_line_error']:.2e}m", flush=True)
            comparisons.append(build_case(case, args.samples))
        dataset = write_dataset(built, ROOT / "output", SOURCE_COMMIT, comparison_cases=comparisons)
        checks = dataset["summary"]
        print(f"Validation: {checks['passed']}/{checks['total']} passed", flush=True)
        comparison_checks = dataset["comparison_summary"]
        print(f"Comparison validation: {comparison_checks['passed']}/{comparison_checks['total']} passed", flush=True)
        if checks["passed"] != checks["total"] or comparison_checks["passed"] != comparison_checks["total"]:
            print("Some validation checks failed; inspect output/validation.json", file=sys.stderr)
            return 1
    print(f"Renderer: {ROOT / 'viewer.py'}")
    if not args.no_open:
        native_python = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        subprocess.Popen([str(native_python if native_python.exists() else sys.executable), str(ROOT / "viewer.py")], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
