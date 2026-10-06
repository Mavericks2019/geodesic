"""Turn numerical trajectories into auditable, offline animation data."""
from pathlib import Path
import json
import time
import numpy as np

from .solver import solve_boundary, integrate_original, integrate_task_geodesic
from .geometry import Metric


def _integral(y, t):
    return float(np.trapezoid(y, t) if hasattr(np, "trapezoid") else np.trapz(y, t))


def build_case(case, samples=241, knots=41, task_mode=False):
    started = time.perf_counter()
    metric = case.metric
    if task_mode and not case.original:
        weights = np.zeros(case.robot.dof)
        weights[-1] = 1.
        metric = Metric(case.robot, weights, 0.)
    if case.original:
        result = integrate_original(case.q0, case.v0, case.duration, samples)
    elif task_mode:
        result = integrate_task_geodesic(case.robot, case.q0, case.robot.points(case.q1)[-1], case.duration, samples)
    else:
        result = solve_boundary(case.metric, case.q0, case.q1, samples=samples, knots=knots)
    if not result.success:
        raise RuntimeError(f"{case.id}: {result.message}")
    t, q, v = result.t, result.q, result.v
    if not case.original and not task_mode and case.duration != 1.0:
        if case.duration <= 0 or not np.isfinite(case.duration):
            raise ValueError("Duration must be finite and positive")
        t, v = t * case.duration, v / case.duration
    target = q[-1] if task_mode or case.q1 is None else case.q1
    u = (t - t[0]) / (t[-1] - t[0])
    linear = case.q0 + u[:, None] * (target - case.q0)
    linear_v = (target - case.q0) / (t[-1] - t[0])
    energies = np.array([metric.energy(a, b) for a, b in zip(q, v)])
    linear_energies = np.array([metric.energy(a, linear_v) for a in linear])
    energy, baseline_energy = _integral(energies, t), _integral(linear_energies, t)
    length = _integral(np.sqrt(2 * energies), t)
    baseline_length = _integral(np.sqrt(2 * linear_energies), t)
    points = np.array([case.robot.points(a) for a in q])
    eigenvalues = np.array([np.linalg.eigvalsh(metric.matrix(a))[0] for a in q])
    # Independent sampled derivative checks the ODE, rather than repeating its RHS.
    numeric_a = np.gradient(v, t, axis=0, edge_order=2)
    if task_mode and not case.original:
        ode_a = np.array([-np.linalg.pinv(case.robot.point_jacobians(a)[-1], rcond=1e-10) @
                          case.robot.directional_accelerations(a, b)[-1] for a, b in zip(q, v)])
    else:
        ode_a = np.array([metric.acceleration(a, b) for a, b in zip(q, v)])
    residual = float(np.linalg.norm((numeric_a - ode_a)[2:-2]) /
                     max(np.linalg.norm(ode_a[2:-2]), 1.0))
    endpoint_error = float(max(np.linalg.norm(q[0] - case.q0), np.linalg.norm(q[-1] - target)))
    drift = float(np.ptp(energies) / max(abs(np.mean(energies)), 1e-12))
    improvement = 100 * (baseline_energy - energy) / max(baseline_energy, 1e-12)
    tip = points[:, -1, :]
    expected_tip = tip[0] + u[:, None] * (tip[-1] - tip[0])
    line_error = float(np.max(np.linalg.norm(tip - expected_tip, axis=1)))
    task_rank = int(np.linalg.matrix_rank(case.robot.point_jacobians(q[0])[-1], tol=1e-8))
    singular_values = np.array([np.linalg.svd(case.robot.point_jacobians(a)[-1], compute_uv=False)[task_rank - 1] for a in q])
    checks = [
        {"name": "求解器收敛", "passed": bool(result.success), "detail": result.message},
        {"name": "数据有限", "passed": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(v))),
         "detail": f"{len(t)} 帧 / {case.robot.dof} 关节"},
        {"name": "起终关节角一致", "passed": endpoint_error < 1e-7,
         "detail": f"最大误差 {endpoint_error:.3e} rad"},
        {"name": "度量正定", "passed": bool(eigenvalues.min() > 1e-9),
         "detail": f"最小特征值 {eigenvalues.min():.3e}"},
        {"name": "测地线能量守恒", "passed": drift < 5e-4,
         "detail": f"相对波动 {drift:.3e}"},
        {"name": "方程残差", "passed": residual < 5e-3,
         "detail": f"采样速度数值微分的相对残差 {residual:.3e}"},
        {"name": "不高于线性插值代价", "passed": bool(energy <= baseline_energy * 1.0001 + 1e-10),
         "detail": f"积分能量 {energy:.6g} / 基线 {baseline_energy:.6g}"},
    ]
    if task_mode and not case.original:
        target_position = case.robot.points(case.q1)[-1]
        task_endpoint_error = float(np.linalg.norm(tip[-1] - target_position))
        endpoint_error = task_endpoint_error
        checks[2] = {"name": "末端目标位置到达", "passed": task_endpoint_error < 1e-6,
                     "detail": f"位置误差 {task_endpoint_error:.3e} m；末端位置固定，关节终姿态由升轨决定"}
        checks[3] = {"name": "任务雅可比保持秩", "passed": bool(singular_values.min() > 1e-7),
                     "detail": f"任务秩 {task_rank}，最小有效奇异值 {singular_values.min():.3e}；关节度量允许半正定"}
        checks[4]["name"] = "原始末端能量守恒"
        checks[5]["name"] = "SVD 升轨方程残差"
        checks.append({"name": "末端匀速直线", "passed": line_error < 1e-6,
                       "detail": f"最大位置偏差 {line_error:.3e} m"})
    if case.original:
        tip = points[:, -1, :]
        expected = tip[0] + u[:, None] * (tip[-1] - tip[0])
        error = float(np.max(np.linalg.norm(tip - expected, axis=1)))
        checks.append({"name": "原始末端匀速直线", "passed": error < 1e-6,
                       "detail": f"最大偏差 {error:.3e} m"})
    def rounded(a):
        return np.round(a, 12).tolist()
    output = {
        "id": case.id, "name": case.name, "kind": case.kind, "dof": case.robot.dof,
        "description": case.description,
        "method": "原始末端拉回度量 · 初值积分" if case.original else
                  ("原始末端能量 · SVD 冗余升轨（无正则项）" if task_mode else "全连杆拉回度量 + 正则项 · 两端边值求解"),
        "metric_mode": "endpoint" if task_mode or case.original else "whole",
        "robot": {"axes": case.robot.axes.tolist(), "offsets": case.robot.offsets.tolist(),
                  "base": case.robot.base.tolist()},
        "metric": {"weights": metric.weights.tolist(), "ridge": metric.ridge},
        "t": rounded(t), "q": rounded(q), "points": rounded(points),
        "baseline": {"q": rounded(linear), "points": rounded(np.array([case.robot.points(a) for a in linear])),
                     "energy": baseline_energy, "length": baseline_length},
        "metrics": {"energy": energy, "length": length, "energy_improvement_pct": improvement,
                    "max_energy_drift": drift, "min_eigenvalue": float(eigenvalues.min()),
                    "endpoint_error": endpoint_error, "geodesic_residual": residual,
                    "solve_seconds": time.perf_counter() - started,
                    "cartesian_line_error": line_error, "task_rank": task_rank,
                    "min_task_singular_value": float(singular_values.min())},
        "checks": checks, "solver": {"success": bool(result.success), "message": result.message},
        "curve": {"energy": rounded(energies), "speed": rounded(np.sqrt(2 * energies)),
                  "baseline_energy": rounded(linear_energies)},
    }
    if task_mode and not case.original:
        output["description"] = f"{case.robot.dof}R 保持原始末端能量，末端沿直线匀速运动；SVD 选择最小范数初速度与加速度。"
        output["reference_q1"] = case.q1.tolist()
        output["requested_target_position"] = target_position.tolist()
    return output


def write_dataset(cases, destination, source_commit, comparison_cases=None):
    checks = [check for c in cases for check in c["checks"]]
    data = {
        "source": {"url": "https://github.com/Mavericks2019/geodesic", "commit": source_commit,
                   "method": "Original end-effector Euclidean energy; separate full-link energy comparison"},
        "cases": cases, "summary": {"passed": sum(c["passed"] for c in checks), "total": len(checks)},
    }
    if comparison_cases is not None:
        comparison_checks = [check for c in comparison_cases for check in c["checks"]]
        data["comparison_cases"] = comparison_cases
        data["comparison_summary"] = {"passed": sum(c["passed"] for c in comparison_checks), "total": len(comparison_checks)}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (destination / "trajectories.json").write_text(json_text + "\n", encoding="utf-8")
    report = {"source": data["source"], "summary": data["summary"],
              "cases": [{key: c[key] for key in ("id", "name", "metric_mode", "metric", "metrics", "checks", "solver")} for c in cases]}
    if comparison_cases is not None:
        report["comparison_summary"] = data["comparison_summary"]
        report["comparison_cases"] = [{key: c[key] for key in ("id", "name", "metric_mode", "metric", "metrics", "checks", "solver")} for c in comparison_cases]
    (destination / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
