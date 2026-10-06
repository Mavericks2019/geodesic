"""Initial-value and fixed-endpoint geodesic solvers."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import DOP853, solve_bvp, solve_ivp
from scipy.optimize import brentq

from .geometry import Metric, Robot, SingularMetricError, _vector


@dataclass
class TrajectoryResult:
    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    success: bool
    message: str
    max_energy_drift: float
    min_eigenvalue: float
    energy: np.ndarray
    endpoint_error: float = 0.0
    task_rank: int | None = None
    min_task_singular_value: float | None = None
    max_task_residual: float | None = None
    max_cartesian_line_error: float | None = None


def _result(metric, t, q, v, success, message, endpoint_error=0.0):
    energy = np.array([metric.energy(position, velocity) for position, velocity in zip(q, v)])
    eigenvalues = np.array([np.linalg.eigvalsh(metric.matrix(position))[0] for position in q])
    scale = max(abs(float(energy[0])), 1e-15)
    drift = float(np.max(np.abs(energy - energy[0])) / scale)
    return TrajectoryResult(
        t=np.asarray(t), q=np.asarray(q), v=np.asarray(v), success=bool(success),
        message=str(message), max_energy_drift=drift,
        min_eigenvalue=float(np.min(eigenvalues)), energy=energy,
        endpoint_error=float(endpoint_error),
    )


def _sample_count(samples):
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    return int(samples)


def _integrate(metric, acceleration, q0, v0, duration, samples):
    n = metric.robot.dof
    q0 = _vector(q0, n, "q0")
    v0 = _vector(v0, n, "v0")
    duration = float(duration)
    samples = _sample_count(samples)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    initial = np.concatenate((q0, v0))
    initial_min = np.linalg.eigvalsh(metric.matrix(q0))[0]
    if metric.ridge == 0.0 and initial_min <= 1e-9:
        return _result(metric, [0.0], [q0], [v0], False,
                       "Initial configuration has a singular or nearly singular metric")

    def rhs(t, state):
        return np.concatenate((state[n:], acceleration(state[:n], state[n:])))

    def near_singular(t, state):
        return np.linalg.eigvalsh(metric.matrix(state[:n]))[0] - 1e-9

    near_singular.terminal = True
    near_singular.direction = -1
    try:
        solution = solve_ivp(
            rhs, (0.0, duration), initial, method="DOP853", dense_output=True,
            rtol=2e-10, atol=2e-12,
            events=near_singular if metric.ridge == 0.0 else None,
            max_step=duration / 40.0,
        )
    except (SingularMetricError, FloatingPointError) as error:
        return _result(metric, [0.0], [q0], [v0], False, str(error))
    final_t = float(solution.t[-1])
    # Always retain the final accepted state, including an early singularity stop.
    t = np.linspace(0.0, final_t, samples)
    states = solution.sol(t)
    singular_stop = metric.ridge == 0.0 and len(solution.t_events[0]) > 0
    message = (
        f"Stopped at t={final_t:.8g}: metric approached a kinematic singularity"
        if singular_stop else solution.message
    )
    return _result(metric, t, states[:n].T, states[n:].T,
                   solution.success and not singular_stop, message)


def integrate_geodesic(metric, q0, v0, duration=1.0, samples=241):
    """Integrate an affine geodesic with accurate adaptive DOP853 steps."""
    return _integrate(metric, metric.acceleration, q0, v0, duration, samples)


def original_acceleration(q, v):
    """Literal port of repository geodesic.m, using q/v rather than interleaved z."""
    q = _vector(q, 2, "q")
    v = _vector(v, 2, "v")
    sine, cosine = np.sin(q[1]), np.cos(q[1])
    if abs(sine) < 1e-10 or abs(cosine - 1.0) < 1e-14:
        raise SingularMetricError("Original two-link metric is singular at this elbow angle")
    a, b = v
    return np.array([
        -(sine * a**2) / (cosine - 1.0) + (2.0 * a * b) / sine + b**2 / sine,
        (2.0 * a**2 * sine) / (cosine - 1.0)
        + (b**2 * sine) / (cosine - 1.0)
        + (2.0 * a * b * sine) / (cosine - 1.0),
    ])


def original_metric():
    """Metric used by step1.m: two unit planar links, tip position only."""
    return Metric(Robot("Original planar 2R", [[0, 0, 1], [0, 0, 1]],
                        [[1, 0, 0], [1, 0, 0]]), [0, 1], ridge=0.0)


def integrate_original(q0, v0, duration=1.0, samples=241):
    """Replay the original ODE, with explicit singularity handling."""
    return _integrate(original_metric(), original_acceleration, q0, v0, duration, samples)


def integrate_task_geodesic(robot, q0, target_position, duration=1.0, samples=241):
    """Lift a straight Cartesian geodesic with the original endpoint-only energy.

    The metric is exactly g=J_tip.T J_tip, without ridge or link weights. For
    redundant robots it is semidefinite, so joint acceleration is not unique.
    This function chooses the Euclidean minimum-norm acceleration
    qddot=-J_tip^+ H_tip[qdot,qdot], with zero null-space acceleration. Initial
    velocity is the minimum-norm lift of the constant requested task velocity.
    Consequently the target fixes a Cartesian endpoint, not a joint endpoint.

    This lift requires the requested line to stay in the reachable task image.
    Rank loss and an unavailable task tangent cause an explicit finite failure.
    """
    n = robot.dof
    q0 = _vector(q0, n, "q0")
    target_position = _vector(target_position, 3, "target_position")
    duration = float(duration)
    samples = _sample_count(samples)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    weights = np.zeros(n)
    weights[-1] = 1.0
    metric = Metric(robot, weights, ridge=0.0)
    initial_position = robot.points(q0)[-1]
    task_velocity = (target_position - initial_position) / duration
    initial_jacobian = robot.point_jacobians(q0)[-1]
    initial_u, initial_s, initial_vh = np.linalg.svd(initial_jacobian, full_matrices=False)
    rank_tolerance = 1e-10 * max(1.0, float(initial_s[0]))
    rank = int(np.count_nonzero(initial_s > rank_tolerance))
    v0 = np.zeros(n)
    if rank:
        v0 = initial_vh[:rank].T @ ((initial_u[:, :rank].T @ task_velocity) / initial_s[:rank])
    projected_velocity = initial_jacobian @ v0
    range_error = float(np.linalg.norm(projected_velocity - task_velocity))
    initial = np.concatenate((q0, v0))
    stop_threshold = 1e-7

    def singular_value(q):
        singular_values = np.linalg.svd(robot.point_jacobians(q)[-1], compute_uv=False)
        return float(singular_values[rank - 1]) if rank else 0.0

    def acceleration_details(q, v):
        _, jacobians, hessian_vv = robot._embedding(q, v)
        jacobian, curvature = jacobians[-1], hessian_vv[-1]
        u, singular_values, vh = np.linalg.svd(jacobian, full_matrices=False)
        # The guard is below the stop threshold: rejected integration stages
        # may probe beyond the event while the last accepted state is healthy.
        inverse_values = 1.0 / np.maximum(singular_values[:rank], 1e-13)
        acceleration = -vh[:rank].T @ ((u[:, :rank].T @ curvature) * inverse_values)
        residual = float(np.linalg.norm(jacobian @ acceleration + curvature))
        retained = float(singular_values[rank - 1]) if rank else 0.0
        return acceleration, residual, retained, float(np.linalg.norm(curvature))

    def finish(t, states, success, message):
        q, v = states[:n].T, states[n:].T
        result = _result(metric, t, q, v, success, message,
                         np.linalg.norm(robot.points(q[-1])[-1] - target_position))
        actual_positions = np.array([robot.points(position)[-1] for position in q])
        expected_positions = initial_position + np.asarray(t)[:, None] * task_velocity
        result.task_rank = rank
        result.min_task_singular_value = min(singular_value(position) for position in q)
        result.max_task_residual = max(acceleration_details(position, velocity)[1]
                                       for position, velocity in zip(q, v))
        result.max_cartesian_line_error = float(np.max(np.linalg.norm(
            actual_positions - expected_positions, axis=1)))
        if result.success and (result.endpoint_error > 1e-7
                               or result.max_cartesian_line_error > 1e-7):
            result.success = False
            result.message = "Cartesian lift failed its endpoint or straight-line accuracy check"
        return result

    if rank == 0 or (rank and initial_s[rank - 1] <= stop_threshold):
        return finish(np.array([0.0]), initial[:, None], False,
                      "Initial end-effector Jacobian is singular or nearly singular")
    if range_error > 1e-10 * max(1.0, float(np.linalg.norm(task_velocity))):
        return finish(np.array([0.0]), initial[:, None], False,
                      f"Target displacement lies outside the initial task tangent; residual={range_error:.3g}")

    def rhs(t, state):
        acceleration, residual, retained, curvature_norm = acceleration_details(state[:n], state[n:])
        if retained > stop_threshold and residual > 1e-8 * max(1.0, curvature_norm):
            raise SingularMetricError(
                "Cartesian straight-line lift is unavailable: task curvature leaves the active tangent"
            )
        return np.concatenate((state[n:], acceleration))

    # Stepping the public DOP853 solver directly preserves every accepted state
    # even if a later stage fails. Dense polynomials also locate the rank event.
    segments = []
    final_t = 0.0
    success = False
    message = "Task geodesic integration did not complete"
    try:
        integrator = DOP853(rhs, 0.0, initial, duration, rtol=2e-10, atol=2e-12,
                            max_step=duration / 40.0)
        previous_singular = float(initial_s[rank - 1])
        accepted_steps = 0
        while integrator.status == "running":
            previous_t = float(integrator.t)
            step_message = integrator.step()
            if integrator.status == "failed":
                if previous_singular <= 10.0 * stop_threshold:
                    # Near a workspace fold, the remaining time scales with
                    # singular_value**2. Floating-point time resolution may
                    # be exhausted just before the requested event threshold.
                    message = (f"Stopped at t={previous_t:.8g}: end-effector task rank "
                               f"approached loss (retained singular value={previous_singular:.3g}; "
                               "adaptive time resolution exhausted)")
                else:
                    message = f"Task geodesic integration failed: {step_message}"
                break
            dense = integrator.dense_output()
            right_t = float(integrator.t)
            retained = singular_value(integrator.y[:n])
            segments.append((right_t, dense))
            final_t = right_t
            accepted_steps += 1
            if retained <= stop_threshold and previous_singular > stop_threshold:
                try:
                    final_t = brentq(
                        lambda time: singular_value(dense(time)[:n]) - stop_threshold,
                        previous_t, right_t, xtol=1e-13,
                    )
                except ValueError:
                    # At machine precision the dense endpoint may differ from
                    # the accepted state enough to erase a tiny event bracket.
                    final_t = right_t
                message = (f"Stopped at t={final_t:.8g}: end-effector task rank approached loss "
                           f"(retained singular value={stop_threshold:g})")
                break
            previous_singular = retained
            if accepted_steps >= 20000:
                message = "Task geodesic integration stopped after excessive adaptive refinement"
                break
            if integrator.status == "finished":
                success = True
                message = (f"Endpoint-only geodesic, task rank {rank}; Euclidean minimum-norm "
                           "acceleration lift with zero null-space acceleration")
    except (SingularMetricError, FloatingPointError, np.linalg.LinAlgError) as error:
        message = str(error)
    if not segments:
        return finish(np.array([0.0]), initial[:, None], False, message)
    t = np.linspace(0.0, final_t, samples)
    states = np.empty((2 * n, samples))
    states[:, 0] = initial
    left_t = 0.0
    for right_t, dense in segments:
        mask = (t >= left_t) & (t <= right_t)
        if np.any(mask):
            states[:, mask] = dense(t[mask])
        left_t = right_t
    return finish(t, states, success, message)


def solve_boundary(metric, q0, q1, samples=241, knots=41):
    """Solve a local geodesic connecting two joint configurations over t in [0,1].

    A collocation BVP solver starts from a straight joint interpolation. If this
    fails, endpoint continuation gradually extends a shorter geodesic. The
    result is a stationary local path; global shortest-path optimality is not
    asserted, and joint limits or obstacles are not imposed by this solver.
    """
    n = metric.robot.dof
    q0 = _vector(q0, n, "q0")
    q1 = _vector(q1, n, "q1")
    samples = _sample_count(samples)
    knots = _sample_count(knots)
    grid = np.linspace(0.0, 1.0, knots)
    delta = q1 - q0

    def rhs(t, state):
        accelerations = metric.acceleration_batch(state[:n].T, state[n:].T).T
        return np.vstack((state[n:], accelerations))

    def run(target, guess):
        def boundary(left, right):
            return np.concatenate((left[:n] - q0, right[:n] - target))

        return solve_bvp(rhs, boundary, grid, guess,
                         tol=2e-5, max_nodes=1800, bc_tol=1e-9)

    guess = np.vstack((q0[:, None] + delta[:, None] * grid,
                       np.repeat(delta[:, None], knots, axis=1)))
    try:
        solution = run(q1, guess)
        if not solution.success:
            previous = None
            previous_fraction = 0.0
            for fraction in (0.25, 0.5, 0.75, 1.0):
                target = q0 + fraction * delta
                if previous is None:
                    step_delta = target - q0
                    stage_guess = np.vstack((q0[:, None] + step_delta[:, None] * grid,
                                             np.repeat(step_delta[:, None], knots, axis=1)))
                else:
                    stage_guess = previous.sol(grid)
                    factor = fraction / previous_fraction
                    stage_guess[:n] = q0[:, None] + factor * (stage_guess[:n] - q0[:, None])
                    stage_guess[n:] *= factor
                solution = run(target, stage_guess)
                if not solution.success:
                    break
                previous, previous_fraction = solution, fraction
    except (SingularMetricError, FloatingPointError) as error:
        q = guess[:n].T
        v = guess[n:].T
        return _result(metric, grid, q, v, False, str(error))

    t = np.linspace(0.0, 1.0, samples)
    state = solution.sol(t)
    endpoint_error = max(np.linalg.norm(state[:n, 0] - q0),
                         np.linalg.norm(state[:n, -1] - q1))
    success = solution.success and endpoint_error < 1e-7
    message = (
        f"Collocation converged in {solution.niter} iterations; local fixed-endpoint geodesic"
        if success else f"Boundary solve did not converge: {solution.message}"
    )
    return _result(metric, t, state[:n].T, state[n:].T, success, message, endpoint_error)
