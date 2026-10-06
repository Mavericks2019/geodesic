"""Independent numerical checks for the differential geometry and path solvers.

Run from the repository root with: python -m unittest discover -s tests -v
The finite differences here operate on positions/metrics, independently of the
analytic Jacobian and acceleration implementations under test.
"""

import unittest

import numpy as np

from geodesic_lab.geometry import Metric, Robot
from geodesic_lab.solver import (
    integrate_geodesic,
    integrate_original,
    integrate_task_geodesic,
    original_acceleration,
    solve_boundary,
)


def planar_robot():
    return Robot(
        "two planar unit links",
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )


def spatial_robot(dof):
    axes = np.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [0, 1, 0], [1, 0, 0]], dtype=float
    )[:dof]
    offsets = np.array(
        [[0.0, 0.0, 0.42], [0.43, 0.03, 0.0], [0.32, 0.0, 0.06],
         [0.25, -0.04, 0.0], [0.18, 0.0, 0.08], [0.16, 0.03, 0.0],
         [0.13, 0.0, 0.02], [0.12, -0.02, 0.01], [0.11, 0.02, 0.0],
         [0.10, 0.0, 0.03]], dtype=float
    )[:dof]
    return Robot("spatial test arm", axes, offsets, np.array([0.3, -0.2, 0.4]))


def finite_difference_jacobians(robot, q, step=1e-6):
    jac = np.empty((robot.points(q).shape[0] - 1, 3, len(q)))
    for j in range(len(q)):
        delta = np.zeros_like(q)
        delta[j] = step
        jac[:, :, j] = (robot.points(q + delta)[1:] - robot.points(q - delta)[1:]) / (2 * step)
    return jac


def metric_christoffel_acceleration(metric, q, v, step=1e-6):
    derivatives = np.empty((len(q), len(q), len(q)))
    for j in range(len(q)):
        delta = np.zeros_like(q)
        delta[j] = step
        derivatives[j] = (metric.matrix(q + delta) - metric.matrix(q - delta)) / (2 * step)
    force = np.einsum("jlk,j,k->l", derivatives, v, v)
    force -= 0.5 * np.einsum("ljk,j,k->l", derivatives, v, v)
    return -np.linalg.solve(metric.matrix(q), force)


class KinematicsTests(unittest.TestCase):
    def test_two_link_coordinates_match_original_matlab(self):
        robot = planar_robot()
        q = np.array([np.pi / 3, -2 * np.pi / 3])
        np.testing.assert_allclose(
            robot.points(q),
            [[0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [1, 0, 0]],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            robot.point_jacobians(q)[-1],
            [[0, np.sqrt(3) / 2], [1, 0.5], [0, 0]],
            atol=1e-14,
        )

    def test_spatial_six_and_seven_axis_jacobians_against_positions(self):
        rng = np.random.default_rng(7419)
        for dof in (6, 7):
            robot = spatial_robot(dof)
            for _ in range(3):
                q = rng.uniform(-1.2, 1.2, dof)
                with self.subTest(dof=dof, q=q):
                    np.testing.assert_allclose(
                        robot.point_jacobians(q),
                        finite_difference_jacobians(robot, q),
                        rtol=2e-6, atol=5e-8,
                    )

    def test_spatial_directional_hessians_against_positions(self):
        rng = np.random.default_rng(2391)
        step = 2e-4
        for dof in (6, 7):
            robot = spatial_robot(dof)
            for _ in range(3):
                q = rng.uniform(-1.0, 1.0, dof)
                v = rng.uniform(-0.4, 0.4, dof)
                numerical = (
                    robot.points(q + step * v)[1:]
                    - 2 * robot.points(q)[1:]
                    + robot.points(q - step * v)[1:]
                ) / step**2
                with self.subTest(dof=dof, q=q):
                    np.testing.assert_allclose(
                        robot.directional_accelerations(q, v), numerical,
                        rtol=1e-5, atol=1e-7,
                    )


class RiemannianMetricTests(unittest.TestCase):
    def test_original_metric_and_determinant(self):
        robot = planar_robot()
        metric = Metric(robot, np.array([0.0, 1.0]), ridge=0.0)
        for elbow in (-2.2, -0.7, 0.4, 1.8):
            q = np.array([0.31, elbow])
            c = np.cos(elbow)
            expected = np.array([[2 + 2 * c, 1 + c], [1 + c, 1]])
            np.testing.assert_allclose(metric.matrix(q), expected, atol=1e-13)
            self.assertAlmostEqual(np.linalg.det(metric.matrix(q)), np.sin(elbow)**2, places=12)

    def test_weighted_metric_matches_finite_difference_pullback(self):
        robot = spatial_robot(7)
        q = np.array([0.3, -0.7, 0.2, 0.8, -0.6, 0.1, -0.4])
        weights = np.linspace(0.2, 1.0, 7)
        metric = Metric(robot, weights, ridge=0.035)
        numerical_j = finite_difference_jacobians(robot, q)
        expected = np.einsum("i,iaj,iak->jk", weights, numerical_j, numerical_j)
        expected += 0.035 * np.eye(7)
        np.testing.assert_allclose(metric.matrix(q), expected, rtol=1e-7, atol=5e-9)

    def test_ridge_regularizes_redundant_end_effector_metric(self):
        robot = spatial_robot(7)
        q = np.array([0.2, -0.5, 0.6, -0.4, 0.3, -0.7, 0.8])
        weights = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
        unregularized = Metric(robot, weights, ridge=0.0).matrix(q)
        self.assertLessEqual(np.linalg.matrix_rank(unregularized, tol=1e-9), 3)
        regularized = Metric(robot, weights, ridge=0.05).matrix(q)
        self.assertGreaterEqual(np.linalg.eigvalsh(regularized)[0], 0.05 - 1e-12)
        np.linalg.cholesky(regularized)

    def test_acceleration_matches_metric_derivative_christoffel_formula(self):
        robot = spatial_robot(6)
        metric = Metric(robot, np.linspace(0.2, 1.0, 6), ridge=0.035)
        q = np.array([0.4, -0.7, 0.3, 0.6, -0.2, 0.5])
        v = np.array([0.3, -0.2, 0.4, -0.35, 0.2, 0.15])
        np.testing.assert_allclose(
            metric.acceleration(q, v), metric_christoffel_acceleration(metric, q, v),
            rtol=2e-6, atol=2e-8,
        )

    def test_original_rhs_matches_general_geodesic_acceleration(self):
        metric = Metric(planar_robot(), np.array([0.0, 1.0]), ridge=0.0)
        rng = np.random.default_rng(507)
        for elbow in (-2.7, -1.2, -0.2, 0.2, 0.8, 2.7):
            q = np.array([rng.uniform(-2, 2), elbow])
            v = rng.uniform(-0.8, 0.8, 2)
            with self.subTest(elbow=elbow):
                np.testing.assert_allclose(
                    metric.acceleration(q, v), original_acceleration(q, v),
                    rtol=2e-12, atol=2e-12,
                )

    def test_batch_acceleration_matches_scalar_and_metric_derivatives(self):
        rng = np.random.default_rng(1952)
        for dof in (6, 7):
            robot = spatial_robot(dof)
            # Oblique joint axes also exercise the general Rodrigues rotations.
            robot = Robot(robot.name, robot.axes + rng.uniform(-0.2, 0.2, (dof, 3)),
                          robot.offsets, robot.base)
            metric = Metric(robot, np.linspace(0.2, 1.0, dof), ridge=0.035)
            qs = rng.uniform(-1.1, 1.1, (5, dof))
            vs = rng.uniform(-0.5, 0.5, (5, dof))
            batch = metric.acceleration_batch(qs, vs)
            scalar = np.array([metric.acceleration(q, v) for q, v in zip(qs, vs)])
            independent = np.array([
                metric_christoffel_acceleration(metric, q, v)
                for q, v in zip(qs, vs)
            ])
            with self.subTest(dof=dof):
                np.testing.assert_allclose(batch, scalar, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(batch, independent, rtol=2e-6, atol=3e-8)


class GeodesicSolverTests(unittest.TestCase):
    def test_original_three_second_demo_stops_at_reachable_workspace_edge(self):
        robot = planar_robot()
        q0 = np.array([np.pi / 3, -2 * np.pi / 3])
        v0 = np.array([1.1152, -0.8164])
        result = integrate_original(q0, v0, duration=3.0, samples=121)
        initial_tip = robot.points(q0)[-1]
        task_velocity = robot.point_jacobians(q0)[-1] @ v0
        roots = np.roots([
            task_velocity @ task_velocity,
            2 * (initial_tip @ task_velocity),
            initial_tip @ initial_tip - 4.0,
        ])
        expected_stop = max(roots)
        self.assertFalse(result.success)
        self.assertGreater(len(result.t), 1)
        self.assertAlmostEqual(result.t[-1], expected_stop, places=4)
        self.assertTrue(np.isfinite(result.q).all())
        self.assertTrue(np.isfinite(result.v).all())
        self.assertIn("singular", result.message.lower())

    def test_singular_boundary_metric_returns_explicit_failure(self):
        robot = spatial_robot(7)
        metric = Metric(robot, np.array([0, 0, 0, 0, 0, 0, 1]), ridge=0.0)
        q0 = np.array([0.2, -0.5, 0.6, -0.4, 0.3, -0.7, 0.8])
        q1 = q0 + 0.1
        result = solve_boundary(metric, q0, q1, samples=31, knots=11)
        self.assertFalse(result.success)
        self.assertIn("singular", result.message.lower())
        self.assertTrue(np.isfinite(result.q).all())
        self.assertTrue(np.isfinite(result.v).all())

    def test_original_pullback_geodesic_is_straight_in_task_space(self):
        robot = planar_robot()
        metric = Metric(robot, np.array([0.0, 1.0]), ridge=0.0)
        q0 = np.array([np.pi / 3, -2 * np.pi / 3])
        v0 = np.array([1.1152, -0.8164])
        result = integrate_geodesic(metric, q0, v0, duration=1.0, samples=121)
        self.assertTrue(result.success, result.message)
        positions = np.array([robot.points(q)[-1] for q in result.q])
        expected = robot.points(q0)[-1] + result.t[:, None] * (robot.point_jacobians(q0)[-1] @ v0)
        np.testing.assert_allclose(positions, expected, rtol=1e-6, atol=2e-6)

    def test_regularized_spatial_geodesic_conserves_energy(self):
        robot = spatial_robot(7)
        metric = Metric(robot, np.linspace(0.2, 1.0, 7), ridge=0.035)
        q0 = np.array([0.2, -0.5, 0.6, -0.4, 0.3, -0.7, 0.8])
        v0 = np.array([0.25, -0.2, 0.15, 0.1, -0.15, 0.2, -0.1])
        result = integrate_geodesic(metric, q0, v0, duration=1.2, samples=121)
        self.assertTrue(result.success, result.message)
        energies = np.array([metric.energy(q, v) for q, v in zip(result.q, result.v)])
        relative_drift = np.max(np.abs(energies / energies[0] - 1.0))
        self.assertLess(relative_drift, 2e-5)
        self.assertGreaterEqual(result.min_eigenvalue, 0.035 - 1e-12)

    def test_fixed_endpoint_spatial_path_reduces_action(self):
        robot = spatial_robot(6)
        metric = Metric(robot, np.linspace(0.2, 1.0, 6), ridge=0.035)
        q0 = np.array([-0.6, 0.4, -0.35, 0.2, -0.3, 0.15])
        q1 = np.array([0.75, -0.6, 0.4, -0.45, 0.5, -0.3])
        result = solve_boundary(metric, q0, q1, samples=161, knots=31)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.q[0], q0, atol=1e-12)
        np.testing.assert_allclose(result.q[-1], q1, atol=1e-12)
        self.assertTrue(np.isfinite(result.q).all())
        self.assertTrue(np.isfinite(result.v).all())
        optimized_energy = np.array([metric.energy(q, v) for q, v in zip(result.q, result.v)])
        linear_q = q0 + result.t[:, None] * (q1 - q0)
        linear_energy = np.array([metric.energy(q, q1 - q0) for q in linear_q])
        dt = np.diff(result.t)
        optimized_action = np.sum(0.5 * (optimized_energy[1:] + optimized_energy[:-1]) * dt)
        linear_action = np.sum(0.5 * (linear_energy[1:] + linear_energy[:-1]) * dt)
        self.assertLess(optimized_action, linear_action * 0.999)


class TaskSpaceLiftTests(unittest.TestCase):
    """The original endpoint energy requires Cartesian affine straightness.

    For redundant arms this energy is semidefinite. These checks verify the
    explicitly selected SVD lift, rather than asserting a unique joint path.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        q0_all = np.array([0.3, -0.65, 0.35, 0.6, -0.4, 0.45, -0.25, -0.15, 0.3, -0.2])
        delta_all = np.array([0.12, -0.1, 0.08, -0.11, 0.06, -0.07, 0.09, -0.08, 0.06, 0.08])
        for dof in (6, 7, 10):
            robot = spatial_robot(dof)
            q0 = q0_all[:dof].copy()
            reference_q = q0 + delta_all[:dof]
            target = robot.points(reference_q)[-1]
            result = integrate_task_geodesic(robot, q0, target, duration=1.3, samples=241)
            cls.cases.append((robot, q0, reference_q, target, result))

    def test_spatial_lifts_have_straight_tips_constant_speed_and_original_energy(self):
        for robot, q0, reference_q, target, result in self.cases:
            with self.subTest(dof=robot.dof):
                self.assertTrue(result.success, result.message)
                self.assertAlmostEqual(result.t[-1], 1.3, places=12)
                self.assertEqual(result.task_rank, 3)
                self.assertGreater(result.min_task_singular_value, 1e-7)
                start = robot.points(q0)[-1]
                requested_velocity = (target - start) / 1.3
                actual_tips = np.array([robot.points(q)[-1] for q in result.q])
                expected_tips = start + result.t[:, None] * requested_velocity
                np.testing.assert_allclose(actual_tips, expected_tips, atol=2e-7, rtol=1e-7)
                np.testing.assert_allclose(actual_tips[-1], target, atol=2e-7)
                dt = result.t[1] - result.t[0]
                observed_velocity = (actual_tips[2:] - actual_tips[:-2]) / (2 * dt)
                np.testing.assert_allclose(
                    observed_velocity,
                    np.broadcast_to(requested_velocity, observed_velocity.shape),
                    atol=2e-7, rtol=2e-6,
                )
                expected_energy = 0.5 * (requested_velocity @ requested_velocity)
                np.testing.assert_allclose(result.energy, expected_energy, atol=2e-9, rtol=2e-6)
                measured_energy = np.array([
                    0.5 * np.linalg.norm(robot.point_jacobians(q)[-1] @ v)**2
                    for q, v in zip(result.q, result.v)
                ])
                np.testing.assert_allclose(result.energy, measured_energy, atol=2e-12)
                # The reference posture specifies only the Cartesian target.
                self.assertGreater(np.linalg.norm(result.q[-1] - reference_q), 1e-4)

    def test_spatial_lifts_use_projected_initial_velocity_and_zero_null_acceleration(self):
        step = 2e-4
        for robot, q0, reference_q, target, result in self.cases:
            with self.subTest(dof=robot.dof):
                self.assertTrue(result.success, result.message)
                requested_velocity = (target - robot.points(q0)[-1]) / 1.3
                initial_j = finite_difference_jacobians(robot, q0)[-1]
                expected_v0 = np.linalg.lstsq(initial_j, requested_velocity, rcond=1e-12)[0]
                np.testing.assert_allclose(result.v[0], expected_v0, atol=2e-8, rtol=2e-6)
                dt = result.t[1] - result.t[0]
                for index in (10, 55, 100, 150, 200):
                    q, v = result.q[index], result.v[index]
                    # Fourth-order differentiation of solver output, independent
                    # of its internal acceleration function.
                    observed_a = (
                        -result.v[index + 2] + 8 * result.v[index + 1]
                        - 8 * result.v[index - 1] + result.v[index - 2]
                    ) / (12 * dt)
                    j = finite_difference_jacobians(robot, q)[-1]
                    h_vv = (
                        robot.points(q + step * v)[-1] - 2 * robot.points(q)[-1]
                        + robot.points(q - step * v)[-1]
                    ) / step**2
                    np.testing.assert_allclose(j @ observed_a + h_vv, 0, atol=2e-7)
                    null_acceleration = (np.eye(robot.dof) - np.linalg.pinv(j) @ j) @ observed_a
                    np.testing.assert_allclose(null_acceleration, 0, atol=2e-7)

    def test_two_link_svd_lift_matches_original_matlab_trajectory(self):
        robot = planar_robot()
        q0 = np.array([np.pi / 3, -2 * np.pi / 3])
        v0 = np.array([1.1152, -0.8164])
        target = robot.points(q0)[-1] + robot.point_jacobians(q0)[-1] @ v0
        lifted = integrate_task_geodesic(robot, q0, target, duration=1.0, samples=121)
        original = integrate_original(q0, v0, duration=1.0, samples=121)
        self.assertTrue(lifted.success, lifted.message)
        np.testing.assert_allclose(lifted.q, original.q, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(lifted.v, original.v, atol=2e-8, rtol=2e-8)

    def test_singular_start_with_unrealizable_cartesian_velocity_fails_explicitly(self):
        robot = planar_robot()
        result = integrate_task_geodesic(
            robot, np.array([0.0, 0.0]), np.array([1.9, 0.1, 0.0]),
            duration=1.0, samples=61,
        )
        self.assertFalse(result.success)
        self.assertTrue(np.isfinite(result.q).all())
        self.assertTrue(np.isfinite(result.v).all())
        self.assertTrue(any(word in result.message.lower() for word in ("rank", "singular", "range", "reachable", "tangent")),
                        result.message)

    def test_task_lift_stops_on_runtime_rank_loss_and_keeps_accepted_path(self):
        robot = planar_robot()
        q0 = np.array([np.pi / 3, -2 * np.pi / 3])
        result = integrate_task_geodesic(
            robot, q0, np.array([1.0, 2.5, 0.0]), duration=1.0, samples=61,
        )
        self.assertFalse(result.success)
        self.assertEqual(len(result.t), 61)
        self.assertAlmostEqual(result.t[-1], np.sqrt(3) / 2.5, places=4)
        self.assertIn("rank", result.message.lower())
        self.assertTrue(np.isfinite(result.q).all())
        self.assertTrue(np.isfinite(result.v).all())
        observed_tip = np.array([robot.points(q)[-1] for q in result.q])
        expected_tip = np.column_stack((np.ones(61), 2.5 * result.t, np.zeros(61)))
        np.testing.assert_allclose(observed_tip, expected_tip, atol=2e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
