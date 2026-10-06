"""Differential geometry for serial revolute robots.

Joint axes live in the preceding link's coordinate frame. A joint rotates that
frame around its local axis, then its link offset is applied in the rotated
frame. The embedding consists of the positions of all moving link endpoints.
"""

from dataclasses import dataclass, field

import numpy as np


class SingularMetricError(ValueError):
    """The pullback metric cannot uniquely determine joint acceleration."""


def _vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return vector


def _rotation(axis, angle):
    """Rodrigues rotation for an already normalized axis."""
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    cosine, sine = np.cos(angle), np.sin(angle)
    return cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis, axis) + sine * cross


@dataclass
class Robot:
    name: str
    axes: np.ndarray
    offsets: np.ndarray
    base: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.axes = np.array(self.axes, dtype=float, copy=True)
        self.offsets = np.array(self.offsets, dtype=float, copy=True)
        if self.axes.ndim != 2 or self.axes.shape[1] != 3 or len(self.axes) == 0:
            raise ValueError("axes must have shape (number of joints, 3)")
        if self.offsets.shape != self.axes.shape:
            raise ValueError("offsets must have the same shape as axes")
        if not np.all(np.isfinite(self.axes)) or not np.all(np.isfinite(self.offsets)):
            raise ValueError("axes and offsets must be finite")
        norms = np.linalg.norm(self.axes, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("joint axes must be nonzero")
        self.axes /= norms[:, None]
        self.base = _vector(self.base, 3, "base").copy()

    @property
    def dof(self):
        return len(self.axes)

    def _embedding(self, q, v=None):
        q = _vector(q, self.dof, "q")
        if v is not None:
            v = _vector(v, self.dof, "v")
        points = np.empty((self.dof + 1, 3))
        points[0] = self.base
        world_axes = np.empty((self.dof, 3))
        orientation = np.eye(3)
        hessian_vv = np.empty((self.dof, 3)) if v is not None else None
        if v is not None:
            omega = np.zeros(3)
            alpha = np.zeros(3)
            point_acceleration = np.zeros(3)
        for i in range(self.dof):
            axis = orientation @ self.axes[i]
            world_axes[i] = axis
            orientation = orientation @ _rotation(self.axes[i], q[i])
            displacement = orientation @ self.offsets[i]
            points[i + 1] = points[i] + displacement
            if v is not None:
                # Differentiate FK twice along q(t) = q + t*v. Joint
                # accelerations vanish, but upstream axes still rotate.
                alpha = alpha + np.cross(omega, axis) * v[i]
                omega = omega + axis * v[i]
                point_acceleration = (
                    point_acceleration
                    + np.cross(alpha, displacement)
                    + np.cross(omega, np.cross(omega, displacement))
                )
                hessian_vv[i] = point_acceleration
        jacobians = np.zeros((self.dof, 3, self.dof))
        for endpoint in range(self.dof):
            arms = points[endpoint + 1] - points[: endpoint + 1]
            jacobians[endpoint, :, : endpoint + 1] = np.cross(
                world_axes[: endpoint + 1], arms
            ).T
        return points, jacobians, hessian_vv

    def points(self, q):
        """Return base and consecutive link endpoints, shape (n+1, 3)."""
        return self._embedding(q)[0]

    def point_jacobians(self, q):
        """Return endpoint derivatives, indexed [endpoint, xyz, joint]."""
        return self._embedding(q)[1]

    def directional_accelerations(self, q, v):
        """Return exact Hessian contractions H_i(q)[v,v] for each endpoint."""
        return self._embedding(q, v)[2]

    def _embedding_batch(self, q, v):
        """Vectorized embedding derivatives for collocation solver nodes."""
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        if q.ndim != 2 or q.shape[1] != self.dof or v.shape != q.shape:
            raise ValueError("batched q and v must have matching shape (batch, dof)")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
            raise ValueError("batched q and v must be finite")
        count = len(q)
        points = np.empty((count, self.dof + 1, 3))
        points[:, 0] = self.base
        world_axes = np.empty((count, self.dof, 3))
        orientation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        hessian_vv = np.empty((count, self.dof, 3))
        omega = np.zeros((count, 3))
        alpha = np.zeros((count, 3))
        point_acceleration = np.zeros((count, 3))
        for i in range(self.dof):
            axis = np.einsum("bij,j->bi", orientation, self.axes[i])
            world_axes[:, i] = axis
            x, y, z = self.axes[i]
            skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            cosine = np.cos(q[:, i])[:, None, None]
            sine = np.sin(q[:, i])[:, None, None]
            local_rotation = (
                cosine * np.eye(3)
                + (1.0 - cosine) * np.outer(self.axes[i], self.axes[i])
                + sine * skew
            )
            orientation = np.einsum("bij,bjk->bik", orientation, local_rotation)
            displacement = np.einsum("bij,j->bi", orientation, self.offsets[i])
            points[:, i + 1] = points[:, i] + displacement
            alpha = alpha + np.cross(omega, axis) * v[:, i, None]
            omega = omega + axis * v[:, i, None]
            point_acceleration = (
                point_acceleration + np.cross(alpha, displacement)
                + np.cross(omega, np.cross(omega, displacement))
            )
            hessian_vv[:, i] = point_acceleration
        jacobians = np.zeros((count, self.dof, 3, self.dof))
        for endpoint in range(self.dof):
            arms = points[:, endpoint + 1, None, :] - points[:, : endpoint + 1]
            jacobians[:, endpoint, :, : endpoint + 1] = np.cross(
                world_axes[:, : endpoint + 1], arms
            ).swapaxes(1, 2)
        return jacobians, hessian_vv


@dataclass
class Metric:
    robot: Robot
    weights: np.ndarray
    ridge: float = 0.0

    def __post_init__(self):
        self.weights = _vector(self.weights, self.robot.dof, "weights").copy()
        self.ridge = float(self.ridge)
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative")
        if not np.any(self.weights > 0) and self.ridge == 0.0:
            raise ValueError("at least one weight or ridge must be positive")

    def _matrix_from_jacobians(self, jacobians):
        matrix = np.einsum("icj,i,ick->jk", jacobians, self.weights, jacobians)
        matrix.flat[:: self.robot.dof + 1] += self.ridge
        return matrix

    def matrix(self, q):
        """g = sum_i w_i J_i.T J_i + ridge I."""
        return self._matrix_from_jacobians(self.robot.point_jacobians(q))

    def acceleration(self, q, v):
        """Compute -g^-1 sum_i w_i J_i.T H_i[v,v].

        This is the Levi-Civita geodesic equation for the weighted Cartesian
        embedding (augmented by sqrt(ridge)*q). It avoids finite differences
        of Christoffel symbols while preserving the same differential geometry.
        """
        v = _vector(v, self.robot.dof, "v")
        _, jacobians, hessian_vv = self.robot._embedding(q, v)
        matrix = self._matrix_from_jacobians(jacobians)
        if self.ridge == 0.0:
            eigenvalues = np.linalg.eigvalsh(matrix)
            if eigenvalues[0] <= 1e-13 * max(1.0, eigenvalues[-1]):
                raise SingularMetricError(
                    "Singular pullback metric: select more link endpoints or add a positive ridge"
                )
        force = np.einsum("icj,i,ic->j", jacobians, self.weights, hessian_vv)
        try:
            acceleration = np.linalg.solve(matrix, -force)
        except np.linalg.LinAlgError as error:
            raise SingularMetricError("Singular pullback metric") from error
        if not np.all(np.isfinite(acceleration)):
            raise SingularMetricError("Geodesic acceleration became nonfinite")
        return acceleration

    def energy(self, q, v):
        """Affine geodesic energy E = (1/2) v.T g(q) v."""
        v = _vector(v, self.robot.dof, "v")
        return float(0.5 * v @ self.matrix(q) @ v)

    def acceleration_batch(self, q, v):
        """Evaluate accelerations with shape (batch, dof), without node loops."""
        jacobians, hessian_vv = self.robot._embedding_batch(q, v)
        matrix = np.einsum("bicj,i,bick->bjk", jacobians, self.weights, jacobians)
        diagonal = np.arange(self.robot.dof)
        matrix[:, diagonal, diagonal] += self.ridge
        if self.ridge == 0.0:
            eigenvalues = np.linalg.eigvalsh(matrix)
            if np.any(eigenvalues[:, 0] <= 1e-13 * np.maximum(1.0, eigenvalues[:, -1])):
                raise SingularMetricError(
                    "Singular pullback metric: select more link endpoints or add a positive ridge"
                )
        force = np.einsum("bicj,i,bic->bj", jacobians, self.weights, hessian_vv)
        try:
            acceleration = np.linalg.solve(matrix, -force[..., None])[..., 0]
        except np.linalg.LinAlgError as error:
            raise SingularMetricError("Singular pullback metric") from error
        if not np.all(np.isfinite(acceleration)):
            raise SingularMetricError("Geodesic acceleration became nonfinite")
        return acceleration
