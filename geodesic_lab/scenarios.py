"""Reproducible robot geometries and endpoint experiments (SI units, radians)."""
from dataclasses import dataclass
import numpy as np

from .geometry import Robot, Metric


@dataclass
class Scenario:
    id: str
    name: str
    kind: str
    description: str
    robot: Robot
    metric: Metric
    q0: np.ndarray
    q1: np.ndarray | None = None
    v0: np.ndarray | None = None
    duration: float = 1.0
    original: bool = False


def planar(name, lengths):
    lengths = np.asarray(lengths, dtype=float)
    return Robot(name, np.tile([0., 0., 1.], (len(lengths), 1)),
                 np.column_stack((lengths, np.zeros((len(lengths), 2)))))


def distributed_metric(robot, ridge=0.035):
    # All intermediate link tips participate, with higher weight near the tool.
    return Metric(robot, np.linspace(0.25, 1.0, robot.dof), ridge)


def scenarios():
    r2 = planar("原仓库两连杆", [1., 1.])
    r3 = planar("三关节平面臂", [0.75, 0.6, 0.45])
    r5 = planar("五关节平面臂", [0.55, 0.45, 0.4, 0.3, 0.25])
    axes6 = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0],
                      [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    r6 = Robot("六关节空间臂", axes6,
               np.array([[0, 0, .38], [.45, 0, 0], [.37, 0, 0],
                         [.14, 0, 0], [.18, 0, 0], [.12, 0, 0]]),
               base=np.array([0., 0., .12]))
    r7 = Robot("七关节冗余空间臂",
               np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
                         [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
               np.array([[0, 0, .36], [.32, 0, 0], [.22, 0, .05],
                         [.30, 0, 0], [.18, 0, .04], [.18, 0, 0], [.12, 0, 0]]),
               base=np.array([0., 0., .12]))
    axes10 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
                          [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
                          [0, 0, 1], [0, 1, 0]], dtype=float)
    r10 = Robot("十关节空间蛇形臂", axes10,
                np.array([[.24, 0, .05], [.22, 0, 0], [.21, .03, 0],
                          [.20, 0, 0], [.19, 0, 0], [.18, 0, 0],
                          [.17, .02, 0], [.16, 0, 0], [.15, 0, 0], [.14, 0, 0]]),
                base=np.array([0., 0., .45]))
    return [
        Scenario("original-2r", "2R · 原始算法复现", "planar",
                 "保持原 MATLAB 的初始角度与速度，积分到 2 秒；末端应沿直线运动。",
                 r2, Metric(r2, np.array([0., 1.]), 0.),
                 np.array([np.pi / 3, -2 * np.pi / 3]),
                 v0=np.array([1.1152, -0.8164]), duration=2., original=True),
        Scenario("planar-3r", "3R · 冗余平面臂", "planar",
                 "三关节共享二维工作空间，使用全连杆度量求解给定起终姿态的测地线。",
                 r3, distributed_metric(r3), np.array([.3, -.85, .55]),
                 np.array([1.05, -.2, -.6])),
        Scenario("planar-5r", "5R · 多连杆平面臂", "planar",
                 "五个相对转角同时变化，比较测地线与关节角线性插值的全臂运动代价。",
                 r5, distributed_metric(r5), np.array([-.35, .8, -.65, .55, -.35]),
                 np.array([.85, -.45, .7, -.5, .45])),
        Scenario("spatial-6r", "6R · 空间工业臂结构", "spatial",
                 "包含底座偏航、肩肘俯仰和腕部滚转；使用理想串联运动学模型。",
                 r6, distributed_metric(r6), np.array([-.6, -.8, .65, -.4, -.55, .3]),
                 np.array([.65, -.35, -.45, .65, .4, -.55])),
        Scenario("spatial-7r", "7R · 冗余空间臂", "spatial",
                 "七个自由度、交替局部旋转轴，展示冗余结构中的全臂测地线运动。",
                 r7, distributed_metric(r7), np.array([-.65, -.75, .35, .8, -.45, -.6, .25]),
                 np.array([.65, -.35, -.5, -.45, .6, .25, -.4])),
        Scenario("spatial-10r", "10R · 空间蛇形臂", "spatial",
                 "十个关节交替绕三轴旋转，检验高维雅可比、二阶运动学与边值求解。",
                 r10, distributed_metric(r10),
                 np.array([-.4, -.25, .3, -.3, .45, -.2, -.3, .25, -.35, -.2]),
                 np.array([.45, .2, -.4, .25, -.3, .3, .35, -.2, .4, .25])),
    ]


def custom_scenario(spec):
    """Load an arbitrary serial revolute chain from a small JSON object."""
    robot = Robot(spec.get("name", "自定义机械臂"), np.array(spec["axes"], float),
                  np.array(spec["offsets"], float), np.array(spec.get("base", [0, 0, 0]), float))
    weights = np.array(spec.get("weights", np.linspace(.25, 1., robot.dof)), float)
    metric = Metric(robot, weights, float(spec.get("ridge", .035)))
    q0, q1 = np.array(spec["q0"], float), np.array(spec["q1"], float)
    if q0.shape != (robot.dof,) or q1.shape != (robot.dof,):
        raise ValueError("q0/q1 must each contain one angle per joint")
    if not np.all(np.isfinite(np.r_[q0, q1])):
        raise ValueError("Joint angles must be finite")
    return Scenario(spec.get("id", "custom"), robot.name, spec.get("kind", "spatial"),
                    spec.get("description", "自定义串联旋转关节机械臂"), robot, metric, q0, q1,
                    duration=float(spec.get("duration", 1.)))
