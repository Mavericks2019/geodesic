# 机械臂测地线动画实验台 · ImGui + OpenGL

基于 [Mavericks2019/geodesic](https://github.com/Mavericks2019/geodesic) 的微分几何计算方法，将原始两关节 MATLAB 算法移植到 Python，并测试复杂的串联旋转关节机械臂。**默认保持原算法的末端位置能量，因此末端沿匀速直线运动。** 对冗余关节用 SVD 选择一条连续的关节运动；全连杆能量作为单独的对照模式。动画使用 Dear ImGui 控制面板、GLFW 窗口和 OpenGL 3.3 Core 三维渲染。原仓库四个 `.m` 文件保持原样；参考版本为 `2203fb21e332f038cdf7f554d255656e5d8340a1`。

## 打开动画

首次使用先双击 **`install-runtime.bat`** 安装依赖，再双击 **`open-animation.bat`** 打开原生动画窗口。启动脚本使用项目内的 `.venv` 环境。也可以从 PowerShell 启动：

```powershell
cd D:\projects\geodesic
.\.venv\Scripts\python.exe viewer.py
```

窗口可切换 **原始末端能量（直线）/ 全连杆能量（对照）**，并提供机械臂场景选择、播放/暂停、播放速度、时间拖动、关节角线性插值对比和轨迹残影。三维视图支持拖动环绕和滚轮缩放，并可切换俯视、侧视；空格键切换播放，默认 0.25 倍速便于观察。左侧展开的计算方法与模型说明介绍算法和范围。首次启动会自动计算两种模式的轨迹并保存到 `output/trajectories.json`，计算完成后打开窗口。后续播放复用这些数据，无需重新求解，也无需联网或 MATLAB。

在另一台机器上首次安装依赖：

```powershell
cd D:\projects\geodesic
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe viewer.py
```

建议使用 Python 3.12。界面绑定使用 `imgui-bundle==1.92.900`；GLFW 与 PyOpenGL 提供原生窗口和图形接口，NumPy 与 SciPy 负责计算。需要支持 OpenGL 3.3 的图形驱动。Pillow 用于保存渲染截图。生成的轨迹、报告和截图留在本地 `output/`，不纳入版本控制。

重新计算全部轨迹与验证报告：

```powershell
.\.venv\Scripts\python.exe run.py --build --no-open
```

轨迹输出到 `output/trajectories.json`，验证报告输出到 `output/validation.json`。省略 `--no-open` 会在计算后启动原生窗口。`--samples 481` 可以提高输出采样密度。

运行独立数值测试和图形自检：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe viewer.py --self-test --screenshot output/opengl-preview.png
```

图形自检创建实际的隐藏 GLFW / OpenGL 上下文，绘制机械臂与 ImGui 界面，再从渲染结果读取截图。当前独立数值测试 **19 项全部通过**；默认末端模式的场景验证 **48 / 48 通过**，全连杆对照 **43 / 43 通过**。图形自检检查每种模式各组起点、中点与终点的 OpenGL 错误及机械臂/轨迹颜色像素，报告写入 `output/render-validation.json`；默认预览另有一帧 7R 中间姿态。

## 计算方法与原仓库的关系

原始平面两连杆的相对关节角为 `q = [p, q₂]`，连杆长度均为 1，末端位置为

```text
f(q) = [cos(p)+cos(p+q₂), sin(p)+sin(p+q₂)]
g(q) = J(q)ᵀJ(q)
     = [[2+2cos(q₂), 1+cos(q₂)], [1+cos(q₂), 1]]
```

`step1.m` 由度量计算 Christoffel 符号，`geodesic.m` 是展开后的测地线微分方程。本项目的 `original_acceleration` 逐项复现这个方程。默认模式保持它所使用的末端位置能量：

```text
f(q) = 机械臂末端位置
J(q) = ∂f/∂q
g(q) = J(q)ᵀJ(q)
E = ½ q̇ᵀg(q)q̇ = ½ ||J(q)q̇||² = ½ ||ḟ||²
```

原始两关节在非奇异姿态下，雅可比对二维末端位置可逆，测地线方程等价于 `f̈ = Jq̈ + H[q̇,q̇] = 0`。因此末端是匀速直线；复杂机械臂的默认模式也保持这项能量，没有加入中间连杆运动或关节正则项。

当关节数多于任务空间维度时，`g = JᵀJ` 是半正定矩阵，不能按普通逆矩阵唯一确定关节加速度。末端位置能量也不惩罚雅可比零空间中的运动。本项目明确选择以下 SVD 伪逆提升，把任务空间中的直线转换成一条关节轨迹：

```text
c = (f_target − f(q0)) / T
q̇(0) = J(q0)⁺ c
q̈(t) = −J(q(t))⁺ H(q(t))[q̇(t),q̇(t)]
f(q(t)) = f(q0) + c t
```

`J⁺` 为 Moore–Penrose 伪逆。初始速度选满足末端速度的最小范数解；每一步加速度选满足零末端加速度的最小范数解，额外的零空间加速度设为零。**这是一条明确选择的提升，不是冗余关节空间里唯一的黎曼测地线。** 在雅可比维持任务秩且直线可达的区间，末端应保持匀速直线，能量恒为 `½||c||²`。雅可比秩下降或请求的速度不在可实现方向内时，求解器明确失败或提前终止。

默认复杂场景用参考关节姿态的正运动学得到 `f_target`，仅约束末端位置；实际终点关节角由上述提升决定，通常不同于参考姿态。线性插值对照连接同一组实际起终关节角，并使用相同时间区间和同一项末端能量比较。

**全连杆能量对照**对复杂机械臂使用另外一套目标函数。2R 原始复现作为基准，在两种模式中保持相同的原始方程。令 `fᵢ(q)` 为第 i 根连杆末端的三维位置，`Jᵢ` 为其雅可比，`Hᵢ[v,v]` 为沿速度方向的二阶导数，复杂机械臂对照的度量与方程为

```text
g(q) = Σᵢ wᵢ Jᵢ(q)ᵀJᵢ(q) + λI
q̈ = −g(q)⁻¹ Σᵢ wᵢ Jᵢ(q)ᵀHᵢ(q)[q̇,q̇]
E = ½ q̇ᵀg(q)q̇
L = ∫ sqrt(q̇ᵀg(q)q̇) dt
```

在该对照模式中，中间连杆运动也有代价，正则项等价于在嵌入中追加 `sqrt(λ) q`，正的 λ 使度量保持正定。此时二阶嵌入形式与 Christoffel 方程等价，但末端单独的加速度不要求为零，末端轨迹可以弯曲。权重与 λ 是这一替代目标的几何设计参数，不是已识别的连杆质量或惯量；两种模式显示的能量都不代表电机实际能耗。

默认末端模式与原始两关节复现采用 DOP853 初值积分；复杂机械臂的全连杆对照采用 SciPy 配点边值求解，固定起点和终点的关节角，失败时用终点连续延拓重试。对照输出收敛的局部测地线，没有声称得到全局最短路径。两种模式的雅可比与二阶方向导数均用解析递推；有限差分只用于独立验证。

原始 `step4.m` 的 3 秒积分存在明确的可达性问题：从末端 `(1,0)` 出发，初速度约为 `(-0.707,0.707)`，沿直线在约 **2.5783 秒** 达到半径为 2 的伸直位置。`det(g)=sin²(q₂)` 此时为零，折叠位置同样奇异。默认复现到 2 秒；数值测试独立检查 3 秒复现的提前终止，以及冗余空间臂奇异度量的明确失败。

## 测试场景

| 场景 | 自由度 | 主要验证内容 |
| --- | ---: | --- |
| 原始算法复现 | 2 | 原 MATLAB 初始角与速度；末端匀速直线；能量守恒 |
| 冗余平面臂 | 3 | 二维末端直线的冗余关节提升 |
| 多连杆平面臂 | 5 | 多关节协调实现末端匀速直线 |
| 空间工业臂结构 | 6 | 偏航、俯仰、滚转轴组合；三维末端直线 |
| 冗余空间臂 | 7 | 三维任务的 SVD 提升与零空间选择 |
| 空间蛇形臂 | 10 | 高维冗余运动学、末端直线与能量守恒 |

默认模式检查数据有限、末端端点误差、末端直线误差、匀速性、末端能量守恒、任务秩、非零任务奇异值和方程残差。冗余臂 `g = JᵀJ` 有零特征值是预期结果，不以正定性作为默认模式的通过条件。全连杆对照另外检查正定度量、关节端点误差及相对于线性插值的积分代价。`tests/test_geometry.py` 使用有限差分和解析两连杆公式，独立验证 6 / 7 / 10 自由度末端直线、SVD 初始速度、零空间加速度、解析运动学、Christoffel 方程、能量守恒、奇异终止和边值求解。

上述模型是理想串联旋转关节模型，长度使用米，角度使用弧度，空间场景 z 轴向上。求解中未施加障碍物、自碰撞、关节限位、末端姿态约束、电机动力学或速度/加速度上限，不能直接作为真实机械臂控制指令。

## 添加自己的机械臂

参考 `examples/custom-4r.json`，为每个关节提供一个局部旋转轴 `axes[i]` 和一根连杆偏移向量 `offsets[i]`。转换顺序是：在上一连杆坐标系中绕局部轴旋转，再在旋转后的坐标系中施加偏移。旋转轴会自动归一化。`q0` 是起始相对关节角；默认末端模式用 `q1` 的末端位置作为目标，实际终点关节角由提升决定。全连杆对照中 `q1` 才是固定的终点关节角。

```powershell
.\.venv\Scripts\python.exe run.py --robot examples/custom-4r.json
```

自定义计算会将 `output/trajectories.json` 替换为该机械臂的两种能量模式轨迹，并启动原生窗口。恢复六个预设场景及其全连杆对照：

```powershell
.\.venv\Scripts\python.exe run.py --build
```

可配置字段：`base` 为固定底座坐标，`duration` 为动画的仿真时长。`weights` 与 `ridge` 用于全连杆对照模式，分别是连杆末端的非负权重和非负正则参数；默认末端能量不包含这些附加项。全连杆对照建议保持正的 `ridge`。角度按提供的数值使用，不自动改成模 2π 的另一条路径。

## 文件组织

```text
geodesic.m / step*.m     原始 MATLAB 文件
geodesic_lab/geometry.py 运动学、雅可比、二阶方向导数、度量
geodesic_lab/solver.py   原始方程、SVD 任务直线提升、全连杆边值求解
geodesic_lab/scenarios.py 六个复杂度递增的场景与自定义模型
geodesic_lab/export.py   验证与轨迹数据导出
tests/test_geometry.py  独立数值/解析回归测试
viewer.py               ImGui 控件与 OpenGL 实时三维渲染
open-animation.bat      Windows 原生动画启动脚本
output/trajectories.json 预计算动画数据
output/validation.json  两种能量模式的计算验证报告
requirements.txt        Python 计算与图形依赖
run.py                  重算与导出入口
```
