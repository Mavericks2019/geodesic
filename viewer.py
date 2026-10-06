"""Dear ImGui + OpenGL 3.3 desktop geodesic robot laboratory."""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys
import time
import traceback

# Import GLFW first to keep the Python backend on a single GLFW runtime.
import glfw
import numpy as np
from OpenGL import GL as gl
from imgui_bundle import imgui, implot
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from geodesic_lab.gl_renderer import Renderer

ROOT = Path(__file__).resolve().parent
CYAN = imgui.ImVec4(.28, .85, .83, 1)
AMBER = imgui.ImVec4(.96, .70, .35, 1)
MUTED = imgui.ImVec4(.55, .65, .77, 1)


def load_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("cases"):
        raise ValueError("No robot trajectories; run python run.py --build --no-open")
    for case in data["cases"] + data.get("comparison_cases", []):
        for key in ("t", "q", "points"):
            case[key] = np.asarray(case[key], dtype=np.float64)
        for key in ("q", "points"):
            case["baseline"][key] = np.asarray(case["baseline"][key], dtype=np.float64)
        for key in case["curve"]:
            case["curve"][key] = np.asarray(case["curve"][key], dtype=np.float64)
    return data


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def theme():
    imgui.style_colors_dark()
    style = imgui.get_style()
    style.window_padding = imgui.ImVec2(18, 14)
    style.frame_padding = imgui.ImVec2(9, 6)
    style.item_spacing = imgui.ImVec2(10, 9)
    style.window_rounding = 0
    style.child_rounding = 8
    style.frame_rounding = 5
    style.grab_rounding = 5
    style.window_border_size = 0
    style.child_border_size = 0
    colors = {
        "window_bg": (.045, .065, .105, 1), "child_bg": (.065, .09, .145, 1),
        "text": (.9, .93, .98, 1), "text_disabled": (.48, .58, .71, 1),
        "frame_bg": (.115, .16, .23, 1), "frame_bg_hovered": (.16, .23, .31, 1),
        "frame_bg_active": (.17, .28, .35, 1), "button": (.12, .20, .26, 1),
        "button_hovered": (.17, .35, .4, 1), "button_active": (.2, .43, .46, 1),
        "header": (.12, .29, .34, 1), "header_hovered": (.14, .32, .39, 1),
        "header_active": (.16, .38, .42, 1), "check_mark": (.28, .85, .83, 1),
        "slider_grab": (.28, .85, .83, 1), "slider_grab_active": (.36, .95, .9, 1),
        "separator": (.16, .21, .29, 1), "border": (.16, .22, .3, 1),
    }
    for name, color in colors.items():
        style.set_color_(getattr(imgui.Col_, name), imgui.ImVec4(*color))


class App:
    def __init__(self, window, data):
        self.window, self.data = window, data
        self.renderer = Renderer()
        self.mode = "endpoint"
        self.index = 0
        self.progress = 0.
        self.playing = True
        self.loop = True
        self.speed = .25
        self.show_baseline = True
        self.show_ghost = True
        self.status = "拖动视口旋转 · 滚轮缩放 · 空格播放/暂停"
        self.viewport_rect = None
        self.scene_texture = 0
        self.select(next((i for i, c in enumerate(self.cases) if c["id"] == "spatial-7r"), 0))

    @property
    def cases(self):
        return self.data.get("comparison_cases", self.data["cases"]) if self.mode == "whole" else self.data["cases"]

    @property
    def case(self):
        return self.cases[self.index]

    def select(self, index):
        self.index = index
        self.progress = 0.
        case = self.case
        pts = np.concatenate((case["points"].reshape(-1, 3), case["baseline"]["points"].reshape(-1, 3)))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.center = (lo + hi) / 2
        self.radius = max(float(np.linalg.norm(hi - lo) / 2), .5)
        self.floor_height = min(0., float(lo[2] - .08 * self.radius))
        self.camera = [-.95, .48, self.radius * 3.4]
        if case["kind"] == "planar":
            self.camera[:2] = [0., math.pi / 2 - .005]

    def tick(self, dt):
        if self.playing:
            duration = max(float(self.case["t"][-1] - self.case["t"][0]), 1e-9)
            self.progress += min(dt, .1) * self.speed / duration
            if self.progress >= 1:
                if self.loop:
                    self.progress %= 1
                else:
                    self.progress, self.playing = 1., False

    def sample(self, frames):
        location = np.clip(self.progress, 0., 1.) * (len(frames) - 1)
        first = int(location)
        last = min(first + 1, len(frames) - 1)
        weight = location - first
        return frames[first] * (1 - weight) + frames[last] * weight

    def export_case(self):
        path = ROOT / "output" / f"{self.mode}-{self.case['id']}-trajectory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.case, default=json_default, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status = f"已导出：{path.name}"

    def sidebar(self, height):
        imgui.begin_child("cases", imgui.ImVec2(225, height))
        imgui.text_colored(CYAN, "GEODESIC / 测地线")
        imgui.text("机械臂路径实验室")
        imgui.separator()
        imgui.text_colored(MUTED, "能量函数")
        imgui.set_next_item_width(-1)
        mode_labels = ["原始末端能量（直线）"]
        if self.data.get("comparison_cases"):
            mode_labels.append("全连杆能量（对照）")
        changed, choice = imgui.combo("##energy_mode", 0 if self.mode == "endpoint" else 1,
                                     mode_labels)
        if changed:
            self.mode = "endpoint" if choice == 0 else "whole"
            self.select(min(self.index, len(self.cases) - 1))
        imgui.text_colored(MUTED, "测试场景")
        for i, case in enumerate(self.cases):
            if imgui.selectable(case["name"], i == self.index, size=imgui.ImVec2(0, 40))[0]:
                self.select(i)
        imgui.spacing()
        summary = self.data.get("comparison_summary", self.data["summary"]) if self.mode == "whole" else self.data["summary"]
        imgui.text_colored(CYAN, f"{summary['passed']} / {summary['total']} 验证通过")
        imgui.text_wrapped("当前模式中，比较同一起终关节角的求解路径与线性插值。")
        imgui.separator()
        _, self.show_baseline = imgui.checkbox("显示线性插值", self.show_baseline)
        _, self.show_ghost = imgui.checkbox("显示起终点残影", self.show_ghost)
        imgui.spacing()
        if imgui.button("导出当前轨迹 JSON", imgui.ImVec2(-1, 0)):
            self.export_case()
        if imgui.button("保存渲染截图", imgui.ImVec2(-1, 0)):
            self.pending_screenshot = ROOT / "output" / "opengl-snapshot.png"
        imgui.spacing()
        if imgui.collapsing_header("计算方法"):
            actual_endpoint = self.case.get("metric_mode", self.mode) == "endpoint"
            if self.case["id"] == "original-2r":
                imgui.text_wrapped("2R 是原始 MATLAB 复现基准，在两种模式中保持不变。")
            if actual_endpoint:
                imgui.text_wrapped("E = 0.5 ||J末端 q̇||²")
                imgui.text_wrapped("q̈ = −J末端⁺ H末端[q̇,q̇]")
                imgui.text_wrapped("不加入全连杆或关节正则能量。复杂机械臂由 SVD 选择最小范数初速度与加速度，保持末端直线。")
                imgui.text_wrapped("复杂算例只固定末端目标位置，关节终姿态由升轨决定。关节度量允许半正定。")
            else:
                imgui.text_wrapped("g = Σ wᵢ Jᵢᵀ Jᵢ + λI")
                imgui.text_wrapped("q̈ = −g⁻¹ Σ wᵢ Jᵢᵀ Hᵢ[q̇,q̇]")
                imgui.text_wrapped("对照模式改变能量：全连杆运动加正则项；复杂算例固定关节起终姿态。末端可以弯曲。")
            imgui.text_wrapped("度量能量不代表电机实际能耗。")
            if actual_endpoint:
                imgui.text_wrapped("可行的末端直线是位置空间的最短路径；冗余关节姿态不唯一。")
            else:
                imgui.text_wrapped("全连杆对照求解的是局部测地线，不保证全局最优。")
        if imgui.collapsing_header("模型说明"):
            imgui.text_wrapped("单位：米、弧度。理想旋转关节；未施加碰撞、关节限位和动力学约束。")
            imgui.text_wrapped("原始 2R 在约 2.578 秒遇到奇异位置，复现算例使用 2 秒区间。")
        imgui.text_colored(MUTED, "Dear ImGui + OpenGL 3.3")
        imgui.end_child()

    def inspector(self, height):
        case, m = self.case, self.case["metrics"]
        imgui.begin_child("inspector", imgui.ImVec2(290, height))
        imgui.text_colored(CYAN, "路径表现")
        imgui.separator()
        if case.get("metric_mode", self.mode) == "endpoint":
            imgui.text_colored(CYAN, "能量：0.5 ||末端速度||²")
            imgui.text_colored(MUTED, f"末端直线偏差 {m.get('cartesian_line_error', 0):.2e} m")
        else:
            imgui.text_colored(AMBER, "能量：全连杆运动 + 关节正则")
        imgui.text("相对于关节角线性插值")
        imgui.text_colored(CYAN, f"度量能量降低 {m['energy_improvement_pct']:.2f}%")
        imgui.text(f"测地线积分能量   {m['energy']:.5f}")
        imgui.text(f"插值积分能量     {case['baseline']['energy']:.5f}")
        imgui.text(f"测地线长度       {m['length']:.4f}")
        imgui.text(f"插值路径长度     {case['baseline']['length']:.4f}")
        imgui.spacing()
        imgui.text_colored(CYAN, "当前关节角 / °")
        angles = self.sample(case["q"]) * 180 / np.pi
        for j in range(0, len(angles), 2):
            text = f"q{j+1:<2} {angles[j]:>7.1f}"
            if j + 1 < len(angles):
                text += f"      q{j+2:<2} {angles[j+1]:>7.1f}"
            imgui.text(text)
        imgui.spacing()
        imgui.text_colored(CYAN, "数值验证")
        imgui.separator()
        for check in case["checks"]:
            color = CYAN if check["passed"] else AMBER
            imgui.text_colored(color, ("通过  " if check["passed"] else "失败  ") + check["name"])
            if imgui.is_item_hovered():
                imgui.set_tooltip(check["detail"])
        imgui.text_colored(MUTED, f"能量相对漂移   {m['max_energy_drift']:.2e}")
        imgui.text_colored(MUTED, f"端点误差       {m['endpoint_error']:.2e}")
        imgui.text_colored(MUTED, f"方程残差       {m['geodesic_residual']:.2e}")
        if case.get("metric_mode", self.mode) == "endpoint":
            imgui.text_colored(MUTED, f"末端直线偏差   {m.get('cartesian_line_error', 0):.2e} m")
            imgui.text_colored(MUTED, f"任务最小奇异值 {m.get('min_task_singular_value', 0):.2e}")
        else:
            imgui.text_colored(MUTED, f"度量最小特征值 {m['min_eigenvalue']:.2e}")
        imgui.end_child()

    def viewport(self, width, height):
        case = self.case
        imgui.begin_child("scene", imgui.ImVec2(width, height), window_flags=imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse)
        for name, azimuth, elevation in (("立体", -.95, .48), ("俯视", 0., math.pi / 2 - .005), ("侧视", -math.pi / 2, .03)):
            if imgui.button(name):
                self.camera[:2] = [azimuth, elevation]
                self.camera[2] = self.radius * 3.4
            imgui.same_line()
        imgui.text_colored(CYAN, "测地线")
        imgui.same_line()
        imgui.text_colored(AMBER, "线性插值")
        available = imgui.get_content_region_avail()
        w, h = max(int(available.x), 1), max(int(available.y - 2), 1)
        self.scene_texture = self.renderer.render(
            points=self.sample(case["points"]), baseline_points=self.sample(case["baseline"]["points"]),
            geodesic_trace=case["points"][:, -1], baseline_trace=case["baseline"]["points"][:, -1],
            width=w, height=h, camera=self.camera, center=self.center, radius=self.radius,
            show_baseline=self.show_baseline, show_ghost=self.show_ghost,
            start_points=case["points"][0], end_points=case["points"][-1], floor_height=self.floor_height)
        position = imgui.get_cursor_screen_pos()
        self.viewport_rect = (position.x, position.y, w, h)
        imgui.image(imgui.ImTextureRef(self.scene_texture), imgui.ImVec2(w, h), imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))
        io = imgui.get_io()
        if imgui.is_item_hovered():
            if imgui.is_mouse_dragging(0):
                self.camera[0] -= io.mouse_delta.x * .007
                self.camera[1] = float(np.clip(self.camera[1] + io.mouse_delta.y * .007, -.2, math.pi / 2 - .005))
            if io.mouse_wheel:
                self.camera[2] = float(np.clip(self.camera[2] * math.exp(-io.mouse_wheel * .12), self.radius * 1.15, self.radius * 10))
        imgui.end_child()

    def playback(self):
        if imgui.button("暂停" if self.playing else "播放"):
            if self.progress == 1.:
                self.progress = 0.
            self.playing = not self.playing
        imgui.same_line()
        if imgui.button("复位"):
            self.progress, self.playing = 0., False
        imgui.same_line()
        _, self.loop = imgui.checkbox("循环", self.loop)
        imgui.same_line()
        imgui.set_next_item_width(135)
        _, self.speed = imgui.slider_float("倍速", self.speed, .1, 3., "%.2f x")
        imgui.same_line()
        imgui.text(f"t = {self.progress * self.case['t'][-1]:.2f} / {self.case['t'][-1]:.2f} s")
        imgui.set_next_item_width(-1)
        changed, self.progress = imgui.slider_float("##timeline", self.progress, 0., 1., "进度 %.3f")
        if changed:
            self.playing = False

    def charts(self, width, height):
        case = self.case
        w = max((width - 10) / 2, 100)
        imgui.push_id(f"{self.mode}-{self.index}")
        compact_legend = case["dof"] > 7 and w < 430
        imgui.begin_group()
        legend_start = imgui.get_cursor_pos_y()
        if compact_legend:
            for j in range(case["dof"]):
                imgui.text_colored(implot.get_colormap_color(j), f"q{j+1}")
                if j % 5 != 4 and j < case["dof"] - 1:
                    imgui.same_line()
        legend_height = imgui.get_cursor_pos_y() - legend_start
        plot_flags = implot.Flags_.no_legend if compact_legend else 0
        if implot.begin_plot("关节运动", imgui.ImVec2(w, height - legend_height), flags=plot_flags):
            implot.setup_axes("时间 / s", "关节角 / °")
            # Keep the legend compact, including all ten joints, instead of
            # placing a vertical list inside the small plot area.
            implot.setup_legend(implot.Location_.north, implot.LegendFlags_.horizontal | implot.LegendFlags_.outside)
            for j in range(case["dof"]):
                implot.plot_line(f"q{j+1}", case["t"], np.ascontiguousarray(case["q"][:, j] * 180 / np.pi))
            implot.plot_inf_lines("##当前帧", np.array([self.progress * case["t"][-1]], dtype=np.float64), implot.Spec(flags=implot.ItemFlags_.no_legend))
            implot.end_plot()
        imgui.end_group()
        imgui.same_line()
        if implot.begin_plot("度量能量", imgui.ImVec2(w, height)):
            implot.setup_axes("时间 / s", "度量能量")
            implot.plot_line("测地线", case["t"], case["curve"]["energy"], implot.Spec(line_color=CYAN, line_weight=2))
            implot.plot_line("线性插值", case["t"], case["curve"]["baseline_energy"], implot.Spec(line_color=AMBER, line_weight=2))
            implot.end_plot()
        imgui.pop_id()

    def draw(self):
        io = imgui.get_io()
        if imgui.is_key_pressed(imgui.Key.space) and not io.want_text_input:
            self.playing = not self.playing
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(io.display_size)
        flags = imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
        imgui.begin("Geodesic Robot Lab", flags=flags)
        content = imgui.get_content_region_avail()
        self.sidebar(content.y)
        imgui.same_line()
        imgui.begin_child("experiment", imgui.ImVec2(0, content.y))
        imgui.text(self.case["name"])
        imgui.text_colored(MUTED, self.case["description"])
        imgui.separator()
        available = imgui.get_content_region_avail()
        # Keep scene and controls visible on a 1000x700 desktop; child scrolling
        # preserves inspector content when windows are smaller.
        charts_height = max(190., min(205., available.y * .26))
        scene_height = max(240., available.y - charts_height - 112)
        scene_width = max(260., available.x - 300)
        self.viewport(scene_width, scene_height)
        imgui.same_line()
        self.inspector(scene_height)
        self.playback()
        self.charts(available.x, charts_height)
        imgui.text_colored(MUTED, self.status)
        imgui.end_child()
        imgui.end()

    def capture(self, path):
        width, height = glfw.get_framebuffer_size(self.window)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glReadBuffer(gl.GL_BACK)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()
        from PIL import Image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
        return image


def main():
    parser = argparse.ArgumentParser(description="Dear ImGui + OpenGL geodesic robot renderer")
    parser.add_argument("--data", type=Path, default=ROOT / "output" / "trajectories.json")
    parser.add_argument("--self-test", action="store_true", help="Render all cases in a hidden real OpenGL context, validate pixels, then exit")
    parser.add_argument("--screenshot", type=Path, help="Save an actual GPU-rendered frame")
    parser.add_argument("--width", type=int, default=1440, help="Initial window width (minimum 1000)")
    parser.add_argument("--height", type=int, default=920, help="Initial window height (minimum 700)")
    args = parser.parse_args()
    default_data = ROOT / "output" / "trajectories.json"
    if not args.data.exists() and args.data.resolve() == default_data.resolve():
        build = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--build", "--no-open"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if build.returncode:
            raise RuntimeError(f"Initial trajectory generation failed:\n{build.stdout}\n{build.stderr}")
    data = load_data(args.data)
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    if args.self_test:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(max(1000, args.width), max(700, args.height), "Geodesic Robot Lab | Dear ImGui + OpenGL", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 3.3 core context")
    glfw.make_context_current(window)
    glfw.swap_interval(0 if args.self_test else 1)
    glfw.set_window_size_limits(window, 1000, 700, glfw.DONT_CARE, glfw.DONT_CARE)
    imgui.create_context()
    implot.create_context()
    imgui.get_io().set_ini_filename("")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        imgui.get_io().fonts.add_font_from_file_ttf(str(font), 17)
    theme()
    backend = GlfwRenderer(window)
    app = App(window, data)
    context_info = {"version": gl.glGetString(gl.GL_VERSION).decode(),
                    "renderer": gl.glGetString(gl.GL_RENDERER).decode(),
                    "vendor": gl.glGetString(gl.GL_VENDOR).decode()}
    print(json.dumps(context_info), flush=True)
    prior = time.perf_counter()
    rendered, test_cases = 0, []
    # Two frames per configuration allow the dynamic Chinese glyph atlas to
    # finish uploading before screenshots and pixel checks.
    modes = ["endpoint", "whole"] if data.get("comparison_cases") else ["endpoint"]
    tests = [(mode, i, progress) for mode in modes for i in range(len(data["cases"])) for progress in (0., .5, 1.)]
    tests.append(("endpoint", next((i for i, c in enumerate(data["cases"]) if c["id"] == "spatial-7r"), 0), .42))
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            backend.process_inputs()
            now = time.perf_counter()
            app.tick(now - prior)
            prior = now
            if args.self_test:
                mode, case_index, progress = tests[min(rendered // 3, len(tests) - 1)]
                if mode != app.mode or case_index != app.index:
                    app.mode = mode
                    app.select(case_index)
                app.progress, app.playing = progress, False
            imgui.new_frame()
            app.draw()
            width, height = glfw.get_framebuffer_size(window)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(.045, .065, .105, 1.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            imgui.render()
            backend.render(imgui.get_draw_data())
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                raise RuntimeError(f"OpenGL error {error}")
            if args.self_test and rendered % 3 == 2:
                # Read actual scene texture, separately from ImGui, to catch a
                # blank viewport even when the surrounding GUI rendered.
                gl.glBindTexture(gl.GL_TEXTURE_2D, app.scene_texture)
                raw = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                pixels = np.frombuffer(raw, dtype=np.uint8)
                rgb = pixels.reshape(-1, 3).astype(np.float32)
                cyan_pixels = int(np.count_nonzero((rgb[:, 1] > 60) & (rgb[:, 1] > rgb[:, 0] * 1.5) & (rgb[:, 2] > rgb[:, 0] * 1.5)))
                if len(pixels) == 0 or pixels.std() < 8 or cyan_pixels < 100:
                    raise RuntimeError(f"Blank or invalid GPU scene: {app.case['id']}")
                test_cases.append({"mode": app.mode, "case": app.case["id"], "progress": app.progress, "pixel_std": float(pixels.std()), "cyan_pixels": cyan_pixels, "gl_error": int(error)})
                if app.progress == .5 and app.case["id"] in ("original-2r", "spatial-10r"):
                    app.capture(ROOT / "output" / f"opengl-{app.mode}-{app.case['id']}.png")
            if getattr(app, "pending_screenshot", None):
                app.capture(app.pending_screenshot)
                app.status = f"已保存：{app.pending_screenshot.name}"
                app.pending_screenshot = None
            if args.screenshot and ((args.self_test and rendered == len(tests) * 3 - 1) or (not args.self_test and rendered == 5)):
                app.capture(args.screenshot)
            glfw.swap_buffers(window)
            rendered += 1
            if args.self_test and rendered >= len(tests) * 3:
                report = {"context": context_info, "rendered_frames": rendered, "checks": test_cases,
                          "success": True, "cases": len(data["cases"])}
                (ROOT / "output" / "render-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"GPU validation passed: {len(test_cases)} scenes / {rendered} frames", flush=True)
                break
    finally:
        app.renderer.shutdown()
        backend.shutdown()
        implot.destroy_context()
        imgui.destroy_context()
        glfw.destroy_window(window)
        glfw.terminate()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        (ROOT / "output").mkdir(exist_ok=True)
        (ROOT / "output" / "renderer-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        raise SystemExit(1)
