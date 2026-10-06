"""OpenGL 3.3 core robot renderer for an ImGui image viewport.

Create Renderer only after making an OpenGL context current. Angles in the
camera tuple are radians; its third component is a world-space orbit distance.
render() returns an RGBA8 texture. Display it with vertically flipped ImGui UVs
(0, 1), (1, 0). No compatibility-profile drawing functions are used.
"""

import ctypes
import math

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader


_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_view_projection;
uniform mat4 u_model;
uniform mat3 u_normal;
out vec3 v_world;
out vec3 v_normal;
void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world = world.xyz;
    v_normal = u_normal * a_normal;
    gl_Position = u_view_projection * world;
}
"""

_FRAGMENT_SHADER = """#version 330 core
in vec3 v_world;
in vec3 v_normal;
uniform vec4 u_color;
uniform vec3 u_eye;
uniform bool u_unlit;
out vec4 frag_color;
void main() {
    vec3 color = u_color.rgb;
    if (!u_unlit) {
        vec3 normal = normalize(v_normal);
        vec3 light = normalize(vec3(-0.35, -0.45, 0.86));
        vec3 view = normalize(u_eye - v_world);
        float diffuse = max(dot(normal, light), 0.0);
        float rim = pow(1.0 - max(dot(normal, view), 0.0), 3.0);
        vec3 half_vector = normalize(light + view);
        float specular = pow(max(dot(normal, half_vector), 0.0), 48.0);
        color = color * (0.32 + 0.68 * diffuse)
              + vec3(0.62, 0.72, 0.78) * specular * 0.48
              + vec3(0.08, 0.16, 0.18) * rim;
    }
    frag_color = vec4(color, u_color.a);
}
"""


def _unit(vector):
    length = np.linalg.norm(vector)
    return vector / max(float(length), 1e-15)


def _perspective(fov, aspect, near, far):
    focal = 1.0 / math.tan(fov / 2.0)
    return np.array([
        [focal / aspect, 0, 0, 0],
        [0, focal, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def _look_at(eye, target):
    forward = _unit(target - eye)
    side = _unit(np.cross(forward, [0.0, 0.0, 1.0]))
    up = np.cross(side, forward)
    matrix = np.eye(4)
    matrix[0, :3] = side
    matrix[1, :3] = up
    matrix[2, :3] = -forward
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix.astype(np.float32)


def _cylinder_mesh(segments=32):
    vertices = []

    def append(position, normal):
        vertices.append((*position, *normal))

    for i in range(segments):
        angle0 = 2 * math.pi * i / segments
        angle1 = 2 * math.pi * (i + 1) / segments
        normal0 = (math.cos(angle0), math.sin(angle0), 0.0)
        normal1 = (math.cos(angle1), math.sin(angle1), 0.0)
        bottom0, top0 = (*normal0[:2], 0.0), (*normal0[:2], 1.0)
        bottom1, top1 = (*normal1[:2], 0.0), (*normal1[:2], 1.0)
        for position, normal in (
            (bottom0, normal0), (bottom1, normal1), (top1, normal1),
            (bottom0, normal0), (top1, normal1), (top0, normal0),
        ):
            append(position, normal)
        for position in ((0, 0, 0), bottom1, bottom0):
            append(position, (0, 0, -1))
        for position in ((0, 0, 1), top0, top1):
            append(position, (0, 0, 1))
    return np.array(vertices, dtype=np.float32)


def _sphere_mesh(longitudes=24, latitudes=16):
    vertices = []

    def point(latitude, longitude):
        phi = -math.pi / 2 + math.pi * latitude / latitudes
        theta = 2 * math.pi * longitude / longitudes
        return (math.cos(phi) * math.cos(theta),
                math.cos(phi) * math.sin(theta), math.sin(phi))

    for latitude in range(latitudes):
        for longitude in range(longitudes):
            p00 = point(latitude, longitude)
            p01 = point(latitude, longitude + 1)
            p10 = point(latitude + 1, longitude)
            p11 = point(latitude + 1, longitude + 1)
            for p in (p00, p01, p11, p00, p11, p10):
                vertices.append((*p, *p))
    return np.array(vertices, dtype=np.float32)


class _Mesh:
    def __init__(self, vertices=None, dynamic=False):
        self.vao = int(GL.glGenVertexArrays(1))
        self.vbo = int(GL.glGenBuffers(1))
        self.dynamic = dynamic
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))
        if vertices is not None:
            self.upload(vertices)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def upload(self, vertices):
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.count = len(vertices)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices,
                        GL.GL_DYNAMIC_DRAW if self.dynamic else GL.GL_STATIC_DRAW)

    def draw(self, mode=GL.GL_TRIANGLES):
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(mode, 0, self.count)

    def shutdown(self):
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteVertexArrays(1, [self.vao])


class Renderer:
    """Lit meshes, orbit camera, comparison paths and reusable offscreen texture."""

    def __init__(self):
        self.program = int(compileProgram(
            compileShader(_VERTEX_SHADER, GL.GL_VERTEX_SHADER),
            compileShader(_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER),
        ))
        self.uniforms = {name: GL.glGetUniformLocation(self.program, f"u_{name}")
                         for name in ("view_projection", "model", "normal", "color", "eye", "unlit")}
        self.cylinder = _Mesh(_cylinder_mesh())
        self.sphere = _Mesh(_sphere_mesh())
        self.lines = _Mesh(dynamic=True)
        self.plane = _Mesh(np.array([
            [-1, -1, 0, 0, 0, 1], [1, -1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1],
            [-1, -1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1], [-1, 1, 0, 0, 0, 1],
        ], dtype=np.float32))
        self.framebuffer = int(GL.glGenFramebuffers(1))
        self.texture = int(GL.glGenTextures(1))
        self.depth = int(GL.glGenRenderbuffers(1))
        self.width = 0
        self.height = 0
        self._closed = False
        self._view_direction = np.array([0., 0., -1.])
        self._trace_scale = .0015

    def _resize(self, width, height):
        if (width, height) == (self.width, self.height):
            return
        previous_texture = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        previous_renderbuffer = int(GL.glGetIntegerv(GL.GL_RENDERBUFFER_BINDING))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self.texture, 0)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                     GL.GL_RENDERBUFFER, self.depth)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindTexture(GL.GL_TEXTURE_2D, previous_texture)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, previous_renderbuffer)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Robot viewport framebuffer is incomplete: 0x{status:04x}")
        self.width, self.height = width, height

    def _draw(self, mesh, model, color, unlit=False, mode=GL.GL_TRIANGLES):
        GL.glUniformMatrix4fv(self.uniforms["model"], 1, GL.GL_TRUE,
                              np.asarray(model, dtype=np.float32))
        normal = np.linalg.inv(model[:3, :3]).T
        GL.glUniformMatrix3fv(self.uniforms["normal"], 1, GL.GL_TRUE,
                              np.asarray(normal, dtype=np.float32))
        GL.glUniform4f(self.uniforms["color"], *color)
        GL.glUniform1i(self.uniforms["unlit"], int(unlit))
        mesh.draw(mode)

    def _rod(self, begin, end, thickness, color):
        direction = np.asarray(end) - np.asarray(begin)
        length = np.linalg.norm(direction)
        if length < 1e-10:
            return
        axis = direction / length
        helper = np.array([0., 0., 1.]) if abs(axis[2]) < .92 else np.array([0., 1., 0.])
        x = _unit(np.cross(helper, axis))
        y = np.cross(axis, x)
        model = np.eye(4)
        model[:3, :3] = np.column_stack((x * thickness, y * thickness, direction))
        model[:3, 3] = begin
        self._draw(self.cylinder, model, color)

    def _ball(self, position, size, color):
        model = np.eye(4)
        model[:3, :3] *= size
        model[:3, 3] = position
        self._draw(self.sphere, model, color)

    def _robot(self, points, thickness, body_color, joint_color, tool_color=None):
        if points is None:
            return
        points = np.asarray(points, dtype=float)
        for start, end in zip(points[:-1], points[1:]):
            self._rod(start, end, thickness, body_color)
        for point in points[:-1]:
            self._ball(point, thickness * 1.55, joint_color)
        self._ball(points[-1], thickness * 1.8,
                   tool_color if tool_color is not None else body_color)

    def _polyline(self, points, color, width=2.0):
        if points is None:
            return
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            return
        if not np.all(np.isfinite(points)):
            return
        # Wide glLineWidth values are invalid in some forward-compatible core
        # contexts. Camera-facing triangle ribbons retain portable trace width.
        direction = points[1:] - points[:-1]
        side = np.cross(direction, self._view_direction)
        lengths = np.linalg.norm(side, axis=1)
        visible = lengths > 1e-12
        if not np.any(visible):
            return
        side = side[visible] / lengths[visible, None] * self._trace_scale * width
        begin, end = points[:-1][visible], points[1:][visible]
        positions = np.stack((begin - side, end - side, end + side,
                              begin - side, end + side, begin + side), axis=1).reshape(-1, 3)
        self.lines.upload(np.column_stack((positions, np.zeros_like(positions))))
        self._draw(self.lines, np.eye(4), color, unlit=True)

    def _ground(self, center, radius, floor):
        extent = radius * 1.6
        model = np.eye(4)
        model[0, 0] = extent
        model[1, 1] = extent
        model[:3, 3] = [center[0], center[1], floor]
        self._draw(self.plane, model, (.085, .11, .14, 1.0))
        minor, major = [], []
        for index in range(-10, 11):
            position = index * extent / 10
            target = major if index % 5 == 0 else minor
            target.extend([
                [center[0] - extent, center[1] + position, floor + radius * .0003],
                [center[0] + extent, center[1] + position, floor + radius * .0003],
                [center[0] + position, center[1] - extent, floor + radius * .0003],
                [center[0] + position, center[1] + extent, floor + radius * .0003],
            ])
        GL.glLineWidth(1.0)
        for points, color in ((minor, (.14, .19, .23, 1.0)),
                              (major, (.20, .27, .32, 1.0))):
            positions = np.asarray(points, dtype=np.float32)
            self.lines.upload(np.column_stack((positions, np.zeros_like(positions))))
            self._draw(self.lines, np.eye(4), color, unlit=True, mode=GL.GL_LINES)

    def render(self, points, baseline_points, geodesic_trace, baseline_trace,
               width, height, camera, center, radius, show_baseline=True,
               show_ghost=True, start_points=None, end_points=None, floor_height=None):
        """Draw one frame; points are (joints+1,3), traces are (samples,3).

        camera = (azimuth, elevation, distance), angles in radians and distance
        in world units. The caller may orbit continuously or set elevation to
        pi/2 for a planar top view. Set floor_height from the complete trajectory
        bounds to keep a fixed floor below every link. Returns a texture ID
        valid until shutdown.
        """
        if self._closed:
            raise RuntimeError("Renderer has been shut down")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("points must have shape (joints+1,3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("robot points must be finite")
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite 3-vector")
        radius = max(float(radius), .01)
        if floor_height is not None and not np.isfinite(float(floor_height)):
            raise ValueError("floor_height must be finite")
        azimuth, elevation, distance = (float(value) for value in camera)
        elevation = np.clip(elevation, -math.pi / 2 + 1e-5, math.pi / 2 - 1e-5)
        distance = max(distance, radius * .4)
        eye = center + distance * np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth), math.sin(elevation),
        ])
        self._view_direction = _unit(center - eye)
        self._trace_scale = radius * .0015
        width, height = max(1, int(width)), max(1, int(height))
        previous_draw = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        previous_read = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        previous_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        previous_program = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        previous_vao = int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING))
        previous_buffer = int(GL.glGetIntegerv(GL.GL_ARRAY_BUFFER_BINDING))
        previous_depth_mask = bool(GL.glGetBooleanv(GL.GL_DEPTH_WRITEMASK))
        previous_depth_function = int(GL.glGetIntegerv(GL.GL_DEPTH_FUNC))
        previous_clear_color = GL.glGetFloatv(GL.GL_COLOR_CLEAR_VALUE)
        previous_line_width = float(GL.glGetFloatv(GL.GL_LINE_WIDTH))
        previous_depth = GL.glIsEnabled(GL.GL_DEPTH_TEST)
        previous_blend = GL.glIsEnabled(GL.GL_BLEND)
        previous_cull = GL.glIsEnabled(GL.GL_CULL_FACE)
        previous_scissor = GL.glIsEnabled(GL.GL_SCISSOR_TEST)
        previous_blend_src = int(GL.glGetIntegerv(GL.GL_BLEND_SRC_RGB))
        previous_blend_dst = int(GL.glGetIntegerv(GL.GL_BLEND_DST_RGB))
        previous_blend_src_alpha = int(GL.glGetIntegerv(GL.GL_BLEND_SRC_ALPHA))
        previous_blend_dst_alpha = int(GL.glGetIntegerv(GL.GL_BLEND_DST_ALPHA))
        previous_blend_equation = int(GL.glGetIntegerv(GL.GL_BLEND_EQUATION_RGB))
        previous_blend_equation_alpha = int(GL.glGetIntegerv(GL.GL_BLEND_EQUATION_ALPHA))
        try:
            self._resize(width, height)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
            GL.glViewport(0, 0, width, height)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDepthFunc(GL.GL_LESS)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glDisable(GL.GL_BLEND)
            GL.glClearColor(.035, .052, .071, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glUseProgram(self.program)
            projection = _perspective(math.radians(42), width / height,
                                      max(.001, radius * .003), distance + radius * 10)
            view_projection = projection @ _look_at(eye, center)
            GL.glUniformMatrix4fv(self.uniforms["view_projection"], 1, GL.GL_TRUE,
                                  view_projection)
            GL.glUniform3f(self.uniforms["eye"], *eye)
            thickness = radius * .030
            floor = (float(floor_height) if floor_height is not None
                     else min(0.0, float(points[0, 2]) - radius * .13))
            self._ground(center, radius, floor)
            # A squat base pedestal anchors the otherwise floating arm model.
            base = points[0]
            self._rod([base[0], base[1], floor], base,
                      thickness * 3.6, (.27, .33, .38, 1.0))
            self._rod(base - [0, 0, thickness * .6], base + [0, 0, thickness * .25],
                      thickness * 4.3, (.38, .46, .50, 1.0))
            self._polyline(geodesic_trace, (.19, .88, .92, 1.0), 2.5)
            if show_baseline:
                self._polyline(baseline_trace, (1.0, .67, .23, 1.0), 2.0)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendEquation(GL.GL_FUNC_ADD)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            if show_ghost:
                self._robot(start_points, thickness * .65,
                            (.40, .63, .65, .19), (.48, .62, .66, .22))
                self._robot(end_points, thickness * .65,
                            (.68, .80, .84, .16), (.68, .80, .84, .19))
            if show_baseline:
                self._robot(baseline_points, thickness * .73,
                            (1.0, .61, .18, .45), (.82, .65, .38, .50))
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
            self._robot(points, thickness, (.13, .67, .72, 1.0),
                        (.53, .62, .68, 1.0), (.50, .93, .81, 1.0))
        finally:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, previous_draw)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, previous_read)
            GL.glViewport(*previous_viewport)
            GL.glUseProgram(previous_program)
            GL.glBindVertexArray(previous_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, previous_buffer)
            GL.glDepthMask(previous_depth_mask)
            GL.glDepthFunc(previous_depth_function)
            GL.glClearColor(*previous_clear_color)
            GL.glLineWidth(previous_line_width)
            GL.glBlendFuncSeparate(previous_blend_src, previous_blend_dst,
                                   previous_blend_src_alpha, previous_blend_dst_alpha)
            GL.glBlendEquationSeparate(previous_blend_equation,
                                       previous_blend_equation_alpha)
            for capability, enabled in ((GL.GL_DEPTH_TEST, previous_depth),
                                        (GL.GL_BLEND, previous_blend),
                                        (GL.GL_CULL_FACE, previous_cull),
                                        (GL.GL_SCISSOR_TEST, previous_scissor)):
                (GL.glEnable if enabled else GL.glDisable)(capability)
        return self.texture

    def shutdown(self):
        """Delete GL resources while their owning context is still current."""
        if self._closed:
            return
        for mesh in (self.cylinder, self.sphere, self.lines, self.plane):
            mesh.shutdown()
        GL.glDeleteFramebuffers(1, [self.framebuffer])
        GL.glDeleteRenderbuffers(1, [self.depth])
        GL.glDeleteTextures([self.texture])
        GL.glDeleteProgram(self.program)
        self._closed = True
