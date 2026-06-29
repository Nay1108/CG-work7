import taichi as ti

ti.init(arch=ti.gpu)

N = 20
mass = 1.0
dt = 5e-4
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

k_s = ti.field(float, shape=())
k_shear = ti.field(float, shape=())
k_bend = ti.field(float, shape=())
k_d = ti.field(float, shape=())

sphere_center = ti.Vector.field(3, dtype=float, shape=())
sphere_radius = ti.field(float, shape=())
enable_collision = ti.field(int, shape=())

k_s[None] = 10000.0
k_shear[None] = 5000.0
k_bend[None] = 1000.0
k_d[None] = 1.0
sphere_center[None] = ti.Vector([0.0, -0.3, 0.0])
sphere_radius[None] = 0.3
enable_collision[None] = 1

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        is_fixed[idx] = 1 if (j == 0 and (i == 0 or i == N - 1)) else 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx2 = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx2])
            spring_lengths[c] = (x[idx] - x[idx2]).norm()
        if j < N - 1:
            idx2 = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx2])
            spring_lengths[c] = (x[idx] - x[idx2]).norm()
        if i < N - 1 and j < N - 1:
            idx2 = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx2])
            spring_lengths[c] = (x[idx] - x[idx2]).norm()
        if i < N - 1 and j > 0:
            idx2 = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx2])
            spring_lengths[c] = (x[idx] - x[idx2]).norm()
        if i < N - 2:
            idx2 = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx2])
            spring_lengths[c] = (x[idx] - x[idx2]).norm()
        if j < N - 2:
            idx2 = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx2])
            spring_lengths[c] = (x[idx] - x[idx2]).norm()

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

@ti.func
def compute_spring_force(pos_a, pos_b, rest_len, stiff):
    d = pos_a - pos_b
    dist = d.norm()
    force = ti.Vector([0.0, 0.0, 0.0])
    if dist > 1e-6:
        force = -stiff * (dist - rest_len) * (d / dist)
    return force

@ti.func
def compute_forces_on(pos, vel, force):
    for i in range(N * N):
        force[i] = gravity * mass - k_d[None] * vel[i]

    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]

        stiffness = 0.0
        if i < 2 * (N - 1) * N:
            stiffness = k_s[None]
        elif i < 2 * (N - 1) * N + 2 * (N - 1) * (N - 1):
            stiffness = k_shear[None]
        else:
            stiffness = k_bend[None]

        f_spring = compute_spring_force(pos_a, pos_b, spring_lengths[i], stiffness)
        ti.atomic_add(force[idx_a], f_spring)
        ti.atomic_add(force[idx_b], -f_spring)

    if enable_collision[None]:
        for i in range(N * N):
            if is_fixed[i] == 0:
                d = pos[i] - sphere_center[None]
                dist = d.norm()
                if dist < sphere_radius[None] and dist > 1e-6:
                    normal = d / dist
                    penetration = sphere_radius[None] - dist
                    force[i] += normal * 10000.0 * penetration

@ti.func
def clamp_velocity(vel, idx):
    v_norm = vel[idx].norm()
    if v_norm > max_velocity:
        vel[idx] = vel[idx] / v_norm * max_velocity

@ti.func
def handle_collision(pos, vel, idx):
    if enable_collision[None] and is_fixed[idx] == 0:
        d = pos[idx] - sphere_center[None]
        dist = d.norm()
        if dist < sphere_radius[None] and dist > 1e-6:
            normal = d / dist
            pos[idx] = sphere_center[None] + normal * sphere_radius[None]
            vn = vel[idx].dot(normal)
            if vn < 0.0:
                vel[idx] -= vn * normal

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            handle_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            handle_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]

    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
                handle_collision(x_next, v_next, i)

    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

def main():
    init_cloth()

    window = ti.ui.Window("Mass-Spring with Shear, Bend & Collision", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    # 用于渲染单个球体的场（形状为 (1,)）
    sphere_point = ti.Vector.field(3, dtype=float, shape=(1,))

    current_method = 1
    paused = False

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.48)

        window.GUI.text("Integration Method:")
        if window.GUI.button("[*] Explicit Euler" if current_method == 0 else "[ ] Explicit Euler"):
            current_method = 0
            init_cloth()
        if window.GUI.button("[*] Semi-Implicit" if current_method == 1 else "[ ] Semi-Implicit"):
            current_method = 1
            init_cloth()
        if window.GUI.button("[*] Implicit Euler" if current_method == 2 else "[ ] Implicit Euler"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        if window.GUI.button("Pause" if not paused else "Resume"):
            paused = not paused
        if window.GUI.button("Reset"):
            init_cloth()

        window.GUI.text("")
        window.GUI.text("Spring Stiffness:")
        new_ks = window.GUI.slider_float("Structural", k_s[None], 0.0, 20000.0)
        new_kshear = window.GUI.slider_float("Shear", k_shear[None], 0.0, 10000.0)
        new_kbend = window.GUI.slider_float("Bending", k_bend[None], 0.0, 5000.0)
        new_kd = window.GUI.slider_float("Damping", k_d[None], 0.0, 10.0)
        k_s[None] = new_ks
        k_shear[None] = new_kshear
        k_bend[None] = new_kbend
        k_d[None] = new_kd

        window.GUI.text("")
        window.GUI.text("Collision:")
        enable_collision[None] = 1 if window.GUI.checkbox("Enable", bool(enable_collision[None])) else 0

        cx = window.GUI.slider_float("Sphere X", sphere_center[None][0], -0.5, 0.5)
        cy = window.GUI.slider_float("Sphere Y", sphere_center[None][1], -0.5, 1.0)
        cz = window.GUI.slider_float("Sphere Z", sphere_center[None][2], -0.5, 0.5)
        sphere_center[None] = ti.Vector([cx, cy, cz])
        sphere_radius[None] = window.GUI.slider_float("Radius", sphere_radius[None], 0.1, 0.5)

        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light((0.5, 1.5, 1.5), (1, 1, 1))

        # 更新球体位置场并绘制
        sphere_point[0] = sphere_center[None]
        if enable_collision[None]:
            scene.particles(sphere_point, radius=sphere_radius[None], color=(1.0, 0.3, 0.3))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.0, color=(0.8, 0.8, 0.8))

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()