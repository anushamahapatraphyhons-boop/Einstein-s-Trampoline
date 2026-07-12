import taichi as ti
import numpy as np

# Initialize Taichi with GPU architecture and fallback to CPU if needed
try:
    ti.init(arch=ti.gpu, device_memory_GB=0.25)
except Exception:
    ti.init(arch=ti.cpu, device_memory_GB=0.25)
    print("Running on CPU, expect lower FPS. Lower the particle count if it crawls.")

# =====================================================================
# Configuration Constants
# =====================================================================
screen_w = 1280
screen_h = 800
num_particles = 70000       # Accretion disk filaments
num_photon_ring = 15000     # High-density photon sphere ring
num_stars = 2500            # Background star field
num_hawking = 120           # Hawking pairs
H_steps = 4                 # History steps for particle trails (filaments)

# Physics Constants
M = 1.0                     # Mass of the black hole
spin = 0.98                 # Kerr spin parameter (a*)
r_event_horizon = M + ti.sqrt(M*M - spin*spin) # Kerr horizon radius
r_shadow = 5.2              # Ray-traced shadow radius (3D)

# =====================================================================
# Taichi Fields for Simulation State
# =====================================================================
# Accretion Disk Fields
pos = ti.Vector.field(3, dtype=ti.float32, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.float32, shape=num_particles)
color = ti.Vector.field(3, dtype=ti.float32, shape=num_particles)
age = ti.field(dtype=ti.float32, shape=num_particles)
history = ti.Vector.field(3, dtype=ti.float32, shape=(num_particles, H_steps))

# Photon Ring Fields (very close to horizon, gold/white)
pr_pos = ti.Vector.field(3, dtype=ti.float32, shape=num_photon_ring)
pr_vel = ti.Vector.field(3, dtype=ti.float32, shape=num_photon_ring)
pr_color = ti.Vector.field(3, dtype=ti.float32, shape=num_photon_ring)
pr_history = ti.Vector.field(3, dtype=ti.float32, shape=(num_photon_ring, H_steps))

# Hawking Radiation Fields
# [i, 0] is infalling (negative energy), [i, 1] is escaping (positive energy)
hawking_pos = ti.Vector.field(3, dtype=ti.float32, shape=(num_hawking, 2))
hawking_vel = ti.Vector.field(3, dtype=ti.float32, shape=(num_hawking, 2))
hawking_age = ti.field(dtype=ti.float32, shape=num_hawking)
hawking_active = ti.field(dtype=ti.int32, shape=num_hawking)

# Background Stars
star_pos = ti.Vector.field(3, dtype=ti.float32, shape=num_stars)
star_color = ti.Vector.field(3, dtype=ti.float32, shape=num_stars)

# Interactive Geodesic Trail
interactive_trail_pos = ti.Vector.field(3, dtype=ti.float32, shape=300)

# Camera State Fields (need to be fields for kernel access)
camera_pos = ti.Vector.field(3, dtype=ti.float32, shape=1)
camera_dist = ti.field(dtype=ti.float32, shape=1)
w_cam = ti.Vector.field(3, dtype=ti.float32, shape=1) # camera forward
u_cam = ti.Vector.field(3, dtype=ti.float32, shape=1) # camera right
v_cam = ti.Vector.field(3, dtype=ti.float32, shape=1) # camera up
r_shadow_pixels = ti.field(dtype=ti.float32, shape=1)

# Global Simulation Parameters
time = ti.field(dtype=ti.float32, shape=1)

# Render Buffers
pixel_buffer = ti.Vector.field(3, dtype=ti.float32, shape=(screen_w, screen_h))
bright_buffer = ti.Vector.field(3, dtype=ti.float32, shape=(screen_w, screen_h))
blur_buffer_temp = ti.Vector.field(3, dtype=ti.float32, shape=(screen_w, screen_h))
blur_buffer = ti.Vector.field(3, dtype=ti.float32, shape=(screen_w, screen_h))

# =====================================================================
# Math and Noise Helpers
# =====================================================================
@ti.func
def hash22(p):
    # Simple pseudo-random hash returning 2D fractional values
    x = ti.sin(p.dot(ti.Vector([127.1, 311.7]))) * 43758.5453123
    y = ti.sin(p.dot(ti.Vector([269.5, 183.3]))) * 43758.5453123
    return ti.Vector([x - ti.floor(x), y - ti.floor(y)])

@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a

@ti.func
def noise(p):
    i = ti.floor(p)
    f = p - i
    # Smoothstep interpolation
    u = f * f * (3.0 - 2.0 * f)
    
    a = hash22(i + ti.Vector([0.0, 0.0])).x
    b = hash22(i + ti.Vector([1.0, 0.0])).x
    c = hash22(i + ti.Vector([0.0, 1.0])).x
    d = hash22(i + ti.Vector([1.0, 1.0])).x
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y)

@ti.func
def get_temperature_color(r):
    # Relativistic temperature-based color mapping
    # Inner disk: white-hot -> Yellow -> Orange -> Red -> Magenta in outer regions
    col = ti.Vector([0.0, 0.0, 0.0])
    if r < 3.8:
        # Extremely hot inner ring near photon sphere
        col = ti.Vector([1.4, 1.1, 0.9])
    elif r < 5.5:
        # Mid accretion flow: warm yellow/orange
        t = (r - 3.8) / 1.7
        col = ti.Vector([1.2, 0.75 - 0.35 * t, 0.15])
    elif r < 8.0:
        # Outer accretion flow: hot red/orange
        t = (r - 5.5) / 2.5
        col = ti.Vector([1.0, 0.4 - 0.25 * t, 0.1])
    else:
        # Turbulent disk edges: deep magenta/indigo filaments
        t = (r - 8.0) / 4.0
        col = ti.Vector([0.8 - 0.5 * t, 0.05, 0.15 + 0.35 * t])
    return col

# Boyer-Lindquist Kerr geodesic integration helpers
@ti.func
def hamiltonian(r, theta, pr, ptheta, Lz, E):
    sin_theta = ti.sin(theta)
    sin2 = sin_theta * sin_theta + 1e-5
    cos2 = ti.cos(theta) * ti.cos(theta)
    Delta = r*r - 2.0*M*r + spin*spin
    Sigma = r*r + spin*spin*cos2
    
    g_rr = Delta / (Sigma + 1e-5)
    g_thth = 1.0 / (Sigma + 1e-5)
    g_phiph = (Delta - spin*spin*sin2) / (Sigma * Delta * sin2 + 1e-5)
    g_tt = -((r*r + spin*spin)**2 - spin*spin*Delta*sin2) / (Sigma * Delta + 1e-5)
    g_tphi = -2.0 * spin * M * r / (Sigma * Delta + 1e-5)
    
    H = 0.5 * (g_rr * pr*pr + g_thth * ptheta*ptheta + g_phiph * Lz*Lz + g_tt * E*E + 2.0 * g_tphi * (-E) * Lz)
    return H

@ti.func
def get_derivatives(r, theta, pr, ptheta, Lz, E):
    sin_theta = ti.sin(theta)
    sin2 = sin_theta * sin_theta + 1e-5
    cos2 = ti.cos(theta) * ti.cos(theta)
    Delta = r*r - 2.0*M*r + spin*spin
    Sigma = r*r + spin*spin*cos2
    
    g_rr = Delta / (Sigma + 1e-5)
    g_thth = 1.0 / (Sigma + 1e-5)
    g_phiph = (Delta - spin*spin*sin2) / (Sigma * Delta * sin2 + 1e-5)
    g_tphi = -2.0 * spin * M * r / (Sigma * Delta + 1e-5)
    
    dr = g_rr * pr
    dtheta = g_thth * ptheta
    dphi = g_phiph * Lz - g_tphi * E
    
    eps = 1e-3
    dH_dr = (hamiltonian(r + eps, theta, pr, ptheta, Lz, E) - hamiltonian(r - eps, theta, pr, ptheta, Lz, E)) / (2.0 * eps)
    dH_dtheta = (hamiltonian(r, theta + eps, pr, ptheta, Lz, E) - hamiltonian(r, theta - eps, pr, ptheta, Lz, E)) / (2.0 * eps)
    
    dpr = -dH_dr
    dptheta = -dH_dtheta
    
    return dr, dtheta, dphi, dpr, dptheta

@ti.func
def rk4_step(r, theta, phi, pr, ptheta, Lz, E, dt):
    dr1, dth1, dph1, dpr1, dpth1 = get_derivatives(r, theta, pr, ptheta, Lz, E)
    
    r2 = r + 0.5 * dt * dr1
    th2 = theta + 0.5 * dt * dth1
    pr2 = pr + 0.5 * dt * dpr1
    pth2 = ptheta + 0.5 * dt * dpth1
    dr2, dth2, dph2, dpr2, dpth2 = get_derivatives(r2, th2, pr2, pth2, Lz, E)
    
    r3 = r + 0.5 * dt * dr2
    th3 = theta + 0.5 * dt * dth2
    pr3 = pr + 0.5 * dt * dpr2
    pth3 = ptheta + 0.5 * dt * dpth2
    dr3, dth3, dph3, dpr3, dpth3 = get_derivatives(r3, th3, pr3, pth3, Lz, E)
    
    r4 = r + dt * dr3
    th4 = theta + dt * dth3
    pr4 = pr + dt * dpr3
    pth4 = ptheta + dt * dpth3
    dr4, dth4, dph4, dpr4, dpth4 = get_derivatives(r4, th4, pr4, pth4, Lz, E)
    
    r_new = r + dt/6.0 * (dr1 + 2.0*dr2 + 2.0*dr3 + dr4)
    th_new = theta + dt/6.0 * (dth1 + 2.0*dth2 + 2.0*dth3 + dth4)
    phi_new = phi + dt/6.0 * (dph1 + 2.0*dph2 + 2.0*dph3 + dph4)
    pr_new = pr + dt/6.0 * (dpr1 + 2.0*dpr2 + 2.0*dpr3 + dpr4)
    pth_new = ptheta + dt/6.0 * (dpth1 + 2.0*dpth2 + 2.0*dpth3 + dpth4)
    
    return r_new, th_new, phi_new, pr_new, pth_new

# =====================================================================
# Gravitational Lensing Projection Engine
# =====================================================================
@ti.func
def project_with_lensing_rgb(p, scale):
    # Computes gravitational lensing projection of 3D point p onto screen space.
    # Returns primary image, primary magnification, secondary image, secondary magnification,
    # and depth along the camera axis.
    d = p - camera_pos[0]
    z_cam = d.dot(w_cam[0])
    
    x_cam = d.dot(u_cam[0])
    y_cam = d.dot(v_cam[0])
    
    f = 850.0 # focal length
    # Perspective projection
    x_proj = (x_cam / (z_cam + 1e-3)) * f
    y_proj = (y_cam / (z_cam + 1e-3)) * f
    
    r0 = ti.sqrt(x_proj*x_proj + y_proj*y_proj + 1e-6)
    ex = x_proj / (r0 + 1e-3)
    ey = y_proj / (r0 + 1e-3)
    
    DL = camera_dist[0]
    
    # Gravitational bending (Einstein radius squared)
    # The lens effect fades out if the particle is in front of the black hole (z_cam < DL)
    RE2 = 0.0
    if z_cam > 0.1 * DL:
        RE2 = (f * f) * (4.0 * M / (DL + 1e-3)) * ti.max(0.0, (z_cam - 0.5 * DL) / (z_cam + 1e-3)) * scale * scale
        
    r1 = 0.0
    r2 = 0.0
    mu1 = 1.0
    mu2 = 0.0
    has_2 = 0
    
    if RE2 > 0.0:
        disc = ti.sqrt(r0*r0 + 4.0 * RE2 + 1e-6)
        r1 = 0.5 * (r0 + disc)
        r2 = 0.5 * (r0 - disc) # negative, appears on opposite side
        
        # Magnification equations
        mu1 = 0.25 * (r0 / (disc + 1e-3) + disc / (r0 + 1e-3) + 2.0)
        mu2 = 0.25 * (r0 / (disc + 1e-3) + disc / (r0 + 1e-3) - 2.0)
        
        # Cap magnifications to prevent singular color values
        mu1 = ti.min(4.0, mu1)
        mu2 = ti.min(3.0, mu2)
        has_2 = 1
    else:
        r1 = r0
        mu1 = 1.0
        
    px1 = screen_w / 2.0 + r1 * ex
    py1 = screen_h / 2.0 + r1 * ey
    
    px2 = screen_w / 2.0 + r2 * ex
    py2 = screen_h / 2.0 + r2 * ey
    
    return ti.Vector([px1, py1]), mu1, ti.Vector([px2, py2]), mu2, has_2, z_cam

@ti.func
def project_with_lensing(p):
    return project_with_lensing_rgb(p, 1.0)

@ti.func
def is_blocked(px, py, z_cam):
    blocked = False
    if z_cam > camera_dist[0]:
        dx = px - screen_w / 2.0
        dy = py - screen_h / 2.0
        if dx*dx + dy*dy < r_shadow_pixels[0] * r_shadow_pixels[0] * 1.02:
            blocked = True
    return blocked

# =====================================================================
# Rendering Helpers
# =====================================================================
@ti.func
def draw_line_channel(p1, p2, col, weight, z_cam):
    # Draws a line segment between p1 and p2 into a specific color channel,
    # respecting event horizon occlusion.
    d = p2 - p1
    length = d.norm()
    if length < 60.0: # Skip segments that jump across the screen due to numeric wrap
        steps = ti.cast(length, ti.int32) + 1
        steps = ti.min(steps, 15)
        for step in range(steps):
            t = step / ti.max(steps - 1, 1)
            p = p1 + t * d
            px, py = ti.cast(p.x, ti.int32), ti.cast(p.y, ti.int32)
            if px >= 0 and px < screen_w and py >= 0 and py < screen_h:
                blocked = False
                if z_cam > camera_dist[0]:
                    dx = p.x - screen_w / 2.0
                    dy = p.y - screen_h / 2.0
                    if dx*dx + dy*dy < r_shadow_pixels[0] * r_shadow_pixels[0]:
                        blocked = True
                if not blocked:
                    if col.x > 0.0:
                        pixel_buffer[px, py].x += col.x * weight
                    if col.y > 0.0:
                        pixel_buffer[px, py].y += col.y * weight
                    if col.z > 0.0:
                        pixel_buffer[px, py].z += col.z * weight

@ti.func
def draw_line_lensed(p1, p2, col, weight, z_cam):
    # Non-chromatic version of line drawer
    d = p2 - p1
    length = d.norm()
    if length < 60.0:
        steps = ti.cast(length, ti.int32) + 1
        steps = ti.min(steps, 15)
        for step in range(steps):
            t = step / ti.max(steps - 1, 1)
            p = p1 + t * d
            px, py = ti.cast(p.x, ti.int32), ti.cast(p.y, ti.int32)
            if px >= 0 and px < screen_w and py >= 0 and py < screen_h:
                blocked = False
                if z_cam > camera_dist[0]:
                    dx = p.x - screen_w / 2.0
                    dy = p.y - screen_h / 2.0
                    if dx*dx + dy*dy < r_shadow_pixels[0] * r_shadow_pixels[0]:
                        blocked = True
                if not blocked:
                    pixel_buffer[px, py] += col * weight

# =====================================================================
# Simulation Kernels
# =====================================================================
@ti.kernel
def init_simulation():
    camera_dist[0] = 13.0
    time[0] = 0.0
    
    # Initialize background stars
    for i in range(num_stars):
        theta = ti.random() * 2.0 * 3.1415926
        phi = ti.acos(2.0 * ti.random() - 1.0)
        r = 50.0
        star_pos[i] = ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.cos(phi), r * ti.sin(phi) * ti.sin(theta)])
        
        # Color distribution: cool stars (blue), warm stars (yellow/red), main sequence (white)
        c_rand = ti.random()
        if c_rand < 0.2:
            star_color[i] = ti.Vector([0.7, 0.85, 1.0]) * (0.4 + ti.random() * 0.6)
        elif c_rand < 0.35:
            star_color[i] = ti.Vector([1.0, 0.8, 0.65]) * (0.4 + ti.random() * 0.6)
        else:
            star_color[i] = ti.Vector([1.0, 1.0, 1.0]) * (0.5 + ti.random() * 0.5)

    # Initialize accretion disk particles (Keplerian orbits + noise)
    for i in range(num_particles):
        u = ti.random()
        # Density peaks near the center, extends outwards
        r = 3.2 + 8.8 * u * u
        theta = ti.random() * 2.0 * 3.1415926
        y = ti.randn() * 0.08 * (r / 6.0) # height profile
        
        pos[i] = ti.Vector([r * ti.cos(theta), y, r * ti.sin(theta)])
        
        v_mag = ti.sqrt(M / (r + 1e-3))
        # Initial circular velocity
        vel[i] = ti.Vector([-v_mag * ti.sin(theta), ti.randn() * 0.015, v_mag * ti.cos(theta)])
        
        color[i] = get_temperature_color(r)
        age[i] = ti.random() * 5.0
        
        # Initialize trail history
        for h in range(H_steps):
            history[i, h] = pos[i]

    # Initialize photon sphere ring particles (extremely narrow, near c)
    for i in range(num_photon_ring):
        r = 2.8 + ti.random() * 0.35
        theta = ti.random() * 2.0 * 3.1415926
        y = ti.randn() * 0.015
        
        pr_pos[i] = ti.Vector([r * ti.cos(theta), y, r * ti.sin(theta)])
        
        # Speed of light c = 1.0 in code units
        v_mag = 1.0 
        pr_vel[i] = ti.Vector([-v_mag * ti.sin(theta), ti.randn() * 0.005, v_mag * ti.cos(theta)])
        
        pr_color[i] = ti.Vector([1.4, 1.15, 0.9])
        
        for h in range(H_steps):
            pr_history[i, h] = pr_pos[i]

    # Initialize Hawking pairs (start inactive)
    for i in range(num_hawking):
        hawking_active[i] = 0
        hawking_age[i] = 0.0

@ti.kernel
def update_history():
    # Update trail histories before physical integration
    for i in range(num_particles):
        for h in range(1, H_steps):
            idx = H_steps - h
            history[i, idx] = history[i, idx - 1]
        history[i, 0] = pos[i]
        
    for i in range(num_photon_ring):
        for h in range(1, H_steps):
            idx = H_steps - h
            pr_history[i, idx] = pr_history[i, idx - 1]
        pr_history[i, 0] = pr_pos[i]

@ti.kernel
def update_particles(dt: ti.float32):
    for i in range(num_particles):
        p = pos[i]
        v = vel[i]
        r2 = p.dot(p) + 1e-6
        r = ti.sqrt(r2)
        
        # Respawn if absorbed by event horizon or flew too far
        if r < r_event_horizon or r > 13.0:
            u = ti.random()
            new_r = 3.2 + 8.8 * u * u
            theta = ti.random() * 2.0 * 3.1415926
            y = ti.randn() * 0.08 * (new_r / 6.0)
            
            p = ti.Vector([new_r * ti.cos(theta), y, new_r * ti.sin(theta)])
            v_mag = ti.sqrt(M / (new_r + 1e-3))
            v = ti.Vector([-v_mag * ti.sin(theta), ti.randn() * 0.015, v_mag * ti.cos(theta)])
            
            pos[i] = p
            vel[i] = v
            color[i] = get_temperature_color(new_r)
            for h in range(H_steps):
                history[i, h] = p
            age[i] = 0.0
        else:
            # Physics updates: Gravity + Lense-Thirring Frame Dragging
            # 1. Schwarzschild Newtonian-like acceleration
            a_grav = - M / (r2 * r + 1e-6) * p
            
            # 2. Viscous friction causes orbital decay and spiral arms
            v -= v * 0.035 * dt
            
            # 3. Lense-Thirring Frame Dragging angular speed
            omega = 2.0 * spin * M * M / (r2 * r + 1e-6)
            v_drag = ti.Vector([-omega * p.z, 0.0, omega * p.x])
            
            # semi-implicit Euler integration
            v += a_grav * dt
            p += (v + v_drag) * dt
            
            pos[i] = p
            vel[i] = v
            age[i] += dt

@ti.kernel
def update_photon_ring(dt: ti.float32):
    for i in range(num_photon_ring):
        p = pr_pos[i]
        v = pr_vel[i]
        r2 = p.dot(p) + 1e-6
        r = ti.sqrt(r2)
        
        # Respawn if captured or escaped
        if r < r_event_horizon or r > 6.0:
            new_r = 2.8 + ti.random() * 0.35
            theta = ti.random() * 2.0 * 3.1415926
            y = ti.randn() * 0.015
            p = ti.Vector([new_r * ti.cos(theta), y, new_r * ti.sin(theta)])
            
            v_mag = 1.0 # c
            v = ti.Vector([-v_mag * ti.sin(theta), ti.randn() * 0.005, v_mag * ti.cos(theta)])
            
            pr_pos[i] = p
            pr_vel[i] = v
            for h in range(H_steps):
                pr_history[i, h] = p
        else:
            # Photon geodesic equation approximation
            L2 = p.cross(v).norm_sqr()
            a = - 1.5 * M * L2 / (r2 * r2 * r + 1e-6) * p
            
            # Frame dragging
            omega = 2.0 * spin * M * M / (r2 * r + 1e-6)
            v_drag = ti.Vector([-omega * p.z, 0.0, omega * p.x])
            
            v += a * dt
            p += (v + v_drag) * dt
            
            # Light velocity vector remains normalized to c=1
            v = v.normalized() * 1.0
            
            pr_pos[i] = p
            pr_vel[i] = v

@ti.kernel
def update_hawking(dt: ti.float32):
    for i in range(num_hawking):
        if hawking_active[i] == 1:
            # ----------------------------------------------------
            # Infalling particle (Negative energy)
            # ----------------------------------------------------
            p_in = hawking_pos[i, 0]
            v_in = hawking_vel[i, 0]
            r2_in = p_in.dot(p_in) + 1e-6
            r_in = ti.sqrt(r2_in)
            
            if r_in < r_event_horizon:
                hawking_active[i] = 0
            else:
                # Stronger pull inwards to simulate event horizon capture
                a_grav = - 1.8 * M / (r2_in * r_in + 1e-6) * p_in
                omega = 2.0 * spin * M * M / (r2_in * r_in + 1e-6)
                v_drag = ti.Vector([-omega * p_in.z, 0.0, omega * p_in.x])
                
                v_in += a_grav * dt
                p_in += (v_in + v_drag) * dt
                
                hawking_pos[i, 0] = p_in
                hawking_vel[i, 0] = v_in

            # ----------------------------------------------------
            # Escaping particle (Positive energy)
            # ----------------------------------------------------
            p_out = hawking_pos[i, 1]
            v_out = hawking_vel[i, 1]
            r2_out = p_out.dot(p_out) + 1e-6
            r_out = ti.sqrt(r2_out)
            
            if r_out > 12.0 or hawking_age[i] > 1.8:
                hawking_active[i] = 0
            else:
                # Slower gravity + radial push to escape
                a_grav = - M / (r2_out * r_out + 1e-6) * p_out + 0.35 * p_out / (r_out + 1e-3)
                omega = 2.0 * spin * M * M / (r2_out * r_out + 1e-6)
                v_drag = ti.Vector([-omega * p_out.z, 0.0, omega * p_out.x])
                
                v_out += a_grav * dt
                p_out += (v_out + v_drag) * dt
                
                hawking_pos[i, 1] = p_out
                hawking_vel[i, 1] = v_out
                
            hawking_age[i] += dt
        else:
            # Spawn new Hawking pair with quiet boil probability
            if ti.random() < 0.02:
                hawking_active[i] = 1
                hawking_age[i] = 0.0
                
                # Spawn just outside horizon surface
                theta = ti.random() * 2.0 * 3.1415926
                phi = ti.acos(2.0 * ti.random() - 1.0)
                r_spawn = r_event_horizon * 1.03
                p = ti.Vector([r_spawn * ti.sin(phi) * ti.cos(theta), r_spawn * ti.cos(phi), r_spawn * ti.sin(phi) * ti.sin(theta)])
                
                hawking_pos[i, 0] = p
                hawking_pos[i, 1] = p
                
                # Ingoing velocity
                v_in = -p.normalized() * 0.4 + ti.Vector([ti.randn()*0.1, ti.randn()*0.1, ti.randn()*0.1])
                # Outgoing velocity (with spin direction sweep)
                v_out = p.normalized() * 1.4 + ti.Vector([ti.randn()*0.15, ti.randn()*0.15, ti.randn()*0.15])
                
                hawking_vel[i, 0] = v_in
                hawking_vel[i, 1] = v_out

@ti.kernel
def compute_interactive_trail(mouse_x: ti.float32, mouse_y: ti.float32, active: ti.int32):
    if active == 1:
        # Camera ray-caster in Kerr geometry using Boyer-Lindquist RK4 geodesic solver
        mx = (mouse_x - 0.5) * (1280.0 / 800.0) * 1.05
        my = (mouse_y - 0.5) * 1.05
        
        w = w_cam[0]
        u = u_cam[0]
        v = v_cam[0]
        
        ray_dir = (w + mx * u + my * v).normalized()
        
        c_pos = camera_pos[0]
        r = c_pos.norm()
        theta = ti.acos(c_pos.y / r)
        phi = ti.atan2(c_pos.z, c_pos.x)
        
        # Coordinate basis vectors at camera position
        e_r = c_pos / r
        e_theta = ti.Vector([ti.cos(theta)*ti.cos(phi), -ti.sin(theta), ti.cos(theta)*ti.sin(phi)]).normalized()
        e_phi = ti.Vector([-ti.sin(phi), 0.0, ti.cos(phi)]).normalized()
        
        vr = ray_dir.dot(e_r)
        vtheta = ray_dir.dot(e_theta)
        vphi = ray_dir.dot(e_phi)
        
        # Boyer-Lindquist initial momenta
        pr = vr
        ptheta = r * vtheta
        Lz = r * ti.sin(theta) * vphi
        E = 1.0
        
        dt_trail = 0.15
        for step in range(300):
            sin_th = ti.sin(theta)
            cos_th = ti.cos(theta)
            sin_ph = ti.sin(phi)
            cos_ph = ti.cos(phi)
            
            x_cart = ti.sqrt(r*r + spin*spin) * sin_th * cos_ph
            y_cart = r * cos_th
            z_cart = ti.sqrt(r*r + spin*spin) * sin_th * sin_ph
            
            interactive_trail_pos[step] = ti.Vector([x_cart, y_cart, z_cart])
            
            if r < 1.21 or r > 40.0:
                for k in range(step, 300):
                    interactive_trail_pos[k] = ti.Vector([0.0, 0.0, 0.0])
                break
                
            r, theta, phi, pr, ptheta = rk4_step(r, theta, phi, pr, ptheta, Lz, E, dt_trail)
    else:
        for step in range(300):
            interactive_trail_pos[step] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def apply_mouse_pull(mouse_x: ti.float32, mouse_y: ti.float32, dt: ti.float32):
    # Applies gravity pull from mouse cursor coordinates
    w = w_cam[0]
    u = u_cam[0]
    v = v_cam[0]
    
    mx = (mouse_x - 0.5) * (1280.0 / 800.0) * 1.05
    my = (mouse_y - 0.5) * 1.05
    
    ray_dir = (w + mx * u + my * v).normalized()
    ray_org = camera_pos[0]
    
    # Disk pull
    for i in range(num_particles):
        p = pos[i]
        d = p - ray_org
        t = d.dot(ray_dir)
        p_ray = ray_org + t * ray_dir
        
        pull_vec = p_ray - p
        dist = pull_vec.norm()
        if dist < 2.5:
            # Gravitational attraction to cursor
            f_pull = 5.0 * (1.0 - dist / 2.5)
            vel[i] += pull_vec / (dist + 1e-3) * f_pull * dt

# =====================================================================
# Render and Camera Update Kernels
# =====================================================================
@ti.kernel
def update_camera(yaw: ti.float32, pitch: ti.float32):
    # Orbit position
    cy = ti.sin(pitch) * camera_dist[0]
    cx = ti.cos(pitch) * ti.sin(yaw) * camera_dist[0]
    cz = ti.cos(pitch) * ti.cos(yaw) * camera_dist[0]
    camera_pos[0] = ti.Vector([cx, cy, cz])
    
    # Screen vectors looking at the origin
    w = -camera_pos[0].normalized()
    up_global = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(w.y) > 0.99:
        up_global = ti.Vector([1.0, 0.0, 0.0])
        
    u = w.cross(up_global).normalized()
    v = u.cross(w).normalized()
    
    w_cam[0] = w
    u_cam[0] = u
    v_cam[0] = v
    
    f = 850.0
    r_shadow_pixels[0] = (r_shadow / (camera_dist[0] + 1e-3)) * f

@ti.kernel
def render_background():
    for i, j in pixel_buffer:
        # Screen UV coordinates
        uv = ti.Vector([float(i) / screen_w, float(j) / screen_h])
        # Two-octave nebular gas simulation
        n = noise(uv * 3.5 + time[0] * 0.04) * 0.65 + noise(uv * 7.0 - time[0] * 0.02) * 0.35
        # Cinematic dark violet nebula
        pixel_buffer[i, j] = ti.Vector([0.015, 0.008, 0.03]) * n

@ti.kernel
def render_stars():
    for i in range(num_stars):
        p = star_pos[i]
        c = star_color[i]
        
        # Screen projection with lensing
        p1, mu1, p2, mu2, has_2, z_cam = project_with_lensing(p)
        
        # Center direction for star distortion orientation
        r_dir1 = p1 - ti.Vector([screen_w / 2.0, screen_h / 2.0])
        r_dist1 = r_dir1.norm()
        t1 = ti.Vector([-r_dir1.y, r_dir1.x]) / (r_dist1 + 1e-3)
        
        # Primary image: stretch into curved Einstein arcs based on magnification
        stretch1 = ti.min(6.0, (mu1 - 1.0) * 2.0)
        if stretch1 > 0.15:
            draw_line_lensed(p1 - t1 * stretch1, p1 + t1 * stretch1, c * 0.75, 0.35 / (stretch1 + 1e-3), z_cam)
        else:
            px, py = ti.cast(p1.x, ti.int32), ti.cast(p1.y, ti.int32)
            if px >= 0 and px < screen_w and py >= 0 and py < screen_h:
                if not is_blocked(p1.x, p1.y, z_cam):
                    pixel_buffer[px, py] += c * 0.75

        # Secondary image (flipped, highly compressed on other side of lens)
        if has_2 == 1 and mu2 > 0.05:
            r_dir2 = p2 - ti.Vector([screen_w / 2.0, screen_h / 2.0])
            r_dist2 = r_dir2.norm()
            t2 = ti.Vector([-r_dir2.y, r_dir2.x]) / (r_dist2 + 1e-3)
            stretch2 = ti.min(5.0, mu2 * 1.5)
            draw_line_lensed(p2 - t2 * stretch2, p2 + t2 * stretch2, c * 0.5, 0.25 / (stretch2 + 1e-3), z_cam)

@ti.kernel
def draw_horizon():
    # Draw absolute black silhouette overlay
    r_sh = r_shadow_pixels[0]
    for i, j in pixel_buffer:
        dx = float(i) - screen_w / 2.0
        dy = float(j) - screen_h / 2.0
        if dx*dx + dy*dy < r_sh * r_sh:
            pixel_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def render_particles_pass(weight: ti.float32):
    for i in range(num_particles):
        col = color[i]
        
        # Doppler Beaming Asymmetry
        # Particle moving towards camera becomes brighter (blue-shifted)
        v_rel = vel[i].dot(w_cam[0])
        doppler = 1.0 + 1.35 * v_rel
        doppler = ti.max(0.12, ti.min(3.0, doppler))
        
        final_color = col * doppler
        
        for h in range(H_steps - 1):
            p_prev = history[i, h + 1]
            p_curr = history[i, h]
            
            # Chromatic aberration projection: separate Red/Green/Blue lensing scaling factors
            p1_prev_r, mu1_prev_r, p2_prev_r, mu2_prev_r, has2_prev, z_cam = project_with_lensing_rgb(p_prev, 1.01)
            p1_curr_r, mu1_curr_r, p2_curr_r, mu2_curr_r, has2_curr, _ = project_with_lensing_rgb(p_curr, 1.01)
            
            p1_prev_g, mu1_prev_g, p2_prev_g, mu2_prev_g, _, _ = project_with_lensing_rgb(p_prev, 1.00)
            p1_curr_g, mu1_curr_g, p2_curr_g, mu2_curr_g, _, _ = project_with_lensing_rgb(p_curr, 1.00)
            
            p1_prev_b, mu1_prev_b, p2_prev_b, mu2_prev_b, _, _ = project_with_lensing_rgb(p_prev, 0.99)
            p1_curr_b, mu1_curr_b, p2_curr_b, mu2_curr_b, _, _ = project_with_lensing_rgb(p_curr, 0.99)
            
            # Draw primary images (chromatic split)
            draw_line_channel(p1_prev_r, p1_curr_r, ti.Vector([final_color.x, 0.0, 0.0]), mu1_curr_r * weight, z_cam)
            draw_line_channel(p1_prev_g, p1_curr_g, ti.Vector([0.0, final_color.y, 0.0]), mu1_curr_g * weight, z_cam)
            draw_line_channel(p1_prev_b, p1_curr_b, ti.Vector([0.0, 0.0, final_color.z]), mu1_curr_b * weight, z_cam)
            
            # Draw secondary images (chromatic split, dimmed slightly)
            if has2_curr == 1 and mu2_curr_g > 0.02:
                draw_line_channel(p2_prev_r, p2_curr_r, ti.Vector([final_color.x, 0.0, 0.0]), mu2_curr_r * weight * 0.7, z_cam)
                draw_line_channel(p2_prev_g, p2_curr_g, ti.Vector([0.0, final_color.y, 0.0]), mu2_curr_g * weight * 0.7, z_cam)
                draw_line_channel(p2_prev_b, p2_curr_b, ti.Vector([0.0, 0.0, final_color.z]), mu2_curr_b * weight * 0.7, z_cam)

@ti.kernel
def render_photon_ring_pass(weight: ti.float32):
    for i in range(num_photon_ring):
        col = pr_color[i]
        
        # High Doppler beaming at photon sphere due to relativistic speeds
        v_rel = pr_vel[i].dot(w_cam[0])
        doppler = 1.0 + 1.6 * v_rel
        doppler = ti.max(0.1, ti.min(4.0, doppler))
        
        final_color = col * doppler
        
        for h in range(H_steps - 1):
            p_prev = pr_history[i, h + 1]
            p_curr = pr_history[i, h]
            
            p1_prev_r, mu1_prev_r, p2_prev_r, mu2_prev_r, has2_prev, z_cam = project_with_lensing_rgb(p_prev, 1.008)
            p1_curr_r, mu1_curr_r, p2_curr_r, mu2_curr_r, has2_curr, _ = project_with_lensing_rgb(p_curr, 1.008)
            
            p1_prev_g, mu1_prev_g, p2_prev_g, mu2_prev_g, _, _ = project_with_lensing_rgb(p_prev, 1.00)
            p1_curr_g, mu1_curr_g, p2_curr_g, mu2_curr_g, _, _ = project_with_lensing_rgb(p_curr, 1.00)
            
            p1_prev_b, mu1_prev_b, p2_prev_b, mu2_prev_b, _, _ = project_with_lensing_rgb(p_prev, 0.992)
            p1_curr_b, mu1_curr_b, p2_curr_b, mu2_curr_b, _, _ = project_with_lensing_rgb(p_curr, 0.992)
            
            # Primary white/gold ring
            draw_line_channel(p1_prev_r, p1_curr_r, ti.Vector([final_color.x, 0.0, 0.0]), mu1_curr_r * weight, z_cam)
            draw_line_channel(p1_prev_g, p1_curr_g, ti.Vector([0.0, final_color.y, 0.0]), mu1_curr_g * weight, z_cam)
            draw_line_channel(p1_prev_b, p1_curr_b, ti.Vector([0.0, 0.0, final_color.z]), mu1_curr_b * weight, z_cam)
            
            # Secondary ring
            if has2_curr == 1 and mu2_curr_g > 0.02:
                draw_line_channel(p2_prev_r, p2_curr_r, ti.Vector([final_color.x, 0.0, 0.0]), mu2_curr_r * weight * 0.8, z_cam)
                draw_line_channel(p2_prev_g, p2_curr_g, ti.Vector([0.0, final_color.y, 0.0]), mu2_curr_g * weight * 0.8, z_cam)
                draw_line_channel(p2_prev_b, p2_curr_b, ti.Vector([0.0, 0.0, final_color.z]), mu2_curr_b * weight * 0.8, z_cam)

@ti.func
def draw_soft_point(p, col, intensity, z_cam):
    px, py = ti.cast(p.x, ti.int32), ti.cast(p.y, ti.int32)
    blocked = False
    if z_cam > camera_dist[0]:
        dx = p.x - screen_w / 2.0
        dy = p.y - screen_h / 2.0
        if dx*dx + dy*dy < r_shadow_pixels[0] * r_shadow_pixels[0]:
            blocked = True
    if not blocked:
        for dx_idx in range(3):
            dx = dx_idx - 1
            for dy_idx in range(3):
                dy = dy_idx - 1
                x = px + dx
                y = py + dy
                if x >= 0 and x < screen_w and y >= 0 and y < screen_h:
                    w = 1.0 / (float(dx*dx + dy*dy) + 1.0)
                    pixel_buffer[x, y] += col * intensity * w

@ti.kernel
def render_hawking_pass():
    for i in range(num_hawking):
        if hawking_active[i] == 1:
            age_f = hawking_age[i]
            
            # 1. Draw Infalling Particle (deep red/pink)
            p_in = hawking_pos[i, 0]
            col_in = ti.Vector([1.2, 0.15, 0.3]) * ti.max(0.0, 1.0 - age_f / 1.5)
            p1_in, mu1_in, _, _, _, z_cam_in = project_with_lensing(p_in)
            draw_soft_point(p1_in, col_in, 8.0 * mu1_in, z_cam_in)

            # 2. Draw Escaping Particle (electric cyan)
            p_out = hawking_pos[i, 1]
            col_out = ti.Vector([0.15, 0.8, 1.3]) * ti.max(0.0, 1.0 - age_f / 1.8)
            p1_out, mu1_out, _, _, _, z_cam_out = project_with_lensing(p_out)
            draw_soft_point(p1_out, col_out, 10.0 * mu1_out, z_cam_out)
                    
            # 3. Draw Entanglement Link (purple line, fades very rapidly)
            if age_f < 0.25:
                w_conn = (1.0 - age_f / 0.25) * 0.65
                col_conn = ti.Vector([0.9, 0.4, 1.1]) * w_conn
                draw_line_lensed(p1_in, p1_out, col_conn, 1.5, z_cam_in)

@ti.kernel
def render_interactive_trail():
    # Render glowing geodesic trail if active
    if interactive_trail_pos[0].norm_sqr() > 1e-3:
        for step in range(299):
            p1 = interactive_trail_pos[step]
            p2 = interactive_trail_pos[step + 1]
            if p1.norm_sqr() > 1e-3 and p2.norm_sqr() > 1e-3:
                # Probes are integrated directly (they are the ray itself, so we project without lensing)
                d1 = p1 - camera_pos[0]
                z_cam1 = d1.dot(w_cam[0])
                x_cam1 = d1.dot(u_cam[0])
                y_cam1 = d1.dot(v_cam[0])
                
                d2 = p2 - camera_pos[0]
                z_cam2 = d2.dot(w_cam[0])
                x_cam2 = d2.dot(u_cam[0])
                y_cam2 = d2.dot(v_cam[0])
                
                f = 850.0
                px1 = screen_w / 2.0 + (x_cam1 / (z_cam1 + 1e-3)) * f
                py1 = screen_h / 2.0 + (y_cam1 / (z_cam1 + 1e-3)) * f
                
                px2 = screen_w / 2.0 + (x_cam2 / (z_cam2 + 1e-3)) * f
                py2 = screen_h / 2.0 + (y_cam2 / (z_cam2 + 1e-3)) * f
                
                # Glowing electric amber/gold trace (white-gold/pale amber color)
                draw_line_lensed(ti.Vector([px1, py1]), ti.Vector([px2, py2]), ti.Vector([1.3, 1.0, 0.4]), 2.2, z_cam1)

# =====================================================================
# Post-Processing Shader Passes
# =====================================================================
@ti.kernel
def extract_bright():
    for i, j in pixel_buffer:
        color = pixel_buffer[i, j]
        # Luminance calculation
        lum = 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z
        if lum > 0.45:
            bright_buffer[i, j] = color * (lum - 0.45)
        else:
            bright_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def blur_h(radius: ti.int32):
    for i, j in bright_buffer:
        sum_col = ti.Vector([0.0, 0.0, 0.0])
        weight_sum = 0.0
        for k_idx in range(2 * radius + 1):
            k = k_idx - radius
            x = i + k
            if x >= 0 and x < screen_w:
                w = ti.exp(-float(k*k) / (2.0 * float(radius*radius) / 9.0))
                sum_col += bright_buffer[x, j] * w
                weight_sum += w
        blur_buffer_temp[i, j] = sum_col / (weight_sum + 1e-3)

@ti.kernel
def blur_v(radius: ti.int32):
    for i, j in blur_buffer_temp:
        sum_col = ti.Vector([0.0, 0.0, 0.0])
        weight_sum = 0.0
        for k_idx in range(2 * radius + 1):
            k = k_idx - radius
            y = j + k
            if y >= 0 and y < screen_h:
                w = ti.exp(-float(k*k) / (2.0 * float(radius*radius) / 9.0))
                sum_col += blur_buffer_temp[i, y] * w
                weight_sum += w
        blur_buffer[i, j] = sum_col / (weight_sum + 1e-3)

@ti.kernel
def apply_tonemap_and_bloom(bloom_strength: ti.float32, exposure: ti.float32):
    for i, j in pixel_buffer:
        color = pixel_buffer[i, j] + blur_buffer[i, j] * bloom_strength
        color *= exposure
        
        # Reinhard tonemapping (compression of high ranges)
        color.x = color.x / (color.x + 1.0)
        color.y = color.y / (color.y + 1.0)
        color.z = color.z / (color.z + 1.0)
        
        pixel_buffer[i, j] = color

# =====================================================================
# Procedural Ambient Sound Generator (Interstellar Organ & Clock-Tick)
# =====================================================================
def play_cosmic_drone():
    try:
        import pygame
        import math
        
        # Initialize pygame mixer (stereo mode by default)
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        actual_freq, actual_size, actual_channels = pygame.mixer.get_init()
        
        # 16-second looping chord progression (Am - F - C - G, 4 seconds each)
        duration = 16.0
        sample_rate = 44100
        num_samples = int(duration * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        chords = [
            [110.0, 164.8, 220.0, 261.6], # A minor (A2, E3, A3, C4)
            [87.3,  130.8, 174.6, 220.0], # F major (F2, C3, F3, A3)
            [65.4,  98.0,  130.8, 164.8], # C major (C2, G2, C3, E3)
            [98.0,  146.8, 196.0, 246.9]  # G major (G2, D3, G3, B3)
        ]
        
        for i in range(num_samples):
            t = i / sample_rate
            chord_idx = int(t / 4.0) % 4
            t_chord = t % 4.0
            
            # Smooth crescendo and decrescendo envelope per chord
            envelope = math.sin(math.pi * t_chord / 4.0)
            
            # Synthesize pipe organ voice for current chord
            val = 0.0
            frequencies = chords[chord_idx]
            for f in frequencies:
                # Pipe organ harmonics stack
                val += math.sin(2.0 * math.pi * f * t)
                val += 0.5 * math.sin(2.0 * math.pi * (2.0 * f) * t)
                val += 0.25 * math.sin(2.0 * math.pi * (3.0 * f) * t)
                val += 0.12 * math.sin(2.0 * math.pi * (4.0 * f) * t)
                
            val = (val / len(frequencies)) * envelope
            
            # Add rhythmic "time ticking" click every 1.25 seconds (Miller's planet reference)
            tick_t = t % 1.25
            tick_val = 0.0
            if tick_t < 0.02:
                tick_val = math.sin(2.0 * math.pi * 1200.0 * tick_t) * math.exp(-300.0 * tick_t) * 0.22
                
            audio_data[i] = val * 0.65 + tick_val
            
        # Scale to 16-bit signed integer range
        audio_int = (audio_data * 14000).astype(np.int16)
        
        # Format the array correctly for mono or stereo output
        sound_array = audio_int.reshape(-1, 1)
        if actual_channels == 2:
            sound_array = np.column_stack((audio_int, audio_int))
            
        sound = pygame.sndarray.make_sound(sound_array)
        sound.play(loops=-1)
        print(f"Interstellar-style organ track initialized procedurally ({actual_channels} channels).")
    except Exception as e:
        print(f"Background music skipped: {e}")

# =====================================================================
# Main Simulation Execution
# =====================================================================
def main():
    # Attempt to play cosmic drone sound
    play_cosmic_drone()

    print("=====================================================================")
    print("                 KERR BLACK HOLE SIMULATOR (a* = 0.98)               ")
    print("=====================================================================")
    print(" Controls:")
    print("  • Mouse Move        : Project real-time photon geodesics")
    print("  • Left Mouse Click  : Gravitationally pull accretion filaments")
    print("  • A / D / LEFT/RGHT : Rotate camera horizontally (yaw)")
    print("  • W / S / UP/DOWN   : Rotate camera vertically (pitch)")
    print("  • SPACE             : Pause / Resume simulation")
    print("  • '=' / '-' keys    : Speed up / Slow down time")
    print("  • R key             : Reset simulation")
    print("=====================================================================")

    window = ti.ui.Window("Einstein's Trampoline (Kerr a*=0.98)", (screen_w, screen_h), vsync=True)
    canvas = window.get_canvas()

    init_simulation()

    # Initial camera parameters
    yaw = 0.0
    pitch = 0.35 # ~20 degrees above disk plane
    time_scale = 1.0
    paused = False

    while window.running:
        # Handle inputs
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.SPACE:
                paused = not paused
                print("Simulation " + ("PAUSED" if paused else "RESUMED"))
            elif event.key == 'r' or event.key == 'R':
                init_simulation()
                yaw = 0.0
                pitch = 0.35
                print("Simulation reset.")
            elif event.key == '=' or event.key == '+':
                time_scale = min(4.0, time_scale * 1.5)
                print(f"Time scale increased to {time_scale:.2f}x")
            elif event.key == '-' or event.key == '_':
                time_scale = max(0.1, time_scale / 1.5)
                print(f"Time scale decreased to {time_scale:.2f}x")

        # Camera Keyboard controls
        if window.is_pressed('a') or window.is_pressed(ti.ui.LEFT):
            yaw -= 0.02
        if window.is_pressed('d') or window.is_pressed(ti.ui.RIGHT):
            yaw += 0.02
        if window.is_pressed('w') or window.is_pressed(ti.ui.UP):
            pitch = min(1.35, pitch + 0.015)
        if window.is_pressed('s') or window.is_pressed(ti.ui.DOWN):
            pitch = max(-1.35, pitch - 0.015)

        # Slow automatic orbit if not dragging/steering
        if not (window.is_pressed('a') or window.is_pressed('d') or window.is_pressed(ti.ui.LEFT) or window.is_pressed(ti.ui.RIGHT)):
            yaw += 0.0018

        # Calculate time-step
        dt = 0.025 * time_scale if not paused else 0.0

        # Update physics
        if not paused:
            time[0] += dt
            update_history()
            update_particles(dt)
            update_photon_ring(dt)
            update_hawking(dt)

        # Mouse Interactions
        mouse_x, mouse_y = window.get_cursor_pos()
        # Verify mouse coordinates are within screen
        mouse_active = 0
        if 0.0 <= mouse_x <= 1.0 and 0.0 <= mouse_y <= 1.0:
            # Only activate geodesic probe if mouse approaches the black hole region (radius ~0.26 of screen center)
            dx = mouse_x - 0.5
            dy = mouse_y - 0.5
            if dx*dx + dy*dy < 0.26 * 0.26:
                mouse_active = 1
            if window.is_pressed(ti.ui.LMB):
                apply_mouse_pull(mouse_x, mouse_y, dt)
        
        # Calculate interactive geodesic trace
        compute_interactive_trail(mouse_x, mouse_y, mouse_active)

        # Update projection coordinate matrices
        update_camera(yaw, pitch)

        # Render pass sequence
        render_background()
        render_stars()
        draw_horizon()
        render_particles_pass(0.13)      # Splat disk filaments
        render_photon_ring_pass(0.35)    # Splat photon ring
        render_hawking_pass()            # Splat Hawking pairs
        render_interactive_trail()       # Splat mouse geodesic

        # Post-process: Extract bright and apply Blur + Tonemapping
        extract_bright()
        blur_h(12)
        blur_v(12)
        apply_tonemap_and_bloom(0.45, 1.25) # bloom strength, exposure

        # Draw to screen canvas
        canvas.set_image(pixel_buffer)
        
        # Render clean HUD panel
        gui = window.get_gui()
        with gui.sub_window("METRICS & CONTROLS", 0.02, 0.02, 0.36, 0.44):
            gui.text(f"Black Hole Mass : {M:.1f} M")
            gui.text(f"Black Hole Spin : {spin:.2f} a*")
            gui.text("----------------------------------------")
            gui.text("Shadow Boundary : Lensed Horizon Silhouette")
            gui.text("Einstein Lensing: Curved Arcs & Secondary Flips")
            gui.text("Photon Sphere   : Razor-Thin Luminous Halo")
            gui.text("Accretion Flow  : Incandescent Plasma Fibers")
            gui.text("Hawking Pairs   : Infalling Red / Escaping Blue")
            gui.text("Cursor Pointer  : Kerr Geodesic Light Ray")
            gui.text("----------------------------------------")
            gui.text("Camera Orbit    : W/A/S/D or Arrow Keys")
            gui.text("Left-Click Drag : Swirls plasma via gravity pull")
            gui.text("Screen Dragging : Shows violet chromatic dispersion")

        window.show()

if __name__ == "__main__":
    main()
