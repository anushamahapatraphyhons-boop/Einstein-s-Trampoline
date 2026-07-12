# Einstein's Trampoline

An interactive, real-time 3D simulation of a spinning **Kerr Black Hole ($a^* = 0.98$)** written in Python and compiled on Vulkan/CPU using Taichi. Includes a procedurally generated ambient pipe organ synthesizer soundtrack.

---

### 🎬 Live Demo Preview



![Einstein's Trampoline Demo](demo.gif)

---

##  Physics Under the Hood

This project is powered by actual relativistic physics equations integrated in real-time on your GPU:

* **Kerr Geometry Spacetime:** Models a rotating black hole of Mass $M=1.0$ and Spin $a^*=0.98$, solving the event horizon boundary.
* **Boyer-Lindquist RK4 Geodesics:** The interactive cursor probe traces actual light paths (null geodesics) in curved spacetime by solving the canonical equations of motion using a **4th-order Runge-Kutta (RK4) integrator**.
* **Lense-Thirring Frame Dragging:** Accretion disk particles, photon sphere orbits, and Hawking pairs are dragged in the azimuth direction ($\phi$) due to the rotation of space itself.
* **Einstein Lensing & Chromatic Dispersion:** Simulates primary and secondary lensed images (creating the lensed accretion disk rings above and below the horizon). When moving the camera, color channels separate to show the **violet/magenta chromatic aberration streaks**.
* **Hawking Radiation Fluctuation:** Simulates particle-antiparticle pairs boiling near the horizon ($r \approx 1.05\ r_+$) where the infalling (red) particle plunges inside and the escaping (blue-white) particle flies away.

---

## Procedural Ambient Soundtrack

The simulation includes a procedural **Interstellar-style pipe organ drone track** generated programmatically (requires no external `.mp3` or `.wav` files):
* **Cathedral Organ Harmonics:** Stacks detuned fundamental, 2nd, 3rd, and 4th octaves to build a rich organ voice.
* **Progression Loop:** Plays a looping chord progression: `A minor` $\to$ `F major` $\to$ `C major` $\to$ `G major` (4 seconds per chord) with soft crescendo/decrescendo volume envelopes.
* **Miller's Clock Tick:** Adds a high-frequency decaying pulse clicking every **1.25 seconds** to symbolize gravitational time dilation.

---

##  Quick Start

### 1. Install Dependencies
Make sure you have Python 3.10-3.12 installed. Install the required libraries:
```bash
pip install taichi numpy pygame
```

### 2. Run the Simulation
Launch the script:
```bash
python sim.py
```

---

##  Interactive Controls

| Control | Action |
| --- | --- |
| **Mouse Hover** | Project and trace a real-time **Kerr geodesic photon trail** (Gold) |
| **Left Click & Drag** | Warp the accretion coordinates, physically swirling plasma filaments |
| **W / S / Up / Down** | Pitch camera vertically |
| **A / D / Left / Right** | Yaw camera horizontally (automatic slow cinematic drift when idle) |
| **SPACE** | Pause / Resume simulation timeline |
| **`+` / `-` keys** | Speed up / Slow down simulation speed |
| **R key** | Reset camera position and clear particles |
