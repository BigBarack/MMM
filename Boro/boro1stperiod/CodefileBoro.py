import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
Nx, Ny = 200, 200         # Grid size
n_pml = 20                # PML thickness (in cells)
steps = 500              # Number of time steps

# --- Physical Constants ---
c0 = 299792458           # Speed of light
mu0 = 4e-7 * np.pi       # Permeability
eps0 = 1 / (mu0 * c0**2) # Permittivity

# --- Grid Spacing and Time Step ---
dx = dy = 1e-3
CFL=1
dt = CFL / (
                c0 * np.sqrt((1 / dx ** 2) + (1 / dy ** 2)))      # Courant condition

# --- Fields ---
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny + 1))
Hy = np.zeros((Nx + 1, Ny))

# --- PML Parameters ---
sigma_max = 2.0
kappa_max = 5.0
order = 3

sigma_x = np.zeros(Nx)
kappa_x = np.ones(Nx)
sigma_y = np.zeros(Ny)
kappa_y = np.ones(Ny)

# Grading profiles for x-direction
for i in range(n_pml):
    sigma_x[i] = sigma_x[-i-1] = sigma_max * ((n_pml - i) / n_pml)**order
    kappa_x[i] = kappa_x[-i-1] = 1 + (kappa_max - 1) * ((n_pml - i) / n_pml)**order

# Grading profiles for y-direction
for j in range(n_pml):
    sigma_y[j] = sigma_y[-j-1] = sigma_max * ((n_pml - j) / n_pml)**order
    kappa_y[j] = kappa_y[-j-1] = 1 + (kappa_max - 1) * ((n_pml - j) / n_pml)**order

# Expand to 2D arrays
sigma_x_2d = np.tile(sigma_x[:, np.newaxis], (1, Ny))
sigma_y_2d = np.tile(sigma_y[np.newaxis, :], (Nx, 1))
kappa_x_2d = np.tile(kappa_x[:, np.newaxis], (1, Ny))
kappa_y_2d = np.tile(kappa_y[np.newaxis, :], (Nx, 1))

# --- Coefficients ---
Ca = (1 - dt * sigma_x_2d / (2 * eps0 * kappa_x_2d)) / (1 + dt * sigma_x_2d / (2 * eps0 * kappa_x_2d))
Cb = dt / (eps0 * dx) / (1 + dt * sigma_x_2d / (2 * eps0 * kappa_x_2d))

Da = (1 - dt * sigma_y_2d / (2 * eps0 * kappa_y_2d)) / (1 + dt * sigma_y_2d / (2 * eps0 * kappa_y_2d))
Db = dt / (eps0 * dy) / (1 + dt * sigma_y_2d / (2 * eps0 * kappa_y_2d))

# --- Source ---
source_x, source_y = Nx // 3, Ny // 3
pulse_duration = 1  # number of time steps the pulse is active

# --- Plotting Setup ---
fig, ax = plt.subplots()
img = ax.imshow(Ez.T, cmap='RdBu', origin='lower', vmin=-1e-3, vmax=1e-3)
ax.set_title("Ez Field with UPML")

# --- Main FDTD Loop ---
def update(n):
    global Ez, Hx, Hy

    # Update H fields (magnetic field)
    Hx[:, 1:-1] -= dt / mu0 * (Ez[:, 1:] - Ez[:, :-1]) / dy
    Hy[1:-1, :] += dt / mu0 * (Ez[1:, :] - Ez[:-1, :]) / dx

    # Compute curl of H fields (magnetic field components)
    curl_H = (Hy[1:Nx+1, :Ny] - Hy[:Nx, :Ny]) / dx - (Hx[:Nx, 1:Ny+1] - Hx[:Nx, :Ny]) / dy

    # Update E field using Da, Db coefficients
    Ez[1:-1, 1:-1] = Da[1:-1, 1:-1] * Ez[1:-1, 1:-1] + \
                     Db[1:-1, 1:-1] * curl_H[1:-1, 1:-1]

    # Inject source
    frequency = 2e9  # 2 GHz, for example
    omega = 2 * np.pi * frequency
    pulse = np.sin(omega * n * dt)  # Sinusoidal wave
    Ez[source_x, source_y] += pulse * 1e-3  # Inject source

    # Update plot
    img.set_array(Ez.T)
    return [img]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
plt.show()
