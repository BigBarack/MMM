import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage

# Physical constants
epsilon_0 = 8.8541878128e-12  # F/m
mu_0 = 4 * np.pi * 1e-7        # H/m
c = 1 / np.sqrt(mu_0 * epsilon_0)  # Speed of light

# Grid setup
n = 100
Ex = np.zeros((n, n))
Ey = np.zeros((n, n))
Hz = np.zeros((n, n))

nx, ny = Hz.shape
dx = dy = 1
CFL = 1
dt = CFL / (c * np.sqrt((1/dx**2) + (1/dy**2)))

# Initialize fields with a pulse
def gaussian_pulse(t, A=1, tc=30, sigma=10):
    return A * np.exp(-((t - tc)**2) / (2 * sigma**2))


# Create figure
fig, ax = plt.subplots()
I = np.sqrt((Ey*Hz)**2 + (Ex*Hz)**2)
mesh = ax.pcolormesh(I, cmap=plt.cm.jet, shading='auto')
plt.colorbar(mesh)

def init():
    mesh.set_array(Ex.ravel())
    return [mesh]

def update(frame):
    global Ex, Ey, Hz
    
    # Update equations
    # Calculate update coefficients
    Cex = dt / (epsilon_0 * dy)
    Cey = -dt / (epsilon_0 * dx)
    Chz = -dt/(dx*dy*mu_0)

    # Update Hz field (curl of E)
    Hz[1:nx, 1:ny] += Chz*(
        (Ey[1:nx, 1:ny] - Ey[:nx-1, 1:ny])*dy - 
        (Ex[1:nx, :ny-1] - Ex[1:nx, 1:ny])*dx
    )
    
    # Update Ex field (y-derivative of Hz)
    Ex[:, :ny-1] += Cex * (Hz[:, 1:ny] - Hz[:, :ny-1])
    
    # Update Ey field (x-derivative of Hz)
    Ey[:nx-1, :] += Cey * (Hz[1:nx, :] - Hz[:nx-1, :])
    
    Ex[n//2,n//2] += gaussian_pulse(frame)

    Ex[0, :] = -Ex[0, :]
    Ex[-1, :] =-Ex[-1, :]
    Ex[:, 0] = -Ex[:, 0]
    Ex[:, -1] = -Ex[:, -1]
    Ey[0, :] = -Ey[0, :]
    Ey[-1, :] = -Ey[-1, :]
    Ey[:, 0] = -Ey[:, 0] 
    Ey[:, -1] = -Ey[:, -1]
    Hz[:,-1] = 0
    Hz[:,0] = 0
    Hz[-1,:] = 0
    Hz[0,:] = 0
    # Update visualization
    mesh.set_array(Ex.ravel())
    return [mesh]

# Create animation
ani = FuncAnimation(fig, update, frames=1000, init_func=init,
                    blit=True, interval=10)

plt.show()
