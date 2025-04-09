import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import ndimage


#Physical constants
epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
mu_0 = 4 * np.pi * 1e-7        # H/m (permeability of vacuum)
c = 1 / np.sqrt(mu_0 * epsilon_0)  # Speed of light in vacuum

epsilon= epsilon_0*np.ones((100,100))

#initialization of the fields!

Ex= np.zeros((100,100))
Ey= np.zeros((100,100))
Hz= np.zeros((100,100))

def update_equations(Ex, Ey, Hz, dx, dy, dt, epsilon, mu):
    """
    Update the TE mode field components (Ex, Ey, Hz) using non-uniform Yee algorithm.
    
    Parameters:
        Ex, Ey, Hz: 2D numpy arrays of field components
        dx, dy: 2D numpy arrays of grid spacings in x and y directions
        dt: time step (scalar)
        epsilon: 2D numpy array of permittivity distribution
        mu: 2D numpy array of permeability distribution
        
    Returns:
        Updated Ex, Ey, Hz fields
    """
    # Calculate update coefficients
    Cex = dt*dx / (epsilon * dy)
    Cey = -dt*dy/ (epsilon * dx)
    Chz = -dt/(dx*dy*mu)
    
    # Get grid dimensions
    nx, ny = Hz.shape
    

    Eyn = Ey*dy
    Exn = Ex*dx
    # Update Hz field (curl of E)
    Hz[1:nx, 1:ny] +=  Chz[1:nx,1:ny]*(
        (Eyn[1:nx, 1:ny] - Eyn[:nx-1, 1:ny])- 
        (Exn[1:nx, :ny-1] - Exn[1:nx, 1:ny])
    )
    
    # Update Ex field (y-derivative of Hz)
    Exn[:,:ny-1] += Cex[:, :ny-1] * (
        Hz[:, 1:ny] - Hz[:, :ny-1]
    )
    
    # Update Ey field (x-derivative of Hz)
    Ey[:nx-1, :] += Cey[:nx-1, :] * (
        Hz[1:nx, :] - Hz[:nx-1, :]
    )
    
    return Ex, Ey, Hz



