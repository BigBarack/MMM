import numpy as np


#Physical constants
Eps0=
Mu=

Epsilon=


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
    Cex = dt / (epsilon * dy)
    Cey = -dt / (epsilon * dx)
    Chz = dt / mu
    
    # Get grid dimensions
    nx, ny = Hz.shape
    
    # Update Hz field (curl of E)
    Hz[1:nx, 1:ny] += Chz[1:nx, 1:ny] * (
        (Ex[1:nx, 1:ny] - Ex[1:nx, :ny-1]) / dy[1:nx, 1:ny] -
        (Ey[1:nx, 1:ny] - Ey[:nx-1, 1:ny]) / dx[1:nx, 1:ny]
    )
    
    # Update Ex field (y-derivative of Hz)
    Ex[:, :ny-1] += Cex[:, :ny-1] * (
        Hz[:, 1:ny] - Hz[:, :ny-1]
    )
    
    # Update Ey field (x-derivative of Hz)
    Ey[:nx-1, :] += Cey[:nx-1, :] * (
        Hz[1:nx, :] - Hz[:nx-1, :]
    )
    
    return Ex, Ey, Hz