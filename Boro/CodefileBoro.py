import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import ndimage


#Physical constants
epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
mu_0 = 4 * np.pi * 1e-7        # H/m (permeability of vacuum)
c = 1 / np.sqrt(mu_0 * epsilon_0)  # Speed of light in vacuum

epsilon= epsilon_0

#initialization of the fields!
n=3
Ex= np.zeros((n,n))
Ey= np.zeros((n,n))
Hz= np.zeros((n,n))

nx, ny = Hz.shape
dx=1
dy=1
CFL=1
dt = CFL / ( c * np.sqrt((1 /dx ** 2) + (1 /dy ** 2)))  # time step from spatial disc. & CF#for now dx and dy uniform

def update_equations(Ex, Ey, Hz):
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
    Cey = -dt/ (epsilon * dx)
    Chz = -dt/(dx*dy*mu_0)


    # Update Hz field (curl of E)
    Hz[1:nx, 1:ny] +=  Chz[1:nx,1:ny]*(
        (Ey[1:nx, 1:ny] - Ey[:nx-1, 1:ny])*dy- 
        (Ex[1:nx, :ny-1] - Ex[1:nx, 1:ny])*dx
    )
    
    # Update Ex field (y-derivative of Hz)
    Ex[:,:ny-1] += Cex[:, :ny-1] * (
        Hz[:, 1:ny] - Hz[:, :ny-1]
    )
    
    # Update Ey field (x-derivative of Hz)
    Ey[:nx-1, :] += Cey[:nx-1, :] * (
        Hz[1:nx, :] - Hz[:nx-1, :]
    )
    return Ex, Ey, Hz
#we need to create a timeframe
t=100
nt=np.floor(t*dt)
#now the update times
tint=np.linspace(0,t,100)
print(tint)
Ex[2,2]=1



I=np.sqrt((Ey*Hz)**2+(Ex*Hz)**2)
#Visualization
fig,axis = plt.subplots()

pcm= axis.pcolormesh(I,cmap=plt.cm.jet)



#simulating


for t in tint:

    Ex,Ey,Hz= update_equations(Ex,Ey,Hz)

    I=np.sqrt((Ey*Hz)**2+(Ex*Hz)**2)
    pcm.set_array(I)
    plt.pause(0.01)


plot.show()



t0=1
sigma=1
A=1
