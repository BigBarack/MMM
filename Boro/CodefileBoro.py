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

Ex= np.zeros((100,100))
Ey= np.zeros((100,100))
Hz= np.zeros((100,100))

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
    Cey = -dt*dy/ (epsilon * dx)
    Chz = -dt/(dx*dy*mu_0)
    

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

t0=1
sigma=1
A=1
def iterate():
        receiver1 = np.zeros((nt, 1))
        receiver2 = np.zeros((nt, 1))
        receiver3 = np.zeros((nt, 1))
        timeseries = np.zeros((nt, 1))

        fig, ax = plt.subplots()
        plt.axis('equal')
        plt.xlim([1, nx + 1])
        plt.ylim([1, ny + 1])
        movie = []

        for it in range(0, self.nt):
            t = (it - 1) * self.dt
            timeseries[it, 0] = t
            print('%d/%d' % (it, self.nt))  # Loading bar while sim is running

            # Update source for new time
            source = A*np.exp(
                -((t - t0) ** 2) / 2 * sigma**2)
            self.p[self.x_source, self.y_source] += source  # Adding source term
            self.step_sit_sip()  # Propagate over one time step

            # Store p field at receiver locations
            """receiver1[it] = self.p[self.x_rec1, self.y_rec1]
            receiver2[it] = self.p[self.x_rec2, self.y_rec2]
            receiver3[it] = self.p[self.x_rec3, self.y_rec3]
            """
            # Create frame for animation
            artists = [
                ax.text(0.5, 1.05, '%d/%d' % (it, self.nt),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes),
                ax.imshow(self.p, vmin=-0.02 * self.A, vmax=0.02 * self.A, origin='lower'),  # Display pressure field
                ax.plot(self.x_source, self.y_source, 'rs', fillstyle="none")[0],
                ax.plot(self.x_rec1, self.y_rec1, 'ko', fillstyle="none")[0],
                ax.plot(self.x_rec2, self.y_rec2, 'ko', fillstyle="none")[0],
                ax.plot(self.x_rec3, self.y_rec3, 'ko', fillstyle="none")[0]
            ]
            
            # Add mask visualization if wedge is present
            if self.wedge is not None:
                artists.append(ax.imshow(self.mask_p, cmap="gray", alpha=0.5, origin="lower"))
            
            movie.append(artists)

        print('iterations done')
        my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000, blit=True)
        plt.show()
        return None
