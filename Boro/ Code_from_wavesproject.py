import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import ndimage

class FDTD:

    def __init__(self,resolution = 0.1, domain_to_lmax = 1):
        """
        initialize instance of FDTD
        :param resolution: for dx
        :param domain_to_lmax: for nx
        """
        # constants needed
        self.c = 340  # [m/s] speed of sound
        self.t0 = 1E-3  # time delay of pulse
        self.d=1
        self.A = 1  # Source amplitude
        self.wedge = None
        # frequency band of interest
        self.f_min = self.c * 0.1 / (np.pi * self.d)
        self.f_max = self.c * 4 / (np.pi * self.d)
        self.fc = (self.f_min + self.f_max) / 2  # central frequency
        self.bandwidth = self.f_max - self.f_min
        self.l_min = np.pi * self.d / 2
        self.l_max = np.pi * self.d * 20
        self.sigma = (1 / (2 * self.bandwidth)) ** 2  # width of pulse according to literature
        # discretization
        self.dx = resolution * self.l_min          # can still reduce dx to fix diagonal propagation
        self.dy = self.dx
        self.nx = int(domain_to_lmax * self.l_max / self.dx)
        self.ny = self.nx
        self.CFL = 1
        self.dt = self.CFL / (
                    self.c * np.sqrt((1 / self.dx ** 2) + (1 / self.dy ** 2)))  # time step from spatial disc. & CFL
        self.nt = int(self.nx // self.CFL)
        # initialize the fields
        self.ox = np.zeros((self.nx + 1, self.ny), dtype=np.float64)
        self.oy = np.zeros((self.nx, self.ny + 1), dtype=np.float64)  # if memory problems arise, can try float32
        self.p = np.zeros((self.nx, self.ny), dtype=np.float64)
        # receivers and sources
        self.x_source = int(self.nx / 2)  # Source is at the center
        self.y_source = int(self.ny / 2)

        self.x_rec1 = int(self.x_source + self.d / self.dx)
        self.y_rec1 = self.y_source + int(1.5 * self.d / self.dx)
        self.x_rec2 = int(self.x_source + 2 * self.d / self.dx)
        self.y_rec2 = self.y_rec1
        self.x_rec3 = int(self.x_source + 3 * self.d / self.dx)
        self.y_rec3 = self.y_rec1



    def step_sit_sip(self):

        # need to change these to update slicing upto 1 cell before the wedge (1)
        self.ox[1:-1, :] += -self.dt / self.dx * (self.p[1:, :] - self.p[:-1, :]) * np.exp(-self.dampx[1:-1, None])
        self.oy[:, 1:-1] += -self.dt / self.dy * (self.p[:, 1:] - self.p[:, :-1]) * np.exp(-self.dampx[None, 1:-1, ])
        self.p += -self.c ** 2 * self.dt * (
                    1 / self.dx * (self.ox[1:, :] - self.ox[:-1, :]) * np.exp(-self.dampx[:-1, None]) +
                    + 1 / self.dy * (self.oy[:, 1:] - self.oy[:, :-1]) * np.exp(-self.dampy[None, 1:])
                    )
        self.ox = self.ox * np.exp(-self.dampx[:, None])
        self.oy = self.oy * np.exp(-self.dampy[None:, ])
        self.p = self.p * np.exp(- self.dampx[:-1, None])
        if self.wedge != None:
            self.p[self.mask_p == 1] = 0
        return None

    def pml(self, sigmamax=0.17,thickness_denom = 10, pml_order = 1,prof_type = 'polynomial',scale=1):
        """
        pml_order: profile of pml; higher order means less reflections from inner side
        prof_type: either polynomial or exponential
        scale: to further reduce lattice mismatch of inner side of PML
        """
        # thickness needs to be small relatively to nx
        self.pml_thickness = max(1, self.nx // thickness_denom)
        # damping coëfficient grid in x and y
        self.dampx = np.zeros(self.nx + 1)
        self.dampy = np.zeros(self.ny + 1)
        def profile(index, thickness, sigmamax,pml_order,prof_type,scale):
            return sigmamax * ((index / thickness) ** pml_order) if prof_type == 'polynomial' else (
                    sigmamax * (1 - np.exp(-pml_order * (index / thickness)**scale)))

        # only uses the values in the pml layer so that the non pml layer doesnt get affected
        for i in range(self.pml_thickness):
            sigma_value = profile(i, self.pml_thickness, sigmamax, pml_order, prof_type, scale)

            self.dampx[i] = sigma_value
            self.dampx[-(i + 1)] = sigma_value
            self.dampy[i] = sigma_value
            self.dampy[-(i + 1)] = sigma_value
            #print(self.dampx)          #debugger, inside values higher??
        return None

    def wedge_mask(self,angle=0):
        self.wedge = 1
        wedge_start_x = self.x_rec1
        wedge_start_y = self.y_source - int(1 * self.d / self.dx)
        #wedge_start_y, wedge_start_x = wedge_start_x, wedge_start_y
        self.mask_p = np.zeros((self.nx,self.ny))
        self.mask_p[wedge_start_y+1:,wedge_start_x+1:] = 1       #not including perimeter
        self.mask_p = np.flipud(self.mask_p)
        if angle != 0:
            self.mask_p = ndimage.rotate(self.mask_p,-angle,reshape=False,order=0)   #fix ever-extending
            #code_block to fix locations of source and receiver
            def rotate_xy(x,y,theta,nx,ny):
                c_x = nx / 2
                c_y = ny / 2
                shifted_x = x - c_x
                shifted_y = y - c_y
                rad_angle = np.deg2rad(theta)
                rotation_matrix = np.array([[np.cos(rad_angle), -np.sin(rad_angle)], [np.sin(rad_angle), np.cos(rad_angle)]])
                original_vector = np.array([shifted_x,shifted_y])   #position vector; to be rotated
                rotated = rotation_matrix @ original_vector
                new_x = rotated[0] + c_x
                new_y = rotated[1] + c_y
                return tuple(np.round([new_x,new_y]).astype(int))

            print(self.x_source,self.y_source)  #debug
            self.x_source, self.y_source = rotate_xy(self.x_source, self.y_source, angle,self.nx,self.ny)
            print(self.x_source, self.y_source) #debug
            self.x_rec1, self.y_rec1 = rotate_xy(self.x_rec1, self.y_rec1, angle,self.nx,self.ny)
            self.x_rec2, self.y_rec2 = rotate_xy(self.x_rec2, self.y_rec2, angle,self.nx,self.ny)
            self.x_rec3, self.y_rec3 = rotate_xy(self.x_rec3, self.y_rec3, angle,self.nx,self.ny)



        print(self.mask_p)                  #debugger

    def debugger(self):
        fig, ax1 = plt.subplots(figsize=(8, 8))
        #plt.imshow(self.mask_p, origin='lower', cmap='gray')
        # Plot source and receivers with same style as animation
        ax1.contourf(self.mask_p, levels=[0.5, 1], colors='black', linestyles='--')
        ax1.plot(self.x_source, self.y_source, 'rs', fillstyle="none", label='Source')
        ax1.plot(self.x_rec1, self.y_rec1, 'ko', fillstyle="none", label='Receiver 1')
        ax1.plot(self.x_rec2, self.y_rec2, 'ko', fillstyle="none", label='Receiver 2')
        ax1.plot(self.x_rec3, self.y_rec3, 'ko', fillstyle="none", label='Receiver 3')
        plt.title('Mask Placement with source and receivers')
        plt.show()

        
    def iterate(self):
        receiver1 = np.zeros((self.nt, 1))
        receiver2 = np.zeros((self.nt, 1))
        receiver3 = np.zeros((self.nt, 1))
        timeseries = np.zeros((self.nt, 1))

        fig, ax = plt.subplots()
        plt.axis('equal')
        plt.xlim([1, self.nx + 1])
        plt.ylim([1, self.ny + 1])
        movie = []

        for it in range(0, self.nt):
            t = (it - 1) * self.dt
            timeseries[it, 0] = t
            print('%d/%d' % (it, self.nt))  # Loading bar while sim is running

            # Update source for new time
            source = self.A * np.sin(2 * np.pi * self.fc * (t - self.t0)) * np.exp(
                -((t - self.t0) ** 2) / self.sigma)
            self.p[self.x_source, self.y_source] += source  # Adding source term
            self.step_sit_sip()  # Propagate over one time step

            # Store p field at receiver locations
            receiver1[it] = self.p[self.x_rec1, self.y_rec1]
            receiver2[it] = self.p[self.x_rec2, self.y_rec2]
            receiver3[it] = self.p[self.x_rec3, self.y_rec3]

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

problem = FDTD(0.1, 1)

problem.pml(sigmamax=0.15, thickness_denom=6, pml_order=12, prof_type='apolynomial', scale=4)
problem.wedge_mask(30)
problem.debugger()
problem.iterate()


#(1) work on surfaces; implement way to detect perimeter using ndimage.dilate to automaticaly slice 
#(2) extend mask
#(3) work out PML kinks
#(4) compare analytical
