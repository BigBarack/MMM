import numpy as np
from os import system, name
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.animation import ArtistAnimation
from matplotlib.ticker import MaxNLocator
from scipy.special import hankel2
from scipy.special import jv
from scipy import ndimage
from scipy import special
# from matplotlib import use
# use('TkAgg')

try:
    from tqdm import tqdm
    using_tqdm = True
except ImportError:
    using_tqdm = False
    # fallback: basic wrapper that does nothing, tqdm was not working in my powershell, not needed to pip install
    def tqdm(iterable, **kwargs):
        return iterable

epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of vacuum)
c = 1 / np.sqrt(mu_0 * epsilon_0)
h = 6.62607015e-34
hbar = h/(2*np.pi)
m_e = 9.10938356e-31  # Electron mass
q_e = 1.602176634e-19 # Electron charge


def clear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def cross_average2Drray(arr):
    """
    takes given array and returns an elements-wise cross average new array
    would have been useful to treat boundaries in the updates insteaad of explicitly handling
    the PEC ghost cells
    :param arr: 2D array to be padded and averaged
    :return: cross average of array
    """
    padded = np.pad(arr,1,mode='edge')
    cross_avg = (padded[1:-1,1:-1] + padded[:-2,1:-1] + padded[2:,1:-1] + padded[1:-1,:-2] + padded[1:-1,2:] ) / 5
    return cross_avg

def field_avg(field,axis):
    if axis=='vertical':
        padded = np.pad(field, ((1, 1), (0, 0)))
        avg = 0.5 * (padded[1:,:] + padded[:-1,:])
    elif axis=='horizontal':
        padded = np.pad(field, ((0, 0), (1, 1)))
        avg = 0.5 * (padded[:,1:] + padded[:,:-1])
    return avg



def laplacian_2D_4o(field,d):
    """
    for a given field array (used on Psi), compute the discretized to fourth order laplacian, using padding=0
    :param field: the field the laplacian acts on
    :param d: uniform grid step
    :return: nabla**2 of field
    """
    padded = np.pad(field,2,constant_values=0)
    lap1 =  (-padded[4:,2:-2] + 16 * padded[3:-1,2:-2] - 30 * padded[2:-2,2:-2] + 16 * padded[1:-3,2:-2] - padded[:-4,2:-2]) / (12 * d**2)
    lap2 =  (-padded[2:-2,4:] + 16 * padded[2:-2,3:-1] - 30 * padded[2:-2,2:-2] + 16 * padded[2:-2,1:-3] - padded[2:-2,:-4]) / (12 * d**2)
    return lap1 + lap2

class ObservationPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.ex_values = []
        self.ey_values = []
        self.hz_values = []
        self.coordinates = {}

    def add_sample(self,ex,ey,hz):
        self.ex_values.append(ex)
        self.ey_values.append(ey)
        self.hz_values.append(hz)


class Scatterer:
    def __init__(self , shape:str , material:str , ID:int , geometry:dict , properties:dict ):
        self.shape = shape
        self.ID = ID                    #useful for up-eq and mask
        self.material = material        #pec / pmc / drude
        self.geometry = geometry        #depends on shape
        self.properties = properties    #e,m,sigma_DC,gamma

    def get_bounds(self):
        """
        for given scatterer, it gives the x-range and y-range
        to be used for defining refinement regions and masks
        """
        if self.shape == 'circle':
            xc = self.geometry['center'][0]
            x_range = ( xc - self.geometry['radius'] , xc + self.geometry['radius'] )
            # y_range = x_range
            return x_range, x_range
        elif self.shape == 'rectangle':
            x_range = ( self.geometry['xi'], self.geometry['xf'] )
            y_range = ( self.geometry['yi'], self.geometry['yf'] )
            return x_range, y_range

    def is_inside(self,X,Y):
        """
        for given X,Y return boolean if it is inside scatterer. Used for the mask creation after having the grid
        :param X: point or array
        :param Y: point or array
        :return: same shape as inputs (better to use arrays)
        """
        if self.shape == 'rectangle':
            x_min , x_max = self.geometry['xi'], self.geometry['xf']
            y_min , y_max = self.geometry['yi'], self.geometry['yf']
            return (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        elif self.shape == 'circle':
            xc , yc  = self.geometry['center']
            r = self.geometry['radius']
            return ( (X - xc)**2 + (Y - yc)**2 ) <= r**2


class FDTD:
    def __init__(self, Lx:float , Ly:float , PW:dict , scatterer_list:list , observation_points:dict ):
        self.m_eff = m_e # temporary definition here

        # sim area
        self.Lx = Lx * 0.01
        self.Ly = Ly * 0.01

        # Plane Wave
        self.lmin = PW['lmin']
        self.dt = PW['dt']
        self.tc = PW['tc']
        self.A = PW['A']
        self.s_pulse = PW['s_pulse']
        self.direction = PW['direction']    # +x -x +y -y
        self.PW_type = PW['PW_type']
        if self.PW_type == 'sinusoidal':
            self.fc = PW['fc']

        # gridding
        self.scatterer_list = scatterer_list
        self.observation_points = observation_points
        self.there_is_Drude = any([ scat.material == 'Drude' for scat in scatterer_list])
        self.there_is_PEC = any([scat.material == 'PEC' for scat in scatterer_list])
        self.there_is_PMC = any([scat.material == 'PMC' for scat in scatterer_list])
        self.there_is_qm = any([scat.material == 'e' for scat in scatterer_list])
        self.dx_coarse = self.lmin / 10
        if self.there_is_qm:
            # decouple source from spatial discretization, override with predetermined values
            # simple first, if works ok, automate with number of cells wanted in q.dot (well)
            self.dx_coarse = 0.1e-9
            print(f'previous dt: {self.dt}')
            new_dt = (self.dx_coarse / 2 )/ (c * np.sqrt(2))
            print(f'new dx leads to maximum dt: {new_dt}')
            self.dt = new_dt
            self.tc = 3 * self.s_pulse
        self.dx_inter1 = self.dx_coarse / ( 2 ** (1/3) )
        self.dx_inter2 = self.dx_inter1 / ( 2 ** (1/3) )
        self.dx_fine = self.dx_coarse / 2




        # PML
        self.PML_n = 15
        self.PML_m = 4
        self.PML_KappaMax = 1.0
        self.PML_SigmaMax = (self.PML_m + 1) / (150 * np.pi)

        # Generate coefficients for the PML
        def sigma_e(self, i):
            return self.PML_SigmaMax * (i / self.PML_n) ** self.PML_m

        def sigma_h(self, i):
            return sigma_e(self, i) * mu_0 / epsilon_0

        def sigmamask(self, array, axis):
            mask = np.zeros_like(array)
            if axis == 0:
                for i in range(self.PML_n):
                    mask[self.PML_n - i, :], mask[-(self.PML_n - i), :] = sigma_e(self, i) * self.dx / (
                                self.dx_coarse ** 2), sigma_e(self, i) * self.dx / (self.dx_coarse ** 2)
                    # print(self.dx)
            elif axis == 1:
                for i in range(self.PML_n):
                    mask[:, self.PML_n - i], mask[:, -(self.PML_n - i)] = sigma_e(self, i) * self.dy / (
                                self.dx_coarse ** 2), sigma_e(self, i) * self.dy / (self.dx_coarse ** 2)
            elif axis == 2:
                for i in range(self.PML_n):
                    mask[self.PML_n - i, :], mask[-(self.PML_n - i), :] = sigma_h(self, i) * self.dx / (
                                self.dx_coarse ** 2), sigma_h(self, i) * self.dx / (self.dx_coarse ** 2)
                    mask[:, self.PML_n - i], mask[:, -(self.PML_n - i)] = sigma_h(self, i) * self.dy / (
                                self.dx_coarse ** 2), sigma_h(self, i) * self.dy / (self.dx_coarse ** 2)
            return mask

        # generate grid; from physical x,y to discrete
        x = [0.0]
        x0 = 0.0  # running cursor
        entering = True
        while x0 < self.Lx:
            if self.in_refined_region(x0, 'x') == 'refine':  # e.g., x0 in [x_min, x_max] of one of the scatterers
                x0 += self.dx_fine
                x.append(x0)
            elif self.in_refined_region(x0, 'x') == 'intermediate': #padding connecting coarse to refined
                if entering:
                    padding = [ x0 + self.dx_inter1,
                                x0 + 2 * self.dx_inter1,
                                x0 + 2 * self.dx_inter1 + self.dx_inter2,
                                x0 + 2 * self.dx_inter1 + 2* self.dx_inter2]
                    x.extend(padding)
                else:
                    padding = [ x0 + self.dx_inter2,
                                x0 + 2 * self.dx_inter2,
                                x0 + 2 * self.dx_inter2 + self.dx_inter1,
                                x0 + 2 * self.dx_inter2 + 2* self.dx_inter1]
                    x.extend(padding)
                x0 += 2 * self.dx_inter1 + 2 * self.dx_inter2
                entering = not entering
            else:
                x0 += self.dx_coarse
                x.append(x0)
        self.x_edges = np.array(x)
        self.dx = np.abs(np.diff(x))  # dx[i] = x[i+1] - x[i]
        self.dx_dual = 0.5 * (self.dx[:-1] + self.dx[1:])

        # refine grid
        y = [0.0]
        y0 = 0.0  # running cursor
        entering = True
        while y0 < self.Ly:
            if self.in_refined_region(y0, 'y') == 'refine':
                y0 += self.dx_fine
                y.append(y0)
            elif self.in_refined_region(y0, 'y') == 'intermediate':
                if entering:
                    padding = [ y0 + self.dx_inter1,
                                y0 + 2 * self.dx_inter1,
                                y0 + 2 * self.dx_inter1 + self.dx_inter2,
                                y0 + 2 * self.dx_inter1 + 2* self.dx_inter2]
                    y.extend(padding)
                else:
                    padding = [ y0 + self.dx_inter2,
                                y0 + 2 * self.dx_inter2,
                                y0 + 2 * self.dx_inter2 + self.dx_inter1,
                                y0 + 2 * self.dx_inter2 + 2* self.dx_inter1]
                    y.extend(padding)
                y0 += 2 * self.dx_inter1 + 2 * self.dx_inter2
                entering = not entering
            else:
                y0 += self.dx_coarse
                y.append(y0)
        self.y_edges = np.array(y)
        self.dy = np.abs(np.diff(y))  # dx[i] = x[i+1] - x[i]
        self.dy_dual = 0.5 * (self.dy[:-1] + self.dy[1:])
        x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.Xc, self.Yc = np.meshgrid(x_centers, y_centers, indexing='xy')
        _ , self.DY_Ex = np.meshgrid(x_centers ,self.dy_dual, indexing='xy')
        self.DX_Ey , _ = np.meshgrid(self.dx_dual, y_centers, indexing='xy')
        self.DX_Hz, self.DY_Hz = np.meshgrid(self.dx, self.dy, indexing='xy')

        # initialize fields
        self.Hz = np.zeros_like(self.Xc)
        self.Ny, self.Nx = self.Hz.shape
        self.Ex = np.zeros((self.Ny-1,self.Nx))
        self.Ey = np.zeros((self.Ny, self.Nx-1))
        self.epsilon_grid = np.full(self.Hz.shape,epsilon_0)
        self.epsilon_yavg = np.zeros_like(self.Ex)                 #epsilon spatially averaged over y, used for Ex update
        self.epsilon_xavg = np.zeros_like(self.Ey)                 #epsilon spatially averaged over x, used for Ey update
        self.mu_grid = np.full(self.Hz.shape,mu_0)
        if self.there_is_Drude:
            self.Jcx = np.zeros_like(self.Ex)
            self.Jcy = np.zeros_like(self.Ey)

        # initialize quantum fields
        if self.there_is_qm:
            self.psi_r = np.zeros_like(self.Hz)
            self.psi_i = np.zeros_like(self.Hz)
            self.psi_i_old = np.zeros_like(self.Hz)
            self.Jqx = np.zeros_like(self.Ex)
            self.Jqy = np.zeros_like(self.Ey)
            self.V = np.zeros_like(self.Hz)
            self.maskQM_Ex = np.zeros_like(self.Ex)
            self.maskQM_Ey = np.zeros_like(self.Ey)
            self.maskQM = np.zeros_like(self.Hz)

        # generate mask
        self.mask_Hz = np.zeros_like(self.Hz)
        self.TFSFhup = np.zeros_like(self.Hz)
        self.TFSFhdown = np.zeros_like(self.Hz)
        self.TFSFhleft = np.zeros_like(self.Hz)
        self.TFSFhright = np.zeros_like(self.Hz)

        self.TFSFeleft = np.zeros_like(self.Ey)
        self.TFSFeright = np.zeros_like(self.Ey)
        self.TFSFeup = np.zeros_like(self.Ex)
        self.TFSFedown = np.zeros_like(self.Ex)
        self.mask_Ex = np.zeros_like(self.Ex)
        self.mask_Ey = np.zeros_like(self.Ey)
        if self.there_is_PEC:
            self.maskPECx = np.zeros_like(self.Ex)
            self.maskPECy = np.zeros_like(self.Ey)
        if self.there_is_PMC:
            self.maskPMCx = np.zeros_like(self.Ex)
            self.maskPMCy = np.zeros_like(self.Ey)
            self.maskPMCz = np.zeros_like(self.Hz)

        _ , self.Y_edges = np.meshgrid(x_centers, self.y_edges[1:-1], indexing='xy') #for Ex
        self.X_edges, _ = np.meshgrid(self.x_edges[1:-1], y_centers, indexing='xy') #for Ey

        for scatterer in self.scatterer_list:
            inside_z = scatterer.is_inside(self.Xc, self.Yc)            #Hz; (Ny,Nx)
            inside_x = scatterer.is_inside(self.Xc[:-1,:], self.Y_edges)     #Ex; (Ny-1,Nx)
            inside_y = scatterer.is_inside(self.X_edges, self.Yc[:,:-1])     #Ey; (Ny,Nx-1)
            self.mask_Hz[inside_z] = scatterer.ID
            self.mask_Ex[inside_x] = scatterer.ID
            self.mask_Ey[inside_y] = scatterer.ID

        if self.there_is_PEC:
            # print('PEC found')              # debugger
            for scatterer in self.scatterer_list:   #locate all the PECs
                if scatterer.material == 'PEC':
                    self.maskPECx[self.mask_Ex == scatterer.ID] = 1
                    self.maskPECy[self.mask_Ey == scatterer.ID] = 1
            # get only their surface
            self.maskPECx -= ndimage.binary_erosion(self.maskPECx).astype(int)
            self.maskPECy -= ndimage.binary_erosion(self.maskPECy).astype(int)
        if self.there_is_PMC:
            bulkx = self.maskPMCx
            bulky = self.maskPMCy
            for scatterer in self.scatterer_list:
                if scatterer.material == 'PMC':     #locate all PMCs
                    bulkx[self.mask_Ex == scatterer.ID] = 1 #not yet bulk here
                    bulky[self.mask_Ey == scatterer.ID] = 1
                    self.maskPMCz[self.mask_Hz == scatterer.ID] = 1

            # get only surfaces
            surfx = bulkx - ndimage.binary_erosion(bulkx).astype(int)
            surfy = bulky - ndimage.binary_erosion(bulky).astype(int)
            self.maskPMCz -= ndimage.binary_erosion(self.maskPMCz).astype(int)
            # get only bulk
            bulkx = ndimage.binary_erosion(bulkx).astype(int)
            bulky = ndimage.binary_erosion(bulky).astype(int)

            self.maskPMCx[:, 1:] = (surfx[:, 1:].astype(bool) & bulkx[:,:-1].astype(bool)).astype(int)
            self.maskPMCx[:, :-1] = (self.maskPMCx[:, :-1].astype(bool) | (surfx[:,:-1].astype(bool) & bulkx[:,1:].astype(bool))).astype(int)
            self.maskPMCy[1:,:] = (surfy[1:, :].astype(bool) & bulky[:-1,:].astype(bool)).astype(int)
            self.maskPMCy[:-1, :] = (self.maskPMCy[:-1, :].astype(bool) | (
                        surfy[:-1, :].astype(bool) & bulky[1:, :].astype(bool))).astype(int)

        if self.there_is_qm:
            for scatterer in self.scatterer_list:   #locate all electron wells
                if scatterer.material == 'e':
                    self.maskQM_Ex[self.mask_Ex == scatterer.ID] = 1   #to be used for backwards coupling
                    self.maskQM_Ey[self.mask_Ey == scatterer.ID] = 1
                    self.maskQM[self.mask_Hz == scatterer.ID] = 1
                    # create potential
                    xc, yc = scatterer.geometry['center']
                    r = scatterer.geometry['radius']
                    self.m_eff = scatterer.properties['m_eff']*m_e
                    omega = scatterer.properties['omega']
                    self.get_V(xc,yc,r, self.m_eff, omega) # create each potential well
                    self.init_psi(xc,yc,scatterer.ID,self.m_eff,omega)
            self.maskQM_Ex = self.maskQM_Ex.astype(bool)
            self.maskQM_Ey = self.maskQM_Ey.astype(bool)
            # self.maskQM = self.maskQM.astype(bool)
            self.boundarymaskQM = self.maskQM
            self.maskQM = ndimage.binary_erosion(self.maskQM)   # not touching BC, therefore stays 0
            self.boundarymaskQM = np.logical_xor(self.boundarymaskQM,self.maskQM)

        for scatterer in self.scatterer_list:
            if scatterer.material == 'Drude':
                self.epsilon_grid[self.mask_Hz == scatterer.ID] = scatterer.properties['e_r'] * epsilon_0
                self.mu_grid[self.mask_Hz == scatterer.ID] = scatterer.properties['m_r'] * mu_0
                #unphysical, is storing e_0,m_0 for PEC & PMC, used for the avg, locations with PMC/PEC use BC updates
        self.epsilon_yavg = (self.epsilon_grid[:-1, :] + self.epsilon_grid[1:, :]) / 2
        self.epsilon_xavg = (self.epsilon_grid[:,:-1] + self.epsilon_grid[:,1:]) / 2
        self.mu_crossavg = cross_average2Drray(self.mu_grid)

        self.PML_ExMask = sigmamask(self, self.epsilon_yavg, 0)
        self.PML_EyMask = sigmamask(self, self.epsilon_xavg, 1)
        self.PML_HzMask = sigmamask(self, self.mu_crossavg, 2)


        # PW TFST
        self.edge_to_TFSF = 20 # decide later depedning on PML, this is in cells or index
        tfi = self.edge_to_TFSF
        self.TFSFhup[tfi, tfi:-tfi ] = 1
        self.TFSFhdown[-tfi - 1, tfi:-tfi ] = 1

        self.TFSFhleft[tfi:-tfi , tfi] = 1
        self.TFSFhright[tfi:-tfi , -tfi - 1] = 1

        self.TFSFeleft[tfi:-tfi , tfi-1] = 1
        self.TFSFeright[tfi:-tfi, -tfi] = 1          

        self.TFSFeup[tfi-1, tfi:-tfi ] = 1
        self.TFSFedown[-tfi, tfi:-tfi] = 1

        if self.direction == '+x':
            self.aux1Dgrid = self.dx[tfi-2:-tfi+2]   #include 2 cells before interface and 2 after other end interface
        elif self.direction == '-x':
            self.aux1Dgrid = self.dx[tfi-2:-tfi+2][::-1]
        elif self.direction == '+y':
            self.aux1Dgrid = self.dy[tfi-2:-tfi+2]
        else:  # self.direction == '-y':
            self.aux1Dgrid = self.dy[tfi-2:-tfi+2][::-1]
        self.auxdual = 0.5 * (self.aux1Dgrid[:-1] + self.aux1Dgrid[1:])
        self.Hz_1D = np.zeros_like(self.aux1Dgrid)    #in update we dont touch [0], source there
        self.E_inc = np.zeros_like(self.auxdual)
        self.auxbc1 = 0                                # stores field value of neighbour in previous time for 1D ADE BC

        if self.direction == '+x' or self.direction == '-y':
            self.auxbc2 = (self.dt * c - self.auxdual[-1]) / (self.dt * c + self.auxdual[-1])
        elif self.direction == '-x' or self.direction == '+y':
            self.auxbc2 = (self.dt * c - self.auxdual[0]) / (self.dt * c + self.auxdual[0])

        # observation points located in respective fields
        for obs in self.observation_points.values():
            # use the x and y edges or centers to locate physical space (x,y) points into our discrete grids
            ix_from_edges = np.searchsorted(self.x_edges,obs.x) - 1
            iy_from_edges = np.searchsorted(self.y_edges,obs.y) - 1
            ix_from_c = np.searchsorted(x_centers[1:-1],obs.x)
            iy_from_c = np.searchsorted(y_centers[1:-1], obs.y)
            obs.coordinates['hz'] = (iy_from_edges,ix_from_edges)
            obs.coordinates['ex'] = (iy_from_c,ix_from_edges)
            obs.coordinates['ey'] = (iy_from_edges,ix_from_c)

        self.update_observation_points()

    def in_refined_region(self, pos:float , axis:str):
        """
        called on FDTD because it needs dx values and scatterer list. called twice when making grid
        :return:  'refine' >  'intermediate' > 'no'
        """

        padding = 2 * (self.dx_inter1 + self.dx_inter2)
        status = 'no'
        for sc in self.scatterer_list:
            if axis == 'x':
                bounds , _ = sc.get_bounds()
            elif axis == 'y':
                _, bounds = sc.get_bounds()
            else:
                raise ValueError("Axis must be 'x' or 'y'")
            lower, upper = bounds
            if lower - padding <= pos <= upper + padding:
                if lower <= pos <= upper:
                    return 'refine'
                else:
                    status = 'intermediate'
        return status

    def get_V(self,xc,yc,r,m,w):
        # for given x,y point and radius r, create parabollic profile for 2D potential,
        # use effective mass and angular frequency of given well
        # update self.V using mask
        x_rel = self.Xc - xc
        y_rel = self.Yc - yc
        mask = (x_rel**2 + y_rel**2) <= r**2    #boolean shape like Hz,V etc, shows were the potential well is
        potential = 0.5 * m * w**2 * (x_rel**2 + y_rel**2)
        self.V[mask] = potential[mask]

    def init_psi(self,xc,yc,ID,m,w, n1=0, n2=0, x_shift=2,y_shift=0):
        """
        Initializes psi for n1=n2=0  coherent state of potential (stationary if no shift applied)
        :param xc: x of potential center
        :param yc: y of potential center
        :param ID: well ID to be used for masking
        :param m: effective mass
        :param w: omega
        :return: updates the psi_r with local wavefunction
        """

        a = np.sqrt(m * w / hbar)
        print(f'Gaussian width sigma = {1/a}')
        x_rel = self.Xc - xc
        y_rel = self.Yc - yc
        mask = ndimage.binary_erosion(self.mask_Hz == ID)   # local e-well mask, not touching the boundary
        hermite_term = special.hermite(n1)(a*x_rel) * special.hermite(n2)(a*y_rel)
        psi_local = np.exp(-0.5 * a**2 * ((x_rel - (x_shift * np.sqrt(2)/a))** 2 + (y_rel - (y_shift * np.sqrt(2)/a))** 2))    # currently full grid, contains unwanted near-zero values
        psi_local *= hermite_term
        # normalize - instead of analytical we compute discrete integral
        norm = np.sqrt(np.sum(psi_local[mask]**2 * self.dx_fine**2))
        psi_local /= norm
        self.psi_r[mask] = psi_local[mask]



    def update(self):

        # Ex & Ey updates
        # self.Ex += self.dt / self.epsilon_0 * (self.Hz[1:, :] - self.Hz[:-1, :]) / self.DY_Ex     # no spatial averaging
        # self.Ex += self.dt / self.epsilon_yavg * (self.Hz[1:, :] - self.Hz[:-1, :]) / self.DY_Ex    # no PML
        self.Ex += self.dt / self.epsilon_yavg * (self.Hz[1:, :] - self.Hz[:-1, :]) / self.DY_Ex - (
                    self.dt / epsilon_0) * np.multiply(self.PML_ExMask, self.Ex)
                    # - self.dt / self.epsilon_yavg * self.Jqx[:,:]                                 # masking j-term is not necessary as long as psi is masked (dirichlet-BC)


        # self.Ey += self.dt / self.epsilon_0 * (self.Hz[:,:-1] - self.Hz[:,1:]) / self.DX_Ey
        # self.Ey += self.dt / self.epsilon_xavg * (self.Hz[:, :-1] - self.Hz[:, 1:]) / self.DX_Ey    # no PML
        self.Ey += self.dt / self.epsilon_xavg * (self.Hz[:, :-1] - self.Hz[:, 1:]) / self.DX_Ey - (
                    self.dt / epsilon_0) * np.multiply(self.PML_EyMask, self.Ey)
                    # - self.dt / self.epsilon_yavg * self.Jqy[:,:]                                 # masking j-term is not necessary as long as psi is masked (dirichlet-BC)

        #if self.there_is_qm:
            #self.Ex[self.maskQM_Ex] -= (self.dt / epsilon_0) * self.Jqx[self.maskQM_Ex]
            #self.Ey[self.maskQM_Ey] -= (self.dt / epsilon_0) * self.Jqy[self.maskQM_Ey]

        if self.there_is_Drude: #run ADEs to update Jc x,y then add that to currently calculated E
            for scatterer in self.scatterer_list:
                if scatterer.material == 'Drude':
                    g = scatterer.properties['gamma']
                    sigma = scatterer.properties['sigma_DC']
                    maskx = self.mask_Ex == scatterer.ID
                    masky = self.mask_Ey == scatterer.ID
                    self.Jcx[maskx] = (1 - self.dt / g) * self.Jcx[maskx] + self.Ex[maskx] * self.dt * sigma / g
                    self.Jcy[masky] = (1 - self.dt / g) * self.Jcy[masky] + self.Ey[masky] * self.dt * sigma / g
                    self.Ex[maskx] -= self.dt * self.Jcx[maskx] / self.epsilon_yavg[maskx]
                    self.Ey[masky] -= self.dt * self.Jcy[masky] / self.epsilon_xavg[masky]

        if self.there_is_PEC:
            self.Ex[self.maskPECx == 1] = 0
            self.Ey[self.maskPECy == 1] = 0

        if self.there_is_PMC:
            self.Ex[self.maskPMCx==1] = 0
            self.Ey[self.maskPMCy==1] = 0

        # TFSF
        if self.direction == '+x' or self.direction == '-x':
            # print(f'down{self.Ex[self.TFSFedown == 1].shape}, up{self.Ex[self.TFSFeup == 1].shape}' )
            self.Ey[self.TFSFeleft == 1] += self.dt * self.Hz_1D[2] / (epsilon_0 * self.DX_Ey[self.TFSFeleft == 1])
            self.Ey[self.TFSFeright == 1] -= self.dt * self.Hz_1D[-3] / (epsilon_0 * self.DX_Ey[self.TFSFeright == 1])
            self.Ex[self.TFSFedown == 1] += self.dt * self.Hz_1D[2:-2] / (epsilon_0 * self.DY_Ex[self.TFSFedown == 1])
            self.Ex[self.TFSFeup == 1] -= self.dt * self.Hz_1D[2:-2] / (epsilon_0 * self.DY_Ex[self.TFSFeup == 1])
        else:
            # print(f'Ey_left {self.Ey[self.TFSFeleft==1].shape} , Hz{self.Hz_1D[2:-2].shape} , {self.aux1Dgrid.shape}')
            # print(self.Hz_1D.shape)
            self.Ex[self.TFSFedown==1] += self.dt * self.Hz_1D[1] / (epsilon_0 * self.DY_Ex[self.TFSFedown == 1])
            self.Ex[self.TFSFeup ==1] -= self.dt * self.Hz_1D[-2] / (epsilon_0 * self.DY_Ex[self.TFSFeup == 1])
            self.Ey[self.TFSFeleft==1] += self.dt * self.Hz_1D[2:-2] / (epsilon_0 * self.DX_Ey[self.TFSFeleft==1])
            self.Ey[self.TFSFeright==1] -= self.dt * self.Hz_1D[2:-2] / (epsilon_0 * self.DX_Ey[self.TFSFeright==1])



        # Hz updates
        self.Hz[1:-1,1:-1] += self.dt / self.mu_grid[1:-1,1:-1] * ( ((self.Ex[1:,1:-1] - self.Ex[:-1,1:-1]) / self.DY_Hz[1:-1,1:-1] ) +
                                           (self.Ey[1:-1,:-1] - self.Ey[1:-1,1:]) / self.DX_Hz[1:-1,1:-1] ) - (self.dt/mu_0) * np.multiply(self.PML_HzMask[1:-1,1:-1],self.Hz[1:-1,1:-1])

        if self.there_is_PMC:
            self.Hz[self.maskPMCz==1] = 0

        # BC bounding box; PEC = tangential electric field set to zero
        # top BC
        self.Hz[0,1:-1] += self.dt / self.mu_crossavg[0,1:-1] * ( ((self.Ex[0,1:-1] - 0 ) / self.DY_Hz[0,1:-1] ) +
                                           ((self.Ey[0,:-1] - self.Ey[0,1:]) / self.DX_Hz[0,1:-1]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[0,1:-1],self.Hz[0,1:-1])
        # bot BC
        self.Hz[-1,1:-1] += self.dt / self.mu_crossavg[-1,1:-1] * ( ((0 - self.Ex[-1,1:-1]) / self.DY_Hz[-1,1:-1] ) +
                                           ((self.Ey[-1,:-1] - self.Ey[-1,1:]) / self.DX_Hz[-1,1:-1]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[-1,1:-1],self.Hz[-1,1:-1])
        # left BC
        self.Hz[1:-1,0] += self.dt / self.mu_crossavg[1:-1,0] * ( ((self.Ex[1:,0] - self.Ex[:-1,0]) / self.DY_Hz[1:-1,0] ) +
                                           (( 0 - self.Ey[1:-1,0]) / self.DX_Hz[1:-1,0]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[1:-1,0],self.Hz[1:-1,0])
        # right BC
        self.Hz[1:-1,-1] += self.dt / self.mu_crossavg[1:-1,-1] * ( ((self.Ex[1:,-1] - self.Ex[:-1,-1]) / self.DY_Hz[1:-1,-1] ) +
                                           ((self.Ey[1:-1,-1] - 0 ) / self.DX_Hz[1:-1,-1]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[1:-1,-1],self.Hz[1:-1,-1])
        # corners topleft, topright, botleft, botright
        self.Hz[0,0] += self.dt / self.mu_crossavg[0,0] * ( ((self.Ex[0,0] - 0) / self.DY_Hz[0,0] ) +
                                           ( 0 - self.Ey[0,0]) / self.DX_Hz[0,0] )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[0,0],self.Hz[0,0])
        self.Hz[0,-1] += self.dt / self.mu_crossavg[0,-1] * ( ((self.Ex[0,-1] - 0) / self.DY_Hz[0,-1] ) +
                                           (( self.Ey[0,-1] - 0) / self.DX_Hz[0,-1]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[0,-1],self.Hz[0,-1])
        self.Hz[-1,0] += self.dt / self.mu_crossavg[-1,0] * ( (( 0 - self.Ex[-1,0]) / self.DY_Hz[-1,0] ) +
                                           (( 0 - self.Ey[-1,0]) / self.DX_Hz[-1,0]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[-1,0],self.Hz[-1,0])
        self.Hz[-1,-1] += self.dt / self.mu_crossavg[-1,-1] * ( (( 0 - self.Ex[-1,-1]) / self.DY_Hz[-1,-1] ) +
                                           ((self.Ey[-1,-1] - 0 ) / self.DX_Hz[-1,-1]) )- (self.dt/mu_0) * np.multiply(self.PML_HzMask[-1,-1],self.Hz[-1,-1])

        #for TFSF, we just += 'known' terms on the interface and right outside
        if self.direction == '+x' or self.direction == '-x':
            self.Hz[self.TFSFhleft == 1] += self.dt / (mu_0 * self.DX_Hz[self.TFSFhleft == 1]) * self.E_inc[1]
            self.Hz[self.TFSFhright == 1] -= self.dt / (mu_0 * self.DX_Hz[self.TFSFhright == 1]) * self.E_inc[-2]
        else:   # +y -y
            self.Hz[self.TFSFhdown == 1] += self.dt / (mu_0 * self.DY_Hz[self.TFSFhdown == 1]) * self.E_inc[1]
            self.Hz[self.TFSFhup == 1] -= self.dt / (mu_0 * self.DY_Hz[self.TFSFhup == 1]) * self.E_inc[-1]
        # for +-x, Ex_inc = 0 so top bottom dont get changed, for +-y Ey_inc = 0

        N = 1e7 # particles per meter (along Z-axis)
        effect_m = 0.15 # (relative) effective mass

        # Update equations for Psi
        if self.there_is_qm:
            ex = field_avg(self.Ex,'vertical')
            ey = field_avg(self.Ey, 'horizontal')

            self.psi_r[self.maskQM] -= self.dt/hbar *( (hbar**2/(2*m_e*effect_m)) * laplacian_2D_4o(self.psi_i,self.dx_fine)[self.maskQM]
                                                               + q_e * ex[self.maskQM] * self.Xc[self.maskQM] * self.psi_i[self.maskQM]
                                                               + q_e * ey[self.maskQM] * self.Yc[self.maskQM] * self.psi_i[self.maskQM]
                                                               - self.V[self.maskQM] * self.psi_i[self.maskQM] )

            self.psi_i[self.maskQM] += self.dt/hbar *( (hbar**2/(2*m_e*effect_m)) * laplacian_2D_4o(self.psi_r,self.dx_fine)[self.maskQM]
                                                               + q_e * ex[self.maskQM] * self.Xc[self.maskQM] * self.psi_r[self.maskQM]
                                                               + q_e * ey[self.maskQM] * self.Yc[self.maskQM] * self.psi_r[self.maskQM]
                                                               - self.V[self.maskQM] * self.psi_r[self.maskQM] )

            # Calculate Jx and Jy
            q = -q_e
            Jqx_sub = N*q*hbar/(2*m_e*effect_m) * 1/self.DX_Ey * (self.psi_r[:,:-1]*(self.psi_i[:,1:]+self.psi_i_old[:,1:])-self.psi_r[:,1:]*(self.psi_i[:,:-1]+self.psi_i_old[:,:-1]))
            self.Jqx = np.pad((Jqx_sub[1:,:]+Jqx_sub[:-1,:])/2,((0,0),(1,0)),'constant',constant_values=(0))
            Jqy_sub = N*q*hbar/(2*m_e*effect_m) * 1/self.DY_Ex * (self.psi_r[1:,:]*(self.psi_i[:-1,:]+self.psi_i_old[:-1,:])-self.psi_r[:-1,:]*(self.psi_i[1:,:]+self.psi_i_old[1:,:]))
            self.Jqy = np.pad((Jqy_sub[:,1:]+Jqy_sub[:,:-1])/2,((1,0),(0,0)),'constant',constant_values=(0))
            self.psi_i_old = self.psi_i

            # temporal average
            # psi_i_temp_avg = 0.5 * (self.psi_i + self.psi_i_old)                                #shape (Ny,Nx)
            # d/dx
            # dpsi_i_t_dx = (psi_i_temp_avg[:, 1:] - psi_i_temp_avg[:, :-1]) / self.dx_fine       #shape (Ny,Nx-1)
            # dpsi_r_dx = (self.psi_r[:,1:] - self.psi_r[:,:-1]) / self.dx_fine                   #shape (Ny,Nx-1)
            # vertical averaging of derivative to match Ex edges
            # dpsi_i_t_dx_avg = field_avg(dpsi_i_t_dx,'vertical')                             #shape (Ny-1,Nx-1)
            # dpsi_r_dx_avg = field_avg(dpsi_r_dx,'vertical')                                 #shape (Ny-1,Nx-1)
            # psi_r & psi_i averaged horizontally and then vertically
            # psi_r_avg_h = field_avg(self.psi_r,'horizontal')                                #shape (Ny,Nx-1)
            # psi_r_avg   = field_avg(psi_r_avg_h,'vertical')                                 #shape (Ny-1,Nx-1)
            # psi_i_avg_h = field_avg(psi_i_temp_avg,'horizontal')
            # psi_i_avg   = field_avg(psi_i_avg_h,'vertical')
            # Jx calc
            # self.Jqx[:,:-1] = N*q*hbar/(m_e*effect_m) * (psi_r_avg * dpsi_i_t_dx_avg - psi_i_avg * dpsi_r_dx_avg )
            # d/dy
            # dpsi_i_t_dy = (psi_i_temp_avg[1:,:] - psi_i_temp_avg[:-1,:]) / self.dx_fine       #shape (Ny-1,Nx)
            # dpsi_r_dy = (self.psi_r[1:,:] - self.psi_r[:-1,:]) / self.dx_fine                 #shape (Ny-1,Nx)
            # horizontal avg of derivative to match Ey
            # dpsi_i_t_dy_avg = field_avg(dpsi_i_t_dy,'horizontal')                        #shape (Ny-1,Nx-1)
            # dpsi_r_dy_avg = field_avg(dpsi_r_dy,'horizontal')                            #shape (Ny-1,Nx-1)
            # Jy calc
            # self.Jqy[:-1,:] = N*q*hbar/(m_e*effect_m) * (psi_r_avg * dpsi_i_t_dy_avg - psi_i_avg * dpsi_r_dy_avg )
            # self.psi_i_old = self.psi_i

        #
        self.update_observation_points()

    def source_pw(self, time):
        """
        call before each update to calculate the incident components needed for TFST
        :param time:  time to update source cell
        :return:
        """
        if self.PW_type == 'gaussian':
            source_term = self.A * np.exp(-(time - self.tc)**2 / (2 * self.s_pulse**2))
        elif self.PW_type == 'sinusoidal':
            source_term = self.A * np.exp(-(time - self.tc) ** 2 / (2 * self.s_pulse ** 2)) * np.sin(2 * np.pi * self.fc * time)
        if self.direction == '+x' or self.direction == '-y':
            self.Hz_1D[0] = source_term
            self.auxbc1 = self.Hz_1D[-2]     #storing value of neighbour before updating
            #update
            self.E_inc += self.dt * (self.Hz_1D[:-1] - self.Hz_1D[1:]) / (epsilon_0 * self.auxdual)
            self.Hz_1D[1:-1] += self.dt * (self.E_inc[:-1] - self.E_inc[1:]) / (mu_0 * self.aux1Dgrid[1:-1])
            # 1d ABC
            self.Hz_1D[-1] = self.auxbc1 + self.auxbc2 * ( self.Hz_1D[-2] - self.Hz_1D[-1] )

        elif self.direction == '-x' or self.direction == '+y':
            self.Hz_1D[-1] = source_term
            self.auxbc1 = self.Hz_1D[1]     #storing value of neighbour before updating
            # update
            self.E_inc += self.dt * (self.Hz_1D[:-1] - self.Hz_1D[1:]) / (epsilon_0 * self.auxdual)
            self.Hz_1D[1:-1] += self.dt * (self.E_inc[:-1] - self.E_inc[1:]) / (mu_0 * self.aux1Dgrid[1:-1])
            # 1d ABC
            self.Hz_1D[0] = self.auxbc1 + self.auxbc2 * (self.Hz_1D[1] - self.Hz_1D[0])

        # self.Hz_1D[0] = self.A * np.exp(-(time - self.tc) ** 2 / (2 * self.s_pulse ** 2)) * np.sin(time * 2 * np.pi * self.fc)    <-- change source type?

    def update_observation_points(self):

        for obs in self.observation_points.values():
            hz = self.Hz[slice(*obs.coordinates['hz'])]
            ex = self.Ex[slice(*obs.coordinates['ex'])]
            ey = self.Ey[slice(*obs.coordinates['ey'])]
            obs.add_sample(ex,ey,hz)

    def debugger(self,show_grid = False, field = 'Hz'):
        """
        used to print the masks on the grids
        :param show_grid: grid lines, shows non-uniformity
        :param field: which field to show
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        if field == 'Hz':
            ax.contourf(self.Xc, self.Yc, self.mask_Hz, levels=[0.5, 1], colors='black', linestyles='--')
            ax.contour(self.Xc, self.Yc, self.TFSFhup, levels=[0.5, 1], colors='black', linestyles='--')
            ax.contour(self.Xc, self.Yc, self.TFSFhdown, levels=[0.5, 1], colors='black', linestyles='--')
            ax.contour(self.Xc, self.Yc, self.TFSFhleft, levels=[0.5, 1], colors='black', linestyles='--')
            ax.contour(self.Xc, self.Yc, self.TFSFhright, levels=[0.5, 1], colors='black', linestyles='--')

        elif field == 'Ex':
            ax.contourf(self.Xc[:-1,:], self.Y_edges, self.mask_Ex, levels=[0.5, 1], colors='black', linestyles='--')
        else:
            ax.contourf(self.X_edges, self.Yc[:-1], self.mask_Ex, levels=[0.5, 1], colors='black', linestyles='--')


        for obs in self.observation_points.values():
            ax.plot(obs.x, obs.y, 'mo', fillstyle="none", label='Observation Point')
        # Plot non-uniform grid lines (physical Yee grid)
        if show_grid:
            h_lines = [[(self.x_edges[0], y), (self.x_edges[-1], y)] for y in self.y_edges]
            v_lines = [[(x, self.y_edges[0]), (x, self.y_edges[-1])] for x in self.x_edges]
            line_col = h_lines + v_lines
            line_collection = LineCollection(line_col)
            ax.add_collection(line_collection)

        ax.set_title("Non-uniform Yee Grid")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.invert_yaxis()
        plt.show()

    def iterate(self,nt, visu=True,saving = False, just1D = False):

        fig, ax = plt.subplots()
        ax.set_xlabel('(m)')
        ax.set_ylabel('(m)')

        ax.invert_yaxis()
        plt.axis('equal')

        if self.there_is_qm:
            Qfig, Qax = plt.subplots()
            Qax.set_xlabel('(m)')
            Qax.set_ylabel('(m)')
            Qax.invert_yaxis()
            plt.axis('equal')
            Qmovie = []

            posfig, posax = plt.subplots()
            Qpos = []
            posax.set_xlabel('(m)')
            posax.set_ylabel('(m)')
            plt.axis('equal')

            momfig, momax = plt.subplots()
            Qmom = []
            momax.set_xlabel('(m)')
            momax.set_ylabel('(m)')
            plt.axis('equal')

            Ekinfig, Ekinax = plt.subplots()
            QEkin = []


        movie = []
        binary = plt.cm.binary(np.linspace(0, 1, 256))
        alphas = np.linspace(0.0, 1.0, 256)
        binary[:, -1] = alphas
        binary_alpha = ListedColormap(binary)

        clear()
        print(f"Lx = {sim.Lx}, Ly = {sim.Ly}" if not sim.there_is_qm else f"Lx = {sim.Lx * 1e9:.2f} nm, Ly = {sim.Ly * 1e9:.2f} nm")
        print(f"dt = {sim.dt}, nt = {nt}, dx_fine = {sim.dx_fine}" if not sim.there_is_qm else f"dt = {sim.dt*1e9} ns, nt = {nt}, dx_fine = {sim.dx_fine*1e9:.2f} nm")
        print(f"lmin = {sim.lmin}, tc = {sim.tc}, A = {sim.A}, sigma = {sim.s_pulse}, dir {sim.direction}" if not sim.there_is_qm else f"lmin = {sim.lmin*1e9:.2f} nm, tc = {sim.tc*1e9} ns, A = {sim.A}, sigma = {sim.s_pulse}, dir {sim.direction}")
        print(f"Grid size = {sim.Nx} x {sim.Ny}")
        if self.there_is_qm:
            print(f"V max: {np.max(sim.V)}, T_osc: {2*np.pi / 50e14 }")

        for it in tqdm(range(0,nt), desc='Simulating'):
            t = (it - 1) * self.dt

            if not using_tqdm and ( it % max(1, nt // 20) == 0):
                print(f'Simulating: {int(100 * it / nt):3d}% ({it:{len(str(nt))}d}/{nt})')



            # hard point source used before PW was implemented

            # y_source = sim.Ny // 2
            # x_source = sim.Ny // 3
            # source = self.A * np.exp(-(t - self.tc)**2 / (2 * self.s_pulse**2))
            # self.Hz[y_source, x_source] += source  # Adding source term to propagation

            self.source_pw(t)   #update incident field for TFSF
            self.update()  # Propagate over one time step

            if visu:
                if  just1D:
                    artists = [
                        ax.plot(np.arange(0, len(self.Hz_1D)), self.Hz_1D, color='r', label='Hz_1D')[0],
                        ax.plot(np.arange(0, len(self.E_inc)), self.E_inc, color='b', label='E_inc')[0],
                        ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes)
                    ]
                else:
                    artists = [
                        ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes),
                        ax.pcolormesh(self.x_edges,self.y_edges,self.Hz,vmin=-1*self.A,vmax=1*self.A,cmap='seismic',)
                    ]

                for obs in sim.observation_points.values():
                    artists.append(ax.plot(obs.x, obs.y, 'ko', fillstyle="none")[0])
                if self.there_is_PMC:
                    artists.append(ax.contourf(self.x_edges[:-1],self.y_edges[:-1],self.maskPMCz,cmap=binary_alpha,vmin=0,vmax=1))
                if self.there_is_PEC:
                    artists.append(ax.contourf(self.x_edges[1:-1],self.y_edges[:-1],self.maskPECy,cmap=binary_alpha,vmin=0,vmax=1))
                if self.there_is_qm:
                    prob = np.sqrt(self.psi_r**2+self.psi_i**2)
                    artists.append(ax.contourf(self.x_edges[:-1],self.y_edges[:-1],self.boundarymaskQM,cmap=binary_alpha,vmin=0,vmax=1))
                    Qartists = [
                        Qax.text(0.5, 1.05, '%d/%d' % (it, nt),
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=ax.transAxes),
                        Qax.pcolormesh(self.x_edges,self.y_edges,prob,cmap='seismic',)
                    ]
                    Qmovie.append(Qartists)
                    Qartists.append(Qax.contourf(self.x_edges[:-1],self.y_edges[:-1],self.boundarymaskQM,cmap=binary_alpha,vmin=0,vmax=1))
                    Qpos.append((np.average(np.multiply(self.Xc,prob)),np.average(np.multiply(self.Yc,prob))))
                    Qmom.append((np.average(np.sqrt(np.add(np.square(hbar*(self.psi_r[1:]-self.psi_r[:-1])/self.DX_Hz[:-1]),np.square(hbar*(self.psi_i[1:]-self.psi_i[:-1])/self.DX_Hz[:-1])))),
                                 np.average(np.sqrt(np.add(np.square(hbar*(self.psi_r[:,1:]-self.psi_r[:,:-1])/self.DY_Hz[:,:-1]),np.square(hbar*(self.psi_i[:,1:]-self.psi_i[:,:-1])/self.DY_Hz[:,:-1]))))))
                    QEkin.append((np.square(Qmom[-1][0])+np.square(Qmom[-1][1]))/(2*self.m_eff))
                movie.append(artists)

        print('Iterations done')
        endtime = time.time()
        if visu:
            if self.there_is_qm:   
                plt.close(fig)
                plt.close(Qfig)
                plt.close(posfig)
                plt.close(momfig)
                plt.close(Ekinfig)
                while True:
                    clear()
                    choice = input("Which animation or plot to view?\n" \
                    "EM animation:      1\n" \
                    "QM animation:      2\n"
                    "Position plot:     3\n"
                    "Momentum plot:     4\n"
                    "Kinetic Energy:    5\n"
                    "\n"
                    "Exit:              0\n")
                    plt.close()
                    if choice == '1':
                        anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000, blit=True)
                        fig.show()
                        plt.show()
                        plt.close(fig)
                    elif choice == '2':
                        Qanim = ArtistAnimation(Qfig, Qmovie, interval=10, repeat_delay=1000, blit=True)
                        Qfig.show()
                        plt.show()
                        plt.close(Qfig)
                    elif choice == '3':
                        posax.plot(*zip(*Qpos))
                        posfig.show()
                        plt.show()
                        plt.close(posfig)
                    elif choice == '4':
                        momax.plot(*zip(*Qmom))
                        momfig.show()
                        plt.show()
                        plt.close(momfig)
                    elif choice == '5':
                        Ekinax.plot(QEkin)
                        Ekinfig.show()
                        plt.show()
                        plt.close(Ekinfig)
                    elif choice == '0':
                        break
            else:
                anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000, blit=True)
                fig.show() 
                plt.show()
            if saving:
                anim.save('H.gif', writer='pillow')
                if self.there_is_qm:
                    Qanim.save('Psi.gif', writer='pillow')
        return endtime



def UI_Size():
    #1. size of sim area
    Lx,Ly = 0,0
    while 1:
        try:
            Lx, Ly = map(float,input('\nPlease provide the lengths Lx [cm] and Ly [cm] in Lx,Ly format:\n').split(','))
            break
        except ValueError:
            print("Please insert 2 valid values!\n")
    return(Lx,Ly)

def UI_PW():
    # 2. PW parameters
    while 1:
        try:
            A, s_pulse = map(float,input('\nPlease provide the amplitude A of the source and the pulse width sigma in A,sigma format:\n').split(','))
            break
        except ValueError:
            print("Please insert 2 valid values!")
    while 1:
        try:
            source_choice = input("Please choose the source type:\n" \
            "Gaussian pulse: 1\n" \
            "Gaussian-modulated sinusoidal radio-frequency (RF) pulse: 2\n\n")
            assert source_choice in ['1','2']
            break
        except AssertionError:
            print("Please make a valid choice!\n")
    if source_choice == '1':
        PW_type = 'gaussian'
    elif source_choice == '2':
        PW_type = 'sinusoidal'
        while 1:
            try:
                fc = float(input('Please provide the central frequency fc [Hz]:\n'))
                break
            except ValueError:
                print('Please enter a valid frequency (float)')
    epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
    mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of vacuum)
    c = 1 / np.sqrt(mu_0 * epsilon_0)
    if PW_type == 'gaussian':
        l_min = 2 * np.pi * c * s_pulse / 3 # could also try / 5; w_max = 5 / s_pulse
    else:
        bandwidth = 0.44 / s_pulse
        f_max = fc + bandwidth / 2
        l_min = c / f_max
    dx_min = l_min / 20
    CFL = 1
    dt = CFL / ( c * np.sqrt((1 / dx_min ** 2) + (1 / dx_min ** 2)))   # time step from spatial disc. & CFL
    tc = 6 * s_pulse                                                   # suggested tc >= 5 * sigma
    while 1:
        try:
            print(f'\nFor the given pulse width, timestep = {dt} and central time tc = {tc} are recommended; If '
                  f'the user prefers manual input enter them in dt,tc format otherwise just enter.\n')
            steps = input()
            if bool(steps):
                dt,tc = map(float,steps.split(','))
            break
        except ValueError:
            print("Please either insert a valid timestep and central time or continue by inserting nothing!")
    while 1:
        try:
            direction = input('Please provide the direction of the plane wave ( +x / -x / +y / -y ):\n')
            assert direction in ['+x','-x','+y','-y']
            break
        except AssertionError:
            print("Please insert a valid direction!\n")
    while 1:
        try:
            nt = int(input("\nPlease insert how many timesteps are needed:\n"))
            break
        except ValueError:
            print("Please insert a valid amount!\n")

    PW = { 'A' : A , 's_pulse' : s_pulse , 'lmin' : l_min , 'dt' : dt, 'tc' : tc, 'direction' : direction, 'PW_type' : PW_type}
    if PW_type == 'sinusoidal':
        PW['fc'] = fc
    return(PW,nt)

def UI_scatterers():
    # 3. scatterers
    counter = 0
    scatter_list = []
    shape = 'circle'

    while shape != 'none':
        while 1:
            try:
                shape = input('\nPlease provide the shape of the scatterer (circle / rectangle / none):\n')     #define free later
                assert shape in ['circle','rectangle','none']
                break
            except AssertionError:
                print("Please insert a valid shape!\n")

        if shape == 'circle':
            while 1:
                try:
                    xc,yc,r = input('\nPlease provide the center coordinates and the radius in xc[cm],xy[cm],radius[cm] format:\n').split(',')
                    geometry = {'center': (float(xc)*0.01, float(yc)*0.01), 'radius': float(r)*0.01}
                    break
                except ValueError:
                    print("Please insert valid coordinates and radius!\n")
        elif shape == 'rectangle':
            while 1:
                try:
                    xi,xf,yi,yf = input('\nPlease provide the coordinate ranges of the scatterer in '
                                        'xmin[cm],xmax[cm],ymin[cm],ymax[cm] format:\n').split(',')
                    geometry = { 'xi':float(xi)*0.01, 'xf':float(xf)*0.01, 'yi':float(yi)*0.01, 'yf':float(yf)*0.01 }
                    break
                except ValueError:
                    print("Please insert valid coordinates!\n")

        if shape == 'none':
            break

        while 1:
            try:
                material = input('\nPlease provide the type of material ( PEC / PMC / Drude ) or e for an electron well: \n')
                assert material in ['PEC','PMC','Drude', 'e']
                break
            except AssertionError:
                print("Please insert a valid material type!\n")
        if material == 'Drude':
            while 1:
                try:
                    e_r,m_r, sigma, gamma  = input('\nPlease provide the material properties in relative permittivity,relative permeability,sigma_DC,gamma format: \n').split(',')
                    properties = { 'e_r' : float(e_r) , 'm_r' : float(m_r), 'sigma_DC' : float(sigma) , 'gamma' : float(gamma)}
                    break
                except ValueError:
                    print("Please insert valid values!\n")
        elif material == 'e':
            while 1:
                try:
                    omega, m_eff = input('\nPlease provide the angular frequency and effective mass used in the potential in omega,m format: \n').split(',')
                    properties = { 'omega' : float(omega), 'm_eff' : float(m_eff)}
                    break
                except ValueError:
                    print("Please insert valid values!\n")
        else:
            properties = {}
        counter += 1
        scatter_list.append(Scatterer(shape, material ,counter, geometry , properties))
    return scatter_list

def UI_obs():
    # 4. location(s) of observation points (x,y)
    while 1:
        try:
            observation_points_lstr = input('\nPlease provide the observation points in x1,y1;...;xn,yn format:\n').split(';')
            obs_dict_tuples = {}
            for xy in observation_points_lstr:
                a,b = xy.split(',')
                obs_dict_tuples[(float(a)*0.01, float(b)*0.01)] = ObservationPoint(float(a)*0.01,float(b)*0.01)
            break
        except ValueError:
            print("Please input valid values!\n")
    return obs_dict_tuples

def user_inputs():
    clear()
    print("==========Custom==========")

    Lx,Ly = UI_Size()

    PW, nt = UI_PW()

    scatter_list = UI_scatterers()

    obs_dict_tuples = UI_obs()

    while 1:
        while 1:
            try:
                clear()
                choice = input("\n\nChoose any aspect of the simulation you want to go back and change:\n"
                "Domain size: 1\n"
                "Plane wave: 2\n"
                "Scatterers: 3\n"
                "Observation points: 4\n"
                "\n"
                "Continue: 0\n\n")
                assert choice in ['0','1','2','3','4']
                break
            except AssertionError:
                print("Please make a valid choice\n")
        if choice == '1':
            Lx,Ly = UI_Size()
        elif choice == '2':
            PW, nt = UI_PW()
        elif choice == '3':
            scatter_list = UI_scatterers()
        elif choice == '4':
            obs_dict_tuples = UI_obs()
        elif choice == '0':
            break

    return Lx, Ly, PW, scatter_list, obs_dict_tuples, nt

def validate_inputs(Lx: float, Ly:float, PW:dict, PML_n = 15, edge_to_TFSF = 20, obs_dict= None)-> bool:
    """
    Validates that:
    - domain is large enough to fit PML and TFSF region for given gridding (dependent on wavelength)
    - all observation points lie within the domain

    :return: True if valid, False if not valid
    """
    valid = True
    dx_coarse = PW['lmin'] / 10
    margins = 2 * dx_coarse * max(PML_n , edge_to_TFSF) # in meters
    if Lx*0.01 < margins or Ly*0.01 < margins:
        print(f"WARNING! Domain is too small ({Lx} cm x {Ly} cm).")
        print(f"At least {margins*100} cm are needed for the UPML & TFSF implementation.")
        valid = False
    if obs_dict:
        for x,y in obs_dict:
            if not(0 <= x*100 <= Lx and 0 <= y*100 <= Ly):
                print(f"WARNING! Observation point ({x*100},{y*100}) is outside the domain.")
                valid = False
    return valid


def Run():
    choice = 0
    while 1:
        try:
            clear()
            choice = input("===========Welcome============\n" \
            "Please make a choice:\n" \
            "Custom input: 1\n" \
            "Examples: 2\n\n")
            assert choice in ['1','2']
            break
        except AssertionError:
            print("Please make a valid choice!\n")
    if choice == '1':
        Lx, Ly, PW, scatterers, obs, nt = user_inputs()
        if not validate_inputs(Lx=Lx, Ly=Ly, PW=PW, obs_dict=obs):
            input("Press Enter to return to the start...")
            return Run()
        return Lx, Ly, PW, scatterers, obs, nt
    elif choice == '2':
        while 1:
            try:
                clear()
                choice = input("==========Examples==========\n" \
                "Please make a choice:\n" \
                "PEC circle: 1\n" \
                "PMC circle: 2\n" \
                "Drude circle: 3\n" \
                "e-Well circle: 4\n" \
                "\n" \
                "Return: 0\n")
                assert choice in ['0','1','2','3','4']
                break
            except AssertionError:
                print("Please make a valid choice!\n")
        if choice == '1':
            return testing(20.0,20.0,1,0.000000000014,'circle',10,10,3,'PEC')
        elif choice == '2':
            return testing(20.0,20.0,1,0.000000000014,'circle',10,10,3,'PMC')
        elif choice == '3':
            return testing(20.0,20.0,1,0.000000000014,'circle',10,10,3,'Drude',
                    10,10,10000000,10000000000000)
        elif choice == '4':                                                                                                                                                        #omega was 50e14
            return testing(15e-7 ,15e-7,1,(5 * 5e-9 * 3) / (2 * 3e8 * np.pi),'circle',7.5e-7,7.5e-7,2.5e-7,'e', rel_m_eff=0.15, omega= 50e13, timesteps=1000)
        elif choice == '0':
            return Run()


def testing(Lx:float, Ly:float,A, s_pulse,shape,xc,yc,r,material,e_r=10,m_r=10, sigma=10000000, gamma=10000000000000, observation_points_lstr=['0.0,0.0','0.0,0.0'], rel_m_eff=0.0, omega=0.0, timesteps=700):
    # 1. size of sim area
    # Lx, Ly = map(float,input('Please provide the lengths Lx [cm] and Ly [cm] in Lx,Ly format: ').split(','))
    # 2. PW parameters
    # A, s_pulse = map(float,input('Please provide the amplitude A of the source and the pulse width sigma in A,sigma format').split(','))
    l_min = 2 * np.pi * c * s_pulse / 3 # could also try / 5; w_max = 5 / s_pulse
    dx_min = l_min / 20
    CFL = 1
    dt = CFL / ( c * np.sqrt((1 / dx_min ** 2) + (1 / dx_min ** 2)))   # time step from spatial disc. & CFL
    tc = 6 * s_pulse                                                   # suggested tc >= 5 * sigma
    # steps = input(f'For the given pulse width, timestep = {dt} and central time tc = {tc} are recommended; If '
    #       f'the user prefers manual input enter them in dt,tc format otherwise just enter.')
    # if bool(steps):
    #     dt,tc = map(float,steps.split(','))
    direction = '+x'
    PW_type = 'gaussian'
    PW = {'PW_type' : PW_type, 'A' : A , 's_pulse' : s_pulse , 'lmin' : l_min , 'dt' : dt, 'tc' : tc, 'direction' : direction}
    # 3. scatterers
    # shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none): ')     #defien free later
    counter = 0
    scatter_list = []
    while shape != 'none':
        if shape == 'circle':
            # xc,yc,r = input('Please provide the center coordinates and the radius in xc,xy,radius format: ').split(',')
            geometry = {'center': (float(xc)*0.01, float(yc)*0.01), 'radius': float(r)*0.01}
        elif shape == 'rectangle':
            xi,xf,yi,yf = input('Please provide the coordinate ranges of the scatterer in '
                                'xmin,xmax,ymin,ymax format: ').split(',')
            geometry = { 'xi':float(xi)*0.01, 'xf':float(xf)*0.01, 'yi':float(yi)*0.01, 'yf':float(yf)*0.01 }
        else:
            # error handling
            shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none); typo: ')
            continue
        # material = input('Please provide the type of material; PEC / PMC / Drude: ')
        if material == 'Drude':
            # e_r,m_r, sigma, gamma  = input('Please provide the material properties in relative permittivity,relative permeability,sigma_DC,gamma format: ').split(',')
            properties = { 'e_r' : float(e_r) , 'm_r' : float(m_r), 'sigma_DC' : float(sigma) , 'gamma' : float(gamma)}
        elif material == 'e':
            properties = { 'm_eff' : float(rel_m_eff) , 'omega' : float(omega)}
        else:
            properties = {}
        counter += 1
        scatter_list.append(Scatterer(shape, material ,counter, geometry , properties))
        # shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none): ')
        shape = 'none'
    # 4. location(s) of observation points (x,y)
    # observation_points_lstr = input('Please provide the observation points in x1,y1;...;xn,yn format: ').split(';')
    obs_dict_tuples = {}
    for xy in observation_points_lstr:
        a,b = xy.split(',')
        obs_dict_tuples[(float(a)*0.01, float(b)*0.01)] = ObservationPoint(float(a)*0.01,float(b)*0.01)

    return Lx, Ly, PW, scatter_list, obs_dict_tuples, timesteps

# to test the Drude implementation, copied numbers from graphene example of syllabus
#  lamda ~ 5 micrometer -> ~ Thz freq
# sigma_for_Thz = 5 / ( np.pi * 2) * 10**(-14)        # [a cm]
# lamda_for_Thz = 2 * np.pi * 10**8 * sigma_for_Thz # [cm]
# L_for_Thz = 20 * lamda_for_Thz
# g_for_Thz = 10**(-12)
# print(L_for_Thz)
# sim = FDTD(*testing(L_for_Thz,L_for_Thz,1,sigma_for_Thz,'circle',L_for_Thz/2,L_for_Thz/2,L_for_Thz/10,'Drude',
#                     1,1,670000,g_for_Thz,['00,0','0,0']))

# sim.debugger(show_grid=False, field='Hz')
# nt = (sim.Lx / 1) / (sim.dt * sim.c)

# for obs in sim.observation_points.values():       #this is how we can access observation points after sim runs
#     print(obs.ex_values)

def frequency_analysis(sim_object):
    fig, axis = plt.subplots(2, 1, figsize=(10, 8))

    for obs in sim_object.observation_points.values():
        # Plot the original Hz values
        axis[0].plot(range(len(obs.hz_values)), obs.hz_values, label=f"Obs ({obs.x}, {obs.y})")

        # Compute the FFT of Hz values
        hz_fft = np.fft.fft(obs.hz_values)
        hz_freq = np.fft.fftfreq(len(obs.hz_values), d=sim.dt)  # Frequency axis

        # Plot the magnitude of the FFT
        axis[1].plot(hz_freq[:len(hz_freq) // 2], np.abs(hz_fft[:len(hz_fft) // 2]), label=f"Obs ({obs.x}, {obs.y})")

        # Add labels and legend for the original Hz values
    axis[0].set_title("Hz Values at Observation Points")
    axis[0].set_xlabel("Time Step")
    axis[0].set_ylabel("Hz Value")
    axis[0].legend()

    # Add labels and legend for the FFT
    axis[1].set_title("FFT of Hz Values at Observation Points")
    axis[1].xaxis.set_major_locator(MaxNLocator(nbins=20))  # Example: 20 ticks
    axis[1].set_xlabel("Frequency (Hz)")
    axis[1].set_ylabel("Magnitude")
    axis[1].legend()

    plt.tight_layout()
    plt.show()

def analytical_solution(sim_ob):
    """"
    this is the total field solution for a plane wave incident on a circular scatterer in free space.
    """

    for obs in sim_ob.observation_points.values():
        #parameters

        rho = np.sqrt(obs.x**2 + obs.y**2)   # Distance from the origin to the observation point
        phi = np.arctan2(obs.y, obs.x)  # Angle in polar coordinates

        # Wavevector (k) based on the wavelength

        k_values = np.linspace(0.1, 2*sim_ob.k, 500)  # Avoid k=0 to prevent division by zero
        a = None
        for scatterer in sim_ob.scatterer_list:
            if scatterer.shape == 'circle':  # Check if the scatterer is a circle
                a = scatterer.geometry['radius']  # Extract the radius
                break

        if a is None:
            raise ValueError("No circular scatterer found in the simulation.")


        A= 1
        # Parameters


        #k_values = np.linspace(0.1, 10, 500)  # Avoid k=0 to prevent division by zero
        Hz_scat_values = []

        # Function to compute nu(n)
        def nu(n):
            if n == 0:
                return 1
            return 2



        for k in k_values:




            ka = k * a
            N_max = int(np.ceil(ka + 10))  # Number of summations
            Hz_scat = 0.0

            for n in range(0, 2 * N_max + 1):
                an = (-1j) ** nu(n)
                term = A * (
                        jv(n, k * rho) - jv(n, ka) / hankel2(n, ka) * hankel2(n, k * rho)
                ) * np.cos(n * phi)
                termcomp = an * term
                Hz_scat += termcomp

            Hz_scat_values.append(np.abs(Hz_scat))  # Store the magnitude of Hz_scat

        # Plot the results
        figH=plt.figure(figsize=(10, 6))
        axH = figH.add_subplot(111)  # Create an Axes object in the figure
        axH.plot(k_values*sim_ob.c, Hz_scat_values, label="|Hz_scat|")  # Plot on the Axes
        axH.set_label(f"Observation Point ({obs.x}, {obs.y})")
        axH.set_title("Frequency Response of Hz_scat")
        axH.set_xlabel("Wavevector k (proportional to frequency)")
        axH.set_ylabel("|Hz_scat|")
        axH.grid(True)
        axH.legend()
        plt.show()

def compare_numerical_analytical(sim_object):
    for obs in sim_object.observation_points.values():
        # --- Numerical FFT ---
        hz_fft = np.fft.fft(obs.hz_values)
        freq_numerical = np.fft.fftfreq(len(obs.hz_values), d=sim_object.dt)
        hz_fft_magnitude = np.abs(hz_fft[:len(hz_fft) // 2])
        freq_numerical = freq_numerical[:len(freq_numerical) // 2]

        # --- Analytical Hz_scat ---
        rho = np.sqrt(obs.x**2 + obs.y**2)
        phi = np.arctan2(obs.y, obs.x)

        k_values = np.linspace(0.1, 2 * sim_object.k, 500)
        a = None
        for scatterer in sim_object.scatterer_list:
            if scatterer.shape == 'circle':
                a = scatterer.geometry['radius']
                break
        if a is None:
            raise ValueError("No circular scatterer found.")

        A = 1
        def nu(n): return 1 if n == 0 else 2

        Hz_scat_values = []
        for k in k_values:
            ka = k * a
            N_max = int(np.ceil(ka + 10))
            Hz_scat = 0.0
            for n in range(0, 2 * N_max + 1):
                an = (-1j) ** nu(n)
                term = A * (
                    jv(n, k * rho) - jv(n, ka) / hankel2(n, ka) * hankel2(n, k * rho)
                ) * np.cos(n * phi)
                Hz_scat += an * term
            Hz_scat_values.append(np.abs(Hz_scat))

        freq_analytical = k_values * sim_object.c  # Convert k to frequency

        # --- Interpolation for alignment ---
        interp_analytical = np.interp(freq_numerical, freq_analytical, Hz_scat_values)

        difference = hz_fft_magnitude - interp_analytical
        diff_max = np.max(np.abs(difference))


        # --- Plot comparison ---
        fig=plt.figure(figsize=(10, 6))
        axc= fig.add_subplot(111)
        axc.plot(freq_numerical, np.abs(difference), label="difference", alpha=0.7)
        axc.set_xlim(0, np.max(freq_numerical))
        axc.set_ylim(0, 2*diff_max) # Adjust y-axis limit for better visibility
        plt.title(f"Frequency Response Comparison at Observation Point ({obs.x}, {obs.y})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#For testing purposes
#sim = FDTD(*testing(20.0,20.0,1,0.000000000014,'circle',10,10,3,'Drude',
#                    10,10,10000000,10000000000000,['6,10','14,10']))
Lx, Ly, PW, scatter_list, obs_dict_tuples, nt = Run()
sim = FDTD(Lx, Ly, PW, scatter_list, obs_dict_tuples)

import time
start = time.time()
end = sim.iterate(int(nt), visu = True, just1D=False, saving=False)
clear()
print(f"Runtime core simulation: {end - start:.2f} seconds")
def plot_potential(V, Xc, Yc, title="Potential V(x, y)"):
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    pcm = plt.pcolormesh(Xc * 1e9, Yc * 1e9, V, shading='auto')
    plt.colorbar(pcm, label="V [J]")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# plot_potential(sim.V, sim.Xc, sim.Yc)

#sim1 = FDTD(*testing(20.0,20.0,1,0.000000000014,'circle',10,10,3,'Drude',
                    #10,10,10000000,10000000000000,['6,10','14,10']))2          These worked with an older version of the code , amount of timesteps was 1400 and the wave moved in the positive x direction
#
#frequency_analysis(sim1)
#analytical_solution(sim_ob=sim1)
#frequency_analysis(sim1)
#analytical_solution(sim_ob=sim1)
