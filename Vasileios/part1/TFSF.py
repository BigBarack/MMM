#start building as I read ( start from what I know; simple update eq)
    #figure out source part later
#how do I do non-uniform?                                                    [resolved]
#array shapes?                                                               [resolved]
#implement BC
#do I make it as a class again? = yes                                        [resolved]
#dont use matrix implementation as instructed by slides.                     [resolved]
#use flattened ?   = no                                                      [resolved]
#visu hz ?   = yes                                                           [resolved]
#scatterers attribute, option for multiple when entering them                [resolved]
#if given gridding is more than 1.5 rule, warning
          #= no need, gridding done automatically                            [resolved]

#what to request as input                                                    [resolved]

#for updates, use np.where(condition, update, update2 )
# instead of nested np.where, better to use np.select
"""
1. size of sim area
2. non-uniform gridding of sim domain & time-step
3. PW parameters [see ch.5]
4. scatterer geometrical & material prop of Drude media, PMC and PEC scatterers
5. location(s) of observation points (x,y)

after input of grid, time-step should be suggested by our code
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import ArtistAnimation




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

        # constants
        self.epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
        self.mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of vacuum)
        self.c = 1 / np.sqrt(self.mu_0 * self.epsilon_0)  # Speed of light in vacuum
        # sim area
        self.Lx = Lx
        self.Ly = Ly
        # Plane Wave
        self.lmin = PW['lmin']
        self.dt = PW['dt']
        self.tc = PW['tc']
        self.A = PW['A']
        self.s_pulse = PW['s_pulse']
        self.direction = PW['direction']    # +x -x +y -y
        # gridding
        self.dx_coarse = self.lmin / 10
        self.dx_inter1 = self.dx_coarse / ( 2 ** (1/3) )
        self.dx_inter2 = self.dx_inter1 / ( 2 ** (1/3) )
        self.dx_fine = self.lmin / 20
        self.scatterer_list = scatterer_list
        self.observation_points = observation_points
        self.there_is_Drude = any([ scat.material == 'Drude' for scat in scatterer_list])
        # print(f'Lx,Ly is {self.Lx},{self.Ly}, l_min is {self.lmin} and dx_coarse is {self.dx_coarse}, matching '
        #       f' dx = lmin/10 {np.abs(self.dx_coarse- self.lmin/10 )< 0.001}')            # debugger
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
        # print(x)                                                        #debugger
        self.dx = np.abs(np.diff(x))  # dx[i] = x[i+1] - x[i]
        self.dx_dual = 0.5 * (self.dx[:-1] + self.dx[1:])
        # print(len(self.x_edges))                                        #debugger
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
        # print(y)                                                        #debugger
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
        print(f"Ny:{self.Ny}, Nx:{self.Nx}")                                      # debugger
        self.Ex = np.zeros((self.Ny-1,self.Nx))
        self.Ey = np.zeros((self.Ny, self.Nx-1))
        self.epsilon_grid = np.full(self.Hz.shape,self.epsilon_0)
        self.epsilon_yavg = np.zeros_like(self.Ex)                 #epsilon spatially averaged over y, used for Ex update
        self.epsilon_xavg = np.zeros_like(self.Ey)                 #epsilon spatially averaged over x, used for Ey update
        self.mu_grid = np.full(self.Hz.shape, self.mu_0)
        self.Jcx = np.zeros_like(self.Ex)
        self.Jcy = np.zeros_like(self.Ey)
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
        _ , Y_edges = np.meshgrid(x_centers, self.y_edges[1:-1], indexing='xy') #for Ex
        X_edges, _ = np.meshgrid(self.x_edges[1:-1], y_centers, indexing='xy') #for Ey
        # print(f"Xc: {self.Xc.shape}, Y_edges{Y_edges.shape}, Yc{self.Yc.shape}, X_edges{X_edges.shape}")    #debugger
        for scatterer in self.scatterer_list:
            inside_z = scatterer.is_inside(self.Xc, self.Yc)            #Hz; (Ny,Nx)
            inside_x = scatterer.is_inside(self.Xc[:-1,:], Y_edges)     #Ex; (Ny-1,Nx)
            inside_y = scatterer.is_inside(X_edges, self.Yc[:,:-1])     #Ey; (Ny,Nx-1)
            self.mask_Hz[inside_z] = scatterer.ID
            self.mask_Ex[inside_x] = scatterer.ID
            self.mask_Ey[inside_y] = scatterer.ID
        for scatterer in self.scatterer_list:
            if scatterer.material == 'Drude':
                self.epsilon_grid[self.mask_Hz == scatterer.ID] = scatterer.properties['e_r'] * self.epsilon_0
                self.mu_grid[self.mask_Hz == scatterer.ID] = scatterer.properties['m_r'] * self.mu_0
                #unphysical, is storing e_0,m_0 for PEC & PMC, used for the avg, locations with PMC/PEC use BC updates
        self.epsilon_yavg = (self.epsilon_grid[:-1, :] + self.epsilon_grid[1:, :]) / 2
        self.epsilon_xavg = (self.epsilon_grid[:,:-1] + self.epsilon_grid[:,1:]) / 2
        self.mu_crossavg = cross_average2Drray(self.mu_grid)
        # PW TFST
        self.edge_to_TFSF = 10 # decide later depedning on PML, this is in cells or index
        tfi = self.edge_to_TFSF
        tfie = tfi -1
        self.TFSFhup[tfi, tfi:-tfi ] = 1
        self.TFSFhdown[-tfi - 1, tfi:-tfi ] = 1

        self.TFSFhleft[tfi:-tfi , tfi] = 1
        self.TFSFhright[tfi:-tfi , -tfi - 1] = 1

        self.TFSFeleft[tfi:-tfie - 1, tfie] = 1
        self.TFSFeright[tfi:-tfie - 1, -tfie - 1] = 1

        self.TFSFeup[tfie, tfie:-tfie ] = 1
        self.TFSFedown[-tfie - 1, tfie:-tfie] = 1

        if self.direction == '+x':
            self.aux1Dgrid = self.dx[tfi-2:-tfi+2]   #include 2 cells before interface and 2 after other end interface
        elif self.direction == '-x':
            self.aux1Dgrid = self.dx[tfi-2:-tfi+2][::-1]
        elif self.direction == '+y':
            self.aux1Dgrid = self.dy[tfi-2:-tfi+2]
        else:  # self.direction == '+y':
            self.aux1Dgrid = self.dy[tfi-2:-tfi+2][::-1]
        self.auxdual = 0.5 * (self.aux1Dgrid[:-1] + self.aux1Dgrid[1:])
        self.Hz_1D = np.zeros_like(self.aux1Dgrid)    #in update we dont touch [0], source there
        self.E_inc = np.zeros_like(self.auxdual)

        # observation points located in respecitive fields
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

    def update(self):

        # print(f'DY shape{self.DY_Ex.shape} , Ex {self.Ex.shape} ,self.Hz[1:,:] {self.Hz[1:,:].shape} ')
        #
        # print(f'self.Ey {self.Ey.shape}, Hz[:,:-1] {self.Hz[:, :-1].shape}, self.DX {self.DX_Ey.shape}')
        #
        # print(f'self.Hz {self.Hz.shape}, (self.Ex[1:,:] - self.Ex[:-1,:]) {(self.Ex[1:,:] - self.Ex[:-1,:]).shape}'
        #       f',(self.Ey[:,:-1] - self.Ey[:,1:]){(self.Ey[:,:-1] - self.Ey[:,1:]).shape},'
        #       f'self.DY_Hz {self.DY_Hz.shape}, self.DX_Hz {self.DX_Hz .shape}')

        # self.Ex += self.dt / self.epsilon_0 * (self.Hz[1:, :] - self.Hz[:-1, :]) / self.DY_Ex
        self.Ex += self.dt / self.epsilon_yavg * (self.Hz[1:, :] - self.Hz[:-1, :]) / self.DY_Ex
        # self.Ey += self.dt / self.epsilon_0 * (self.Hz[:,:-1] - self.Hz[:,1:]) / self.DX_Ey
        self.Ey += self.dt / self.epsilon_xavg * (self.Hz[:, :-1] - self.Hz[:, 1:]) / self.DX_Ey

        if self.there_is_Drude: #run ADEs to update Jc x,y then add that to currently calculated E
            # print('entered in there_is in update')          # debugger
            for scatterer in self.scatterer_list:
                if scatterer.material == 'Drude':
                    # print('found the Drude')                   # debugger
                    g = scatterer.properties['gamma']
                    sigma = scatterer.properties['sigma_DC']
                    maskx = self.mask_Ex == scatterer.ID
                    masky = self.mask_Ey == scatterer.ID
                    self.Jcx[maskx] = (1 - self.dt / g) * self.Jcx[maskx] + self.Ex[maskx] * self.dt * sigma / g
                    self.Jcy[masky] = (1 - self.dt / g) * self.Jcy[masky] + self.Ey[masky] * self.dt * sigma / g
                    self.Ex[maskx] -= self.dt * self.Jcx[maskx] / self.epsilon_yavg[maskx]
                    self.Ey[masky] -= self.dt * self.Jcy[masky] / self.epsilon_xavg[masky]
        # TFSF
        self.Ey[self.TFSFeleft == 1] += self.dt * self.Hz_1D[1] / (self.epsilon_0 * self.DX_Ey[self.TFSFeleft == 1])
        self.Ey[self.TFSFeright == 1] -= self.dt * self.Hz_1D[-2] / (self.epsilon_0 * self.DX_Ey[self.TFSFeright == 1])
        self.Ex[self.TFSFedown == 1] += self.dt * self.Hz_1D[1:-1] / (self.epsilon_0 * self.DY_Ex[self.TFSFeup == 1])
        self.Ex[self.TFSFeup == 1] -= self.dt * self.Hz_1D[1:-1] / (self.epsilon_0 * self.DY_Ex[self.TFSFeup == 1])


        self.Hz[1:-1,1:-1] += self.dt / self.mu_grid[1:-1,1:-1] * ( ((self.Ex[1:,1:-1] - self.Ex[:-1,1:-1]) / self.DY_Hz[1:-1,1:-1] ) +
                                           (self.Ey[1:-1,:-1] - self.Ey[1:-1,1:]) / self.DX_Hz[1:-1,1:-1] )
        # BC PEC = tangential electric field set to zero
        # top BC
        self.Hz[0,1:-1] += self.dt / self.mu_crossavg[0,1:-1] * ( ((self.Ex[0,1:-1] - 0 ) / self.DY_Hz[0,1:-1] ) +
                                           ((self.Ey[0,:-1] - self.Ey[0,1:]) / self.DX_Hz[0,1:-1]) )
        # bot BC
        self.Hz[-1,1:-1] += self.dt / self.mu_crossavg[-1,1:-1] * ( ((0 - self.Ex[-1,1:-1]) / self.DY_Hz[-1,1:-1] ) +
                                           ((self.Ey[-1,:-1] - self.Ey[-1,1:]) / self.DX_Hz[-1,1:-1]) )
        # left BC
        self.Hz[1:-1,0] += self.dt / self.mu_crossavg[1:-1,0] * ( ((self.Ex[1:,0] - self.Ex[:-1,0]) / self.DY_Hz[1:-1,0] ) +
                                           (( 0 - self.Ey[1:-1,0]) / self.DX_Hz[1:-1,0]) )
        # right BC
        self.Hz[1:-1,-1] += self.dt / self.mu_crossavg[1:-1,-1] * ( ((self.Ex[1:,-1] - self.Ex[:-1,-1]) / self.DY_Hz[1:-1,-1] ) +
                                           ((self.Ey[1:-1,-1] - 0 ) / self.DX_Hz[1:-1,-1]) )
        # corners tl, tr, bl, br
        self.Hz[0,0] += self.dt / self.mu_crossavg[0,0] * ( ((self.Ex[0,0] - 0) / self.DY_Hz[0,0] ) +
                                           ( 0 - self.Ey[0,0]) / self.DX_Hz[0,0] )
        self.Hz[0,-1] += self.dt / self.mu_crossavg[0,-1] * ( ((self.Ex[0,-1] - 0) / self.DY_Hz[0,-1] ) +
                                           (( self.Ey[0,-1] - 0) / self.DX_Hz[0,-1]) )
        self.Hz[-1,0] += self.dt / self.mu_crossavg[-1,0] * ( (( 0 - self.Ex[-1,0]) / self.DY_Hz[-1,0] ) +
                                           (( 0 - self.Ey[-1,0]) / self.DX_Hz[-1,0]) )
        self.Hz[-1,-1] += self.dt / self.mu_crossavg[-1,-1] * ( (( 0 - self.Ex[-1,-1]) / self.DY_Hz[-1,-1] ) +
                                           ((self.Ey[-1,-1] - 0 ) / self.DX_Hz[-1,-1]) )
        #for TFSF, we just += 'known' terms on the interface and right outside
        self.Hz[self.TFSFhleft == 1] += self.dt / (self.mu_0 * self.DX_Hz[self.TFSFhleft == 1]) * self.E_inc[1]           #maybe needs -1 *
        self.Hz[self.TFSFhright == 1] -= self.dt / (self.mu_0 * self.DX_Hz[self.TFSFhright == 1]) * self.E_inc[-2]
        # for +x, Ex_inc = 0 so top bottom dont get changed

        self.update_observation_points()

    # BC for PMC and PEC logic

    def source_pw(self, time):
        """
        can generalize with a logic to digure out diagonal gridding later
        call before each update to calculate the incident components needed for TFST
        :param time:  time to update source cell
        :return:
        """
        # print(f'Hz1D {self.Hz_1D.shape},E_inc {self.E_inc.shape} , aux1Dgrid {self.aux1Dgrid.shape} ')
        self.Hz_1D[0] = self.A * np.exp(-(time - self.tc)**2 / (2 * self.s_pulse**2))
        self.E_inc += self.dt * (self.Hz_1D[:-1] - self.Hz_1D[1:]) / (self.epsilon_0 * self.auxdual)
        self.Hz_1D[1:-1] += self.dt * (self.E_inc[:-1] - self.E_inc[1:]) / (self.mu_0 * self.aux1Dgrid[1:-1])
        # add Mur's ABC later?

    def update_observation_points(self):

        for obs in self.observation_points.values():
            hz = self.Hz[*obs.coordinates['hz']]
            ex = self.Ex[*obs.coordinates['ex']]
            ey = self.Ey[*obs.coordinates['ey']]
            obs.add_sample(ex,ey,hz)

    def debugger(self,show_grid = False):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.contourf(self.Xc, self.Yc, self.mask_Hz, levels=[0.5, 1], colors='black', linestyles='--')
        # ax.contourf(self.Xc, self.Yc, self.TFSF, levels=[0.5, 1], colors='black', linestyles='--')
        for obs in self.observation_points.values():
            ax.plot(obs.x, obs.y, 'mo', fillstyle="none", label='Observation Point')

        # Plot non-uniform grid lines (physical Yee grid)
        if show_grid:
            # line_col = []
            # for yv in self.y_edges:
            #     xv = [ (x,yv) for x in self.x_edges]
            #     line_col.append(xv)
            h_lines = [[(self.x_edges[0], y), (self.x_edges[-1], y)] for y in self.y_edges]
            v_lines = [[(x, self.y_edges[0]), (x, self.y_edges[-1])] for x in self.x_edges]
            line_col = h_lines + v_lines
            line_collection = LineCollection(line_col)
            ax.add_collection(line_collection)


        # for obs in self.obs_points:  # If stored in self.obs_points
        #     ax.plot(obs[0], obs[1], 'bo', fillstyle="none", label='Receiver')

        # print("x range:", self.x_edges[0], self.x_edges[-1])      #debugger
        # print("y range:", self.y_edges[0], self.y_edges[-1])      #debugger

        ax.set_title("Non-uniform Yee Grid")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.invert_yaxis()
        # plt.grid(visible=True,)
        plt.show()

    def iterate(self,nt, visu=True,saving = False):

        # timeseries = np.zeros((nt,))   # DELETE??

        fig, ax = plt.subplots()
        ax.invert_yaxis()
        plt.axis('equal')

        # plt.xlim([1, self.nx + 1])            #to avoid panning check if Nx or x
        # plt.ylim([1, self.ny + 1])
        movie = []

        for it in range(0,nt):
            t = (it - 1) * self.dt              #try without the -1
            # timeseries[it,] = t        # DELETE?
            print('%d/%d' % (it, nt))  # Loading bar while sim is running

            # source = self.A * np.exp(-(t - self.tc)**2 / (2 * self.s_pulse**2))

            # self.Hz[y_source, x_source] += source  # Adding source term to propagation
            self.source_pw(t)
            self.update()  # Propagate over one time step

            if visu:
                # Presenting the Hz field
                # artists = [
                #     ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                #             size=plt.rcParams["axes.titlesize"],
                #             ha="center", transform=ax.transAxes),
                #     ax.imshow(self.Hz, vmin=-0.02 * self.A, vmax=0.02 * self.A, origin='upper',extent=(0,self.Lx,0,self.Ly))
                # ]
                artists = [
                    ax.text(0.5, 1.05, '%d/%d' % (it, nt),
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=ax.transAxes),
                    ax.pcolormesh(self.x_edges,self.y_edges,self.Hz,vmin=-0.02*self.A,vmax=0.02*self.A)
                ]
                for obs in sim.observation_points.values():
                    artists.append(ax.plot(obs.x, obs.y, 'ko', fillstyle="none")[0],)
                movie.append(artists)
        print('iterations done')
        if visu:
            my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000,
                                      blit=True)
            if saving:
                my_anim.save(filename='PW.gif', writer='pillow')
            plt.show()


def user_inputs():

    # 1. size of sim area
    Lx, Ly = map(float,input('Please provide the lengths Lx [cm] and Ly [cm] in Lx,Ly format: ').split(','))
    # 2. PW parameters
    A, s_pulse = map(float,input('Please provide the amplitude A of the source and the pulse width sigma in A,sigma format').split(','))
    epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
    mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of vacuum)
    c = 1 / np.sqrt(mu_0 * epsilon_0)
    l_min = 2 * np.pi * c * s_pulse / 3 # could also try / 5; w_max = 5 / s_pulse
    dx_min = l_min / 20
    CFL = 1
    dt = CFL / ( c * np.sqrt((1 / dx_min ** 2) + (1 / dx_min ** 2)))   # time step from spatial disc. & CFL
    tc = 5 * s_pulse                                                   # suggested tc >= 5 * sigma
    steps = input(f'For the given pulse width, timestep = {dt} and central time tc = {tc} are recommended; If '
          f'the user prefers manual input enter them in dt,tc format otherwise just enter.')
    if bool(steps):
        dt,tc = map(float,steps.split(','))
    # direction = input('Please provide the direction of the plane wave +x / -x / +y / -y')
    direction = '+x'
    PW = { 'A' : A , 's_pulse' : s_pulse , 'lmin' : l_min , 'dt' : dt, 'tc' : tc, 'direction' : direction}
    # 3. scatterers
    shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none): ')     #defien free later
    counter = 0
    scatter_list = []
    while shape != 'none':
        if shape == 'circle':
            xc,yc,r = input('Please provide the center coordinates and the radius in xc,xy,radius format: ').split(',')
            geometry = {'center': (float(xc), float(yc)), 'radius': float(r)}
        elif shape == 'rectangle':
            xi,xf,yi,yf = input('Please provide the coordinate ranges of the scatterer in '
                                'xmin,xmax,ymin,ymax format: ').split(',')
            geometry = { 'xi':float(xi), 'xf':float(xf), 'yi':float(yi), 'yf':float(yf) }
        else:
            # error handling
            shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none); typo: ')
            continue
        material = input('Please provide the type of material; PEC / PMC / Drude: ')
        if material == 'Drude':
            e_r,m_r, sigma, gamma  = input('Please provide the material properties in relative permittivity,relative permeability,sigma_DC,gamma format: ').split(',')
            properties = { 'e_r' : float(e_r) , 'm_r' : float(m_r), 'sigma_DC' : float(sigma) , 'gamma' : float(gamma)}
        else:
            properties = {}
        counter += 1
        scatter_list.append(Scatterer(shape, material ,counter, geometry , properties))
        shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none): ')
    # 4. location(s) of observation points (x,y)
    observation_points_lstr = input('Please provide the observation points in x1,y1;...;xn,yn format: ').split(';')
    obs_dict_tuples = {}
    for xy in observation_points_lstr:
        a,b = xy.split(',')
        obs_dict_tuples[(float(a), float(b))] = ObservationPoint(float(a),float(b))

    return Lx, Ly, PW, scatter_list, obs_dict_tuples

def testing(Lx:float, Ly:float,A, s_pulse,shape,xc,yc,r,material,e_r,m_r, sigma, gamma, observation_points_lstr):
    # 1. size of sim area
    # Lx, Ly = map(float,input('Please provide the lengths Lx [cm] and Ly [cm] in Lx,Ly format: ').split(','))
    # 2. PW parameters
    # A, s_pulse = map(float,input('Please provide the amplitude A of the source and the pulse width sigma in A,sigma format').split(','))
    epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
    mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of vacuum)
    c = 1 / np.sqrt(mu_0 * epsilon_0)
    l_min = 2 * np.pi * c * s_pulse / 3 # could also try / 5; w_max = 5 / s_pulse
    dx_min = l_min / 20
    CFL = 1
    dt = CFL / ( c * np.sqrt((1 / dx_min ** 2) + (1 / dx_min ** 2)))   # time step from spatial disc. & CFL
    tc = 5 * s_pulse                                                   # suggested tc >= 5 * sigma
    # steps = input(f'For the given pulse width, timestep = {dt} and central time tc = {tc} are recommended; If '
    #       f'the user prefers manual input enter them in dt,tc format otherwise just enter.')
    # if bool(steps):
    #     dt,tc = map(float,steps.split(','))
    direction = '+x'
    PW = { 'A' : A , 's_pulse' : s_pulse , 'lmin' : l_min , 'dt' : dt, 'tc' : tc, 'direction' : direction}
    # 3. scatterers
    # shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none): ')     #defien free later
    counter = 0
    scatter_list = []
    while shape != 'none':
        if shape == 'circle':
            # xc,yc,r = input('Please provide the center coordinates and the radius in xc,xy,radius format: ').split(',')
            geometry = {'center': (float(xc), float(yc)), 'radius': float(r)}
        elif shape == 'rectangle':
            xi,xf,yi,yf = input('Please provide the coordinate ranges of the scatterer in '
                                'xmin,xmax,ymin,ymax format: ').split(',')
            geometry = { 'xi':float(xi), 'xf':float(xf), 'yi':float(yi), 'yf':float(yf) }
        else:
            # error handling
            shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none); typo: ')
            continue
        # material = input('Please provide the type of material; PEC / PMC / Drude: ')
        if material == 'Drude':
            # e_r,m_r, sigma, gamma  = input('Please provide the material properties in relative permittivity,relative permeability,sigma_DC,gamma format: ').split(',')
            properties = { 'e_r' : float(e_r) , 'm_r' : float(m_r), 'sigma_DC' : float(sigma) , 'gamma' : float(gamma)}
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
        obs_dict_tuples[(float(a), float(b))] = ObservationPoint(float(a),float(b))

    return Lx, Ly, PW, scatter_list, obs_dict_tuples


# sim = FDTD(*user_inputs())

sim = FDTD(*testing(20.0,20.0,1,0.0000000016,'circle',10,10,1,'Drude',
                    10,10,10000000,10000000000000,['1.1,2','7,7']))


#  lamda ~ 5 micrometer -> ~ Thz freq
# sigma_for_Thz = 5 / ( np.pi * 2) * 10**(-14)        # [a cm]
# lamda_for_Thz = 2 * np.pi * 10**8 * sigma_for_Thz # [cm]
# L_for_Thz = 20 * lamda_for_Thz
# g_for_Thz = 10**(-12)
# print(L_for_Thz)
# sim = FDTD(*testing(L_for_Thz,L_for_Thz,1,sigma_for_Thz,'circle',L_for_Thz/2,L_for_Thz/2,L_for_Thz/10,'Drude',
#                     1,1,670000,g_for_Thz,['00,0','0,0']))

# sim.debugger(show_grid=False)
# sim.update()
# nt = (sim.Lx / 1) / (sim.dt * sim.c)
nt = 200

y_source = sim.Ny // 2
x_source = 10
sim.iterate(int(nt), visu = True, saving=False)
# print((sim.Jcx==0).all())


# for obs in sim.observation_points.values():       #this is how we can access observation points after sim runs
#     print(obs.ex_values)



