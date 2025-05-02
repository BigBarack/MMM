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
        self.Z0=(self.mu_0/self.epsilon_0)**(1/2)
        # sim area
        self.Lx = Lx
        self.Ly = Ly
        # Plane Wave
        self.lmin = PW['lmin']
        self.dt = PW['dt']
        self.tc = PW['tc']
        self.A = PW['A']
        self.s_pulse = PW['s_pulse']
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


        #initialize auxilary fields
        self.Hzd = np.zeros_like(self.Hz)
        print(f"Ny:{self.Ny}, Nx:{self.Nx}")
        self.Exd = np.zeros((self.Ny-1,self.Nx))          
        self.Eyd = np.zeros((self.Ny, self.Nx-1))
        self.Exdd = np.zeros((self.Ny-1,self.Nx))
        self.Eydd = np.zeros((self.Ny, self.Nx-1))

        #initialize also the new fields for update equations
        self.Hzdn=self.Hzd
        self.Exdn=self.Exd
        self.Eydn=self.Eyd
        self.Exddn=self.Exdd
        self.Eyddn=self.Eydd

        # generate mask
        self.mask_Hz = np.zeros_like(self.Hz)
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
        self.update_observation_points()


    def update_PML(self):
        """"These are the upaate equations with the PML taken from  2.133-2.138"""
        
        
        # creating the coefficients:
        self.beta_xp=self.kappax_2D/(self.c*self.dt) + self.Z0*self.sigmax_2D/2
        self.beta_xm=self.kappax_2D/(self.c*self.dt) - self.Z0*self.sigmax_2D/2
        self.beta_yp=self.kappay_2D/(self.c*self.dt) + self.Z0*self.sigmay_2D/2
        self.beta_ym=self.kappay_2D/(self.c*self.dt) - self.Z0*self.sigmay_2D/2
        self.beta_zp=self.kappaz_2D/(self.c*self.dt) + self.Z0*self.sigmaz_2D/2
        self.beta_zm=self.kappaz_2D/(self.c*self.dt) - self.Z0*self.sigmaz_2D/2
        
        
        
        self.dtau=self.dt*self.c

        #updating all the fields
        self.Hzdn[1:-1,1:-1] = (self.beta_xm[1:-1,1:-1]*self.Hzd[1:-1,1:-1]                                                    #Hz term
                               
                               - 1/self.DX_Hz[1:-1,1:-1]*(self.Ey[1:-1,:-1] - self.Ey[1:-1,1:])         #Ey term

                               + ((self.Ex[1:,1:-1] - self.Ex[:-1,1:-1]) / self.DY_Hz[1:-1,1:-1] )      #Ex term

                                )/self.beta_xp[1:-1,1:-1]
        self.Hz[1:-1,1:-1] = (self.beta_ym[1:-1,1:-1]*self.Hz[1:-1,1:-1]                                                    #Hz term
                               
                               + self.beta_zp[1:-1,1:-1]*self.Hzdn[1:-1,1:-1]- self.beta_zp[1:-1,1:-1]*self.Hzd[1:-1,1:-1]  
                               
                               )/self.beta_yp[1:-1,1:-1]
        

        #when there is a Drude media make sure that these are accounted for as well


        """if self.there_is_Drude: #run ADEs to update Jc x,y then add that to currently calculated E
            for scatterer in self.scatterer_list:
                if scatterer.material == 'Drude':
                    g = scatterer.properties['gamma']
                    sigma = scatterer.properties['sigma_DC']
                    gt=self.c*g
                    alphap=2*gt/self.dtau+1
                    alpham=2*gt/self.dtau-1
                    deltap= 1/self.dtau+self.Z0*sigma/(2*alphap)
                    deltam= 1/self.dtau-self.Z0*sigma/(2*alphap)
                    deltaw= 1/2 * ( 1 + alpham/alphap)
                    maskx = self.mask_Ex == scatterer.ID
                    masky = self.mask_Ey == scatterer.ID
                    Hmaskx= self.mask_Hz == scatterer.ID
                    Hmasky= self.mask_Hz == scatterer.ID
                    Hz_diff_x = np.zeros_like(self.Hz)
                    Hz_diff_x[1:, :] = self.Hz[1:, :] - self.Hz[:-1, :]
                    Hz_diff_y = np.zeros_like(self.Hz)
                    Hz_diff_y[:,1:] = self.Hz[:, 1:] - self.Hz[:, :-1]
                    #!NOT SURE IF THIS MASKING SHIFT IS GOOD FOR BOUNDARIES SEE

                    self.Exddn[maskx]=(deltam *self.Z0 *self.Exdd[maskx]-deltaw*self.Jcx[maskx] 
                                       
                                       + 1/self.DY_Hz * (Hz_diff_y[Hmaskx])
                                       )/( deltap * self.Z0)
                    
                    self.Eyddn[masky]=(deltam *self.z0 *self.Eydd[masky]-deltaw*self.Jcy[masky]         
                                        - 1/self.DX_Hz * (Hz_diff_x[Hmasky])
                                       )/( deltap * self.Z0)
                    
                    self.Jcx[maskx] = ( alpham * self.Jcx[maskx]
                                       
                                       + self.Z0*sigma * (self.Z0* (self.Exddn[maskx]+self.Exdd[maskx]))


                                        )/alphap



                    self.Jcy[masky] = ( alpham * self.Jcy[masky]
                                       
                                       + self.Z0*sigma * (self.Z0* (self.Eyddn[masky]+self.Eydd[masky]))
                                       

                                        )/alphap"""
        self.Exdn=(self.beta_ym[1:,:] * self.Exd 
                   + 1/self.dtau*(self.Exddn-self.Exdd)

                    ) / self.beta_yp[1:,:]
        self.Eydn=(self.beta_zm[:,:1] * self.Eyd 
                   + 1/self.dtau*(self.Eyddn-self.Eydd)

                    ) / self.beta_zp[:,:1]
        
        self.Ex=(self.beta_zm[1:,:] * self.Ex 
                   + self.beta_xp[1:,:] *(self.Exdn-self.Exd)

                    ) / self.beta_zp[1:,:]
        self.Ey=(self.beta_xm[:,1:] * self.Ey 
                   + self.beta_yp[:,1:]*(self.Eydn-self.Eyd)

                    ) / self.beta_xp[:,1:]
        
        #Boundary conditions
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
        self.update_observation_points()
        
        
        
        
        #for next update make sure that we shift one timestepfurther

        self.Hzd=self.Hzdn
        self.Exdd=self.Exddn
        self.Eydd=self.Eyddn
        self.Eyd=self.Eydn
        self.Exd=self.Exdn


        return None
    
    def pml(self, kappamax=3,thickness_denom = 10, pml_order = 4 ,scale=1):
        """
        pml_order: profile of pml; higher order means less reflections from inner side
        prof_type: either polynomial or exponential
        scale: to further reduce lattice mismatch of inner side of PML
        """
        sigmamax= (pml_order+1)/(150*np.pi)
        # thickness needs to be small relatively to nx
        self.pml_thickness = max(1, self.Nx // thickness_denom)
        # damping coëfficient grid in x and y and kappax and kappy
        self.sigmax = np.zeros(self.Nx )
        self.sigmay = np.zeros(self.Ny )
        self.kappax = np.zeros(self.Nx )
        self.kappay = np.zeros(self.Ny )

        #### NOT SO SURE ABOUT THE PLUS ONE HERE MIGHT ADJUST

        def sigmaprofile(index, thickness, sigmamax,pml_order):
            return sigmamax * ((index / thickness) ** pml_order) 
        def kappaprofile(index, thickness, sigmamax,pml_order):
            return 1+ (kappamax -1)* ((index / thickness) ** pml_order) 
        # only uses the values in the pml layer so that the non pml layer doesnt get affected
        for i in range(self.pml_thickness):
            sigma_val = sigmaprofile(i, self.pml_thickness, sigmamax, pml_order)
            kappa_val = kappaprofile(i, self.pml_thickness, sigmamax, pml_order)

            self.sigmax[i] = sigma_val/self.dx[i]
            self.sigmax[-(i + 1)] = sigma_val/self.dx[-(i+1)]
            self.sigmay[i] = sigma_val/self.dy[i]
            self.sigmay[-(i + 1)] = sigma_val/self.dy[-(i+1)]


            self.kappax[i] = kappa_val / self.dx[i]
            self.kappax[-(i + 1)] = kappa_val / self.dx[-(i + 1)]
            self.kappay[i] = kappa_val / self.dy[i]
            self.kappay[-(i + 1)] = kappa_val / self.dy[-(i + 1)]
            #print(self.dampx)          #debugger, inside values higher??

        #extend over the full axis

        self.sigmax_2D = np.tile(self.sigmax[:, np.newaxis], (1, self.Ny ))
        self.sigmay_2D = np.tile(self.sigmay[np.newaxis, :], (self.Nx , 1))

        self.kappax_2D = np.tile(self.kappax[:, np.newaxis], (1, self.Ny ))
        self.kappay_2D = np.tile(self.kappay[np.newaxis, :], (self.Nx , 1))
        
        
        
        # Final 2D UPML fields using additive damping
        self.sigmaz_2D = self.sigmax_2D + self.sigmay_2D
        self.kappaz_2D = self.kappax_2D * self.kappay_2D

        
        print(self.sigmax_2D)   #Debugging
        return None
    

    def source_pw(self, Aetc):
        pass

    def update_observation_points(self):

        for obs in self.observation_points.values():
            hz = self.Hz[*obs.coordinates['hz']]
            ex = self.Ex[*obs.coordinates['ex']]
            ey = self.Ey[*obs.coordinates['ey']]
            obs.add_sample(ex,ey,hz)

    def debugger(self,show_grid = False):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.contourf(self.Xc, self.Yc, self.mask_Hz, levels=[0.5, 1], colors='black', linestyles='--')
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

    def iterate(self,nt,y_source, x_source):

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

            source = self.A * np.exp(-(t - self.tc)**2 / (2 * self.s_pulse**2))

            self.Hz[y_source, x_source] += source  # Adding source term to propagation
            self.update_PML()  # Propagate over one time step


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
        my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000,
                                  blit=True)
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
    PW = { 'A' : A , 's_pulse' : s_pulse , 'lmin' : l_min , 'dt' : dt, 'tc' : tc}
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
    PW = { 'A' : A , 's_pulse' : s_pulse , 'lmin' : l_min , 'dt' : dt, 'tc' : tc}
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
                    10,1,10000000,10000000000000,['1.1,2','7,7']))

print(sim.there_is_Drude)
print((sim.Jcx==0).all())
# sim.debugger(show_grid=False)
# sim.update()
nt = (sim.Lx / 2) / (sim.dt * sim.c)
y_source = sim.Ny // 2
x_source = sim.Nx // 3
sim.pml(3,10,4,1)
sim.iterate(int(nt+100),y_source,x_source)
sim1 = FDTD(*testing(20.0,20.0,1,0.0000000016,'circle',10,10,1,'Drude',
                    10,1,10000000,10000000000000,['1.1,2','7,7']))

#print(sim1.there_is_Drude)
#print((sim1.Jcx==0).all())
# sim.debugger(show_grid=False)
# sim.update()
#nt = (sim1.Lx / 2) / (sim1.dt * sim1.c)
y_source = sim1.Ny // 2
x_source = sim1.Nx // 3
#sim1.iterate(int(nt+100),y_source,x_source)







# for obs in sim.observation_points.values():       #this is how we can access observation points after sim runs
#     print(obs.ex_values)

