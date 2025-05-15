#100start building as I read ( start from what I know; simple update eq)
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

class Scatterer:

    def __init__(self , shape:str , material:str , ID:int , geometry:dict , properties:dict ):
        self.shape = shape
        self.ID = ID                    #useful for up-eq and mask
        self.material = material        #pec / pmc / drude
        self.geometry = geometry        #depends on shape
        self.properties = properties    #e,m


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

    def __init__(self, Lx:float , Ly:float , PW , scatterer_list:list , observation_points):

        # constants
        self.epsilon_0 = 8.8541878128e-12  # F/m (permittivity of vacuum)
        self.mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of vacuum)
        self.c = 1 / np.sqrt(self.mu_0 * self.epsilon_0)  # Speed of light in vacuum
        self.Lx = Lx
        self.Ly = Ly
        self.lmin = PW      #fix after ch5 implementation (need to choose waveform for now)
        self.dt = PW/ self.c    #fix
        self.dx_coarse = self.lmin / 10
        self.dx_inter1 = self.dx_coarse / ( 2 ** (1/3) )
        self.dx_inter2 = self.dx_inter1 / ( 2 ** (1/3) )
        self.dx_fine = self.lmin / 20
        self.scatterer_list = scatterer_list
        self.observation_points = observation_points
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
        Ny, Nx = self.Hz.shape
        print(f"Ny:{Ny}, Nx:{Nx}")
        self.Ex = np.zeros((Ny-1,Nx))
        self.Ey = np.zeros((Ny, Nx-1))
        self.epsilon_grid = np.full(self.Hz.shape,self.epsilon_0)
        self.epsilon_yavg = np.zeros_like(self.Ex)                 #epsilon spatially averaged over y, used for Ex update
        self.epsilon_xavg = np.zeros_like(self.Ey)                 #epsilon spatially averaged over x, used for Ey update
        self.mu_grid = np.full(self.Hz.shape, self.mu_0)


        #initialize auxilary fields
        self.Hzd = np.zeros_like(self.Hz)
        print(f"Ny:{Ny}, Nx:{Nx}")
        self.Exd = np.zeros((Ny-1,Nx))          
        self.Eyd = np.zeros((Ny, Nx-1))
        self.Exdd = np.zeros((Ny-1,Nx))
        self.Eydd = np.zeros((Ny, Nx-1))


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
                self.epsilon_grid[self.mask_Hz == scatterer.ID] = scatterer.properties['e']
                self.mu_grid[self.mask_Hz == scatterer.ID] = scatterer.properties['m']
                #unphysical, is storing e_0,m_0 for PEC & PMC, used for the avg, locations with PMC/PEC use BC updates
        self.epsilon_yavg = (self.epsilon_grid[:-1, :] + self.epsilon_grid[1:, :]) / 2
        self.epsilon_xavg = (self.epsilon_grid[:,:-1] + self.epsilon_grid[:,1:]) / 2
        self.mu_crossavg = cross_average2Drray(self.mu_grid)

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
        # in 0

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

        #here include logic for sc.type -> set 0 or whatever...

    
    def pml_mask(self,field,percentageN=0.10):
        """
        creates the fieldmask for the pml in 2d 

        choosing the percentages means choosing how much percent of the border will be the PML
        """

        
        Ny,Nx = np.shape(field)
        self.Lx=5
        self.Ly=5
        
        percentage= 1/2-percentageN/2
        self.percentage=percentage
        # Define the 2D Heaviside function
        def heaviside_2d(x, y):

            H1 = np.heaviside( (1/2-percentage)*Lx - x, 1 ) * np.heaviside( ( 1/2-percentage )* Ly-x, 1)

            H2 = np.heaviside((1/2-percentage)*Lx-y, 1) * np.heaviside((1/2-percentage)*Ly-y, 1)

            H3 = np.heaviside(-(1/2+percentage)*Lx+x, 1) * np.heaviside(-(1/2+percentage)*Ly+x, 1)

            H4 = np.heaviside(-(1/2+percentage)*Lx+y, 1) * np.heaviside(-(1/2+percentage)*Ly+y, 1)

            combined = np.logical_or.reduce([H1, H2, H3, H4])
            return combined

        # Create a grid
        x = np.linspace(0, self.Lx,Nx)
        y = np.linspace(0, self.Ly, Ny)
        X, Y = np.meshgrid(x, y)

        # Apply the Heaviside function     
        H = heaviside_2d(X, Y)
                


        return field*H


    def pml(self, sigmamax=0.17,Kappamax=3,thickness_denom = 10, pml_order = 4 ,scale=1):
        """
        pml_order: profile of pml; higher order means less reflections from inner side
        prof_type: either polynomial or exponential
        scale: to further reduce lattice mismatch of inner side of PML
        """
        # thickness needs to be small relatively to nx
        self.pml_thickness = max(1, self.Nx // thickness_denom)
        # damping coëfficient grid in x and y
        self.dampx = np.zeros(self.Nx + 1)
        self.dampy = np.zeros(self.Ny + 1)
        self.Kappax = np.zeros(self.Nx + 1)
        self.Kappay = np.zeros(self.Ny + 1)
        def profile(index, thickness, sigmamax,pml_order):
            return sigmamax * ((index / thickness) ** pml_order) 
        # only uses the values in the pml layer so that the non pml layer doesnt get affected
        for i in range(self.pml_thickness):
            sigma_value = profile(i, self.pml_thickness, sigmamax, pml_order, scale)

            self.dampx[i] = sigma_value
            self.dampx[-(i + 1)] = sigma_value
            self.dampy[i] = sigma_value
            self.dampy[-(i + 1)] = sigma_value
            #print(self.dampx)          #debugger, inside values higher??
        return None
    
    
    def pml2(self, sigmamax=1,thickness_denom = 10, pml_order = 3,prof_type = 'polynomial',scale=1):
        """
        pml_order: profile of pml; higher order means less reflections from inner side
        prof_type: either polynomial or exponential
        scale: to further reduce lattice mismatch of inner side of PML
        
        Add the mask to the the fields and make it such that when it starts the Updated PML equations from 1/5th and 4/5th of
        
        TO-DO:

        Firstly create a succesfull mask that takes gridcells and make it pml grid: Done

        second: figure out what the update equations in time domain

        Third: apply it to the update equation
        
        """




        #change of conductivity with regard of sigmamax
        def sigmaprof(x=[],L=10,sigmamax=1):
            
            sigma=sigmamax * np.sin(np.pi * x /(2*L) )**2

            return sigma

        sigmax=sigmaprof(self.x,self.Lx)  
        sigmay=sigmaprof(self.y,self.Ly)  

        # thickness needs to be small relatively to nx
        self.pml_thicknessx= max(1, self.nx // thickness_denom)
        self.pml_thicknessy= max(1, self.nx // thickness_denom)
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

    def source_pw(self, Aetc):
        pass

    def observation_points(self):
        # use the x and y edges to locate physical space (x,y) points into our discrete grid
        pass

    def debugger(self,show_grid = False):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.contourf(self.Xc, self.Yc, self.mask_Hz, levels=[0.5, 1], colors='black', linestyles='--')
        ax.plot(1, 5, 'rs', fillstyle="none", label='Source')

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










def user_inputs():

    # 1. size of sim area
    Lx , Ly = input('Please provide the lengths Lx [cm] and Ly [cm] in Lx,Ly format: ').split(',')
    Lx = float(Lx)
    Ly = float(Ly)
    # 3. PW parameters
    # suppose I have l_min etc from PW after ch5
    l_min = 1
    # 2. gridding & timestep : after PW to have dx, CFL and after scatterers for gridding
    #first calc min dx & dy, from there show suggested CFL and ask for imput

    # 4. scatterers
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
            e,m = input('Please provide the material properties in permittivity,permeability format: ').split(',')
            properties = { 'e' : float(e) , 'm' : float(m)}
        else:
            properties = {}
        counter += 1
        scatter_list.append(Scatterer(shape, material ,counter, geometry , properties))
        shape = input('Please provide the shape of the scatterer (circle or rectangle or free or none): ')
   
   
   # 5. location(s) of observation points (x,y)
    observation_points_lstr = input('Please provide the observation points in x1,y1;...;xn,yn format: ').split(';')
    obs_list_tuples = []
    for xy in observation_points_lstr:
        a,b = xy.split(',')
        obs_list_tuples.append((float(a),float(b)))

    return Lx, Ly, l_min, scatter_list, obs_list_tuples

#sim = FDTD(*user_inputs())
sim = FDTD()
sim.debugger(show_grid=True)
sim.update()

