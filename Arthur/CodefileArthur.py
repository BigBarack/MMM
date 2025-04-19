import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import time
import scipy as sc
import copy
matplotlib.use('Qt5Agg')

epsilon0 = 8.8541878127e-12
mu0 = 1.25663706127e-6
c = np.sqrt(1/(epsilon0*mu0))

"Main object of the simulation"
class FTDT:
    def __init__(self, grid, objects, tfsf, viewpoints, dt, ntimesteps, BC = "PBC", ax = None):
        self.grid = grid
        self.objects = objects
        self.tfsf = tfsf
        self.viewpoints = viewpoints
        self.BC = BC
        self.dt = dt
        self.ntimesteps = ntimesteps
        self.t = 0
        self.ax = ax
        self.movie = []
        if BC == "PBC":
            self.grid.Hz = self.grid.Hz[:-1,:-1]
            self.grid.cHz = self.grid.cHz[:-1,:-1]
            self.grid.Ex = self.grid.Ex[:-1,:]
            self.grid.Ey = self.grid.Ey[:,:-1]
            self.grid.cEx,self.grid.sEx = self.grid.cEx[:-1,:],self.grid.sEx[:-1,:]
            self.grid.cEy,self.grid.sEy = self.grid.cEy[:,:-1],self.grid.sEy[:,:-1]


    def update(self):
        """
        if self.BC == "PBC":
            self.grid.Ex = np.multiply(self.grid.sEx,self.grid.Ex) + np.multiply(self.grid.cEx,(np.roll(self.grid.Hz,1,1)-self.grid.Hz))
            self.grid.Ey = np.multiply(self.grid.sEy,self.grid.Ey) + np.multiply(self.grid.cEy,(-np.roll(self.grid.Hz,1,0)+self.grid.Hz))
            self.grid.Hz = self.grid.Hz + np.multiply(self.grid.cHz,(np.roll(self.grid.Ey,-1,0)-self.grid.Ey-np.roll(self.grid.Ex,-1,1)+self.grid.Ex))
        """
        self.tfsf.calcwave(self.t,self.grid)
        for i in range(1,np.shape(self.grid.Ex)[0]):
            for j in range(1,np.shape(self.grid.Ex)[1]):
                self.grid.Ex[i,j] = self.grid.sEx[i,j]*self.grid.Ex[i,j] + self.grid.cEx[i,j]*(self.grid.Hz[i,j]-self.grid.Hz[i,j-1]) 

        for i in range(1,np.shape(self.grid.Ey)[0]):
            for j in range(1,np.shape(self.grid.Ey)[1]):
                self.grid.Ey[i,j] = self.grid.sEy[i,j]*self.grid.Ey[i,j] + self.grid.cEy[i,j]*(-self.grid.Hz[i,j]+self.grid.Hz[i-1,j]) 
        
        for i in range(1,np.shape(self.grid.Hz)[0]-1):
            for j in range(1,np.shape(self.grid.Hz)[1]-1):
                self.grid.Hz[i,j] = self.grid.Hz[i,j] + self.grid.cHz[i,j]*(self.grid.Ey[i+1,j]-self.grid.Ey[i,j]-self.grid.Ex[i,j+1]+self.grid.Ex[i,j]) 


        for i in range(1,np.shape(self.grid.Ex)[0]):
            for j in range(1,np.shape(self.grid.Ex)[1]):
                if self.tfsf.tfsfleft<=i<self.tfsf.tfsfright and j == self.tfsf.tfsftop:
                    self.grid.Ex[i,j] += self.grid.cEx[i,j]*(self.tfsf.HtfsfTOP[i-self.tfsf.tfsfleft])
                #elif self.tfsf.tfsfleft<=i<self.tfsf.tfsfright and j == self.tfsf.tfsftop+1:
                #    self.grid.Ex[i,j] -= self.grid.cEx[i,j]*(-self.tfsf.HtfsfTOP[i-self.tfsf.tfsfleft])
                elif self.tfsf.tfsfleft<=i<self.tfsf.tfsfright and j == self.tfsf.tfsfbottom:
                    self.grid.Ex[i,j] -= self.grid.cEx[i,j]*(self.tfsf.HtfsfBOTTOM[i-self.tfsf.tfsfleft])
                #elif self.tfsf.tfsfleft<=i<self.tfsf.tfsfright and j == self.tfsf.tfsfbottom-1:
                #    self.grid.Ex[i,j] -= self.grid.cEx[i,j]*(self.tfsf.HtfsfBOTTOM[i-self.tfsf.tfsfleft])


        for i in range(1,np.shape(self.grid.Ey)[0]):
            for j in range(1,np.shape(self.grid.Ey)[1]):
                if self.tfsf.tfsfbottom<=j<self.tfsf.tfsftop and i == self.tfsf.tfsfleft:
                    self.grid.Ey[i,j] += self.grid.cEy[i,j]*(self.tfsf.HtfsfLEFT[j-self.tfsf.tfsfbottom])
                #elif self.tfsf.tfsfbottom<=i<=self.tfsf.tfsftop and j == self.tfsf.tfsfleft-1:
                #    self.grid.Ey[i,j] -= self.grid.cEy[i,j]*(-self.tfsf.HtfsfLEFT[j-self.tfsf.tfsfbottom+1])
                elif self.tfsf.tfsfbottom<=j<self.tfsf.tfsftop and i == self.tfsf.tfsfright:
                    self.grid.Ey[i,j] -= self.grid.cEy[i,j]*(self.tfsf.HtfsfRIGHT[j-self.tfsf.tfsfbottom])
                #elif self.tfsf.tfsfbottom<=i<=self.tfsf.tfsftop and j == self.tfsf.tfsfright:
                #    self.grid.Ey[i,j] -= self.grid.cEy[i,j]*(self.tfsf.HtfsfRIGHT[j-self.tfsf.tfsfbottom-1])
        
        for i in range(np.shape(self.grid.Hz)[0]-1):
            for j in range(np.shape(self.grid.Hz)[1]-1):
                #print(self.grid.Hz[i,j])
                #print(self.grid.Hz[i,j]+ self.grid.cHz[i,j]*(self.grid.Ey[i+1,j]-self.grid.Ey[i,j]-self.grid.Ex[i,j+1]+self.grid.Ex[i,j]))
                if self.tfsf.tfsfbottom<=j<self.tfsf.tfsftop and i == self.tfsf.tfsfleft:
                    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfLEFT[j-self.tfsf.tfsfbottom])
                #elif self.tfsf.tfsfbottom<=i<=self.tfsf.tfsftop and j == self.tfsf.tfsfleft-1:
                #    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfLEFT[j-self.tfsf.tfsfbottom+1])
                elif self.tfsf.tfsfbottom<=j<self.tfsf.tfsftop and i == self.tfsf.tfsfright:
                    self.grid.Hz[i,j] += self.grid.cHz[i,j]*(self.tfsf.EtfsfRIGHT[j-self.tfsf.tfsfbottom])
                #elif self.tfsf.tfsfbottom<=i<=self.tfsf.tfsftop and j == self.tfsf.tfsfright:
                #    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfRIGHT[j-self.tfsf.tfsfbottom-1])
                elif self.tfsf.tfsfleft<=i<self.tfsf.tfsfright and j == self.tfsf.tfsftop:
                    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfTOP[i-self.tfsf.tfsfleft])
                #elif self.tfsf.tfsfleft<=i<=self.tfsf.tfsfright and j == self.tfsf.tfsftop:
                #    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfTOP[i-self.tfsf.tfsfright-1])
                elif self.tfsf.tfsfleft<=i<self.tfsf.tfsfright and j == self.tfsf.tfsfbottom:
                    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfBOTTOM[i-self.tfsf.tfsfleft])
                #elif self.tfsf.tfsfleft<=i<=self.tfsf.tfsfright and j == self.tfsf.tfsfbottom-1:
                #    self.grid.Hz[i,j] -= self.grid.cHz[i,j]*(self.tfsf.EtfsfBOTTOM[j-self.tfsf.tfsfright+1])

        self.t += self.dt
        if self.ax:
            return [self.ax.imshow(np.transpose(self.grid.Hz),vmin=-3,vmax=3)]


        
    def run(self):
        self.movie = []
        i = 0
        for i in range(self.ntimesteps):
            self.movie.append(self.update())
        return(self.grid.Ex,self.grid.Ey,self.grid.Hz)
        



"Define grid"
class grid:

    def __init__(self, SizeX, SizeY, dX, dY, permittivity = epsilon0, permeability = mu0, conductivity = 0
                , xDiscretization = [], yDiscretization = [], xDualDiscretization = [], yDualDiscretization = [], Ex = [], Ey = [], Hz= []
                , sEx = [], sEy = [], cEx = [], cEy = [], cHz = []):
        self.SizeX = float(SizeX)
        self.SizeY = float(SizeY)
        self.dX = float(dX)
        self.dY = float(dY)
        self.backgroundpermittivity = permittivity
        self.backgroundpermeability = permeability
        self.backgroundconductivity = conductivity
        self.xDiscretization = xDiscretization
        self.yDiscretization = yDiscretization
        self.xDualDiscretization = xDualDiscretization
        self.yDualDiscretization = yDualDiscretization
        self.Ex = Ex
        self.Ey = Ey
        self.Hz = Hz
        self.sEx = sEx
        self.sEy = sEy
        self.cEx = cEx
        self.cEy = cEy
        self.cHz = cHz


    def generate_grid(self, objects):
        xDiscretization = []
        x = 0
        while x <= self.SizeX:
            loc_dx = self.dX
            for i in objects:
                if x >= i.x and x<= i.x+i.Dx:
                    loc_dx = self.dX/i.xrefinement
            x += loc_dx
            xDiscretization.append(loc_dx)
        xDualDiscretization = [(xDiscretization[i+1]+xDiscretization[i])/2 for i in range(len(xDiscretization)-1)]

        yDiscretization = []
        y = 0
        while y <= self.SizeY:
            loc_dy = self.dY
            for i in objects:
                if y >= i.y and y<= i.y+i.Dy:
                    loc_dy = self.dY/i.yrefinement
            y += loc_dy
            yDiscretization.append(loc_dy)
        yDualDiscretization = [(yDiscretization[i+1]+yDiscretization[i])/2 for i in range(len(yDiscretization)-1)]
        Ex = np.zeros((len(xDiscretization),len(yDualDiscretization)))
        Ey = np.zeros((len(xDualDiscretization),len(yDiscretization)))
        Hz = np.zeros((len(xDiscretization),len(yDiscretization)))
        self.Ex = Ex
        self.Ey = Ey
        self.Hz = Hz
        self.xDiscretization = xDiscretization
        self.yDiscretization = yDiscretization
        self.xDualDiscretization = xDualDiscretization
        self.yDualDiscretization = yDualDiscretization
        return(Ex,Ey,Hz,xDiscretization,xDualDiscretization,yDiscretization,yDualDiscretization)
    


    #Currently doesn't properly average epsilon yet
    def generate_sE(self, objects, timestep):
        sEx = np.zeros((len(self.xDiscretization),len(self.yDualDiscretization)))
        sEy = np.zeros((len(self.xDualDiscretization),len(self.yDiscretization)))
        x=0
        for i in range(len(self.xDiscretization)):
            y=0
            for j in range(len(self.yDualDiscretization)):
                for k in objects:
                    if k.inside(x,y):
                        sEx[i,j] = ((k.permittivity/timestep)-(k.conductivity/2))/((k.permittivity/timestep)+(k.conductivity/2))
                    else:
                        sEx[i,j] = ((self.backgroundpermittivity/timestep)-(self.backgroundconductivity/2))/((self.backgroundpermittivity/timestep)+(self.backgroundconductivity/2))
                    x+=self.xDiscretization[i]
                    y+=self.yDualDiscretization[j]
        x=0
        for i in range(len(self.xDualDiscretization)):
            y=0
            for j in range(len(self.yDiscretization)):
                for k in objects:
                    if k.inside(x,y):
                        sEy[i,j] = ((k.permittivity/timestep)-(k.conductivity/2))/((k.permittivity/timestep)+(k.conductivity/2))
                    else:
                        sEy[i,j] = ((self.backgroundpermittivity/timestep)-(self.backgroundconductivity/2))/((self.backgroundpermittivity/timestep)+(self.backgroundconductivity/2))
                    x+=self.xDualDiscretization[i]
                    y+=self.yDiscretization[j]
        self.sEx = sEx
        self.sEy = sEy
        return(sEx,sEy)
    
    def generate_cE(self, objects, timestep):
        cEx = np.zeros((len(self.xDiscretization),len(self.yDualDiscretization)))
        cEy = np.zeros((len(self.xDualDiscretization),len(self.yDiscretization)))
        x=0
        for i in range(len(self.xDiscretization)):
            y=0
            for j in range(len(self.yDualDiscretization)):
                for k in objects:
                    if k.inside(x,y):
                        cEx[i,j] = (self.xDiscretization[i]/self.yDualDiscretization[j])/((k.permittivity/timestep+k.conductivity/2))
                    else:
                        cEx[i,j] = (self.xDiscretization[i]/self.yDualDiscretization[j])/((self.backgroundpermittivity/timestep+self.backgroundconductivity/2))
                    y+=self.yDualDiscretization[j]
            x+=self.xDiscretization[i]
        x=0
        for i in range(len(self.xDualDiscretization)):
            y=0
            for j in range(len(self.yDiscretization)):
                for k in objects:
                    if k.inside(x,y):
                        cEy[i,j] = (self.yDiscretization[j]/self.xDualDiscretization[i])/((k.permittivity/timestep+k.conductivity/2))
                    else:
                        cEy[i,j] = (self.yDiscretization[j]/self.xDualDiscretization[i])/((self.backgroundpermittivity/timestep+self.backgroundconductivity/2))
                    y+=self.yDiscretization[j]
            x+=self.xDualDiscretization[i]
        self.cEx = cEx
        self.cEy = cEy
        return(cEx,cEy)

    def generate_cH(self, objects, timestep):
        cHz = np.zeros((len(self.xDiscretization),len(self.yDiscretization)))
        x=0
        for i in range(len(self.xDiscretization)):
            y=0
            for j in range(len(self.yDiscretization)):
                for k in objects:
                    if k.inside(x,y):
                        cHz[i,j] = -(1/(self.xDiscretization[i]*self.yDiscretization[j]))*(timestep/k.permeability)
                    else:
                        cHz[i,j] = -(1/(self.xDiscretization[i]*self.yDiscretization[j]))*(timestep/self.backgroundpermeability)
                    y+=self.yDiscretization[j]
            x+=self.xDiscretization[i]
        self.cHz = cHz
        return(cHz)
    


class tfsf:
    def __init__(self
                 ,tfsfbottom, tfsftop, tfsfleft, tfsfright
                 , direction = 0, type ="Gaussian", amplitude=1):
        self.direction = direction
        self.type = type
        self.amplitude = amplitude
        self.tfsfbottom = int(tfsfbottom)
        self.tfsftop = int(tfsftop)
        self.tfsfleft = int(tfsfleft)
        self.tfsfright = int(tfsfright)
        self.EtfsfLEFT = np.zeros(int(tfsftop-tfsfbottom))
        self.EtfsfRIGHT = np.zeros(int(tfsftop-tfsfbottom))
        self.EtfsfTOP = np.zeros(int(tfsfright-tfsfleft))
        self.EtfsfBOTTOM = np.zeros(int(tfsfright-tfsfleft))
        self.HtfsfLEFT = np.zeros(int(tfsftop-tfsfbottom))
        self.HtfsfRIGHT = np.zeros(int(tfsftop-tfsfbottom))
        self.HtfsfTOP = np.zeros(int(tfsfright-tfsfleft))
        self.HtfsfBOTTOM = np.zeros(int(tfsfright-tfsfleft))

    def calcwave(self,t,grid):
        xinit = 0.02
        sigma = 0.00001
        x = xinit
        for i in range(len(self.EtfsfLEFT)):
            self.EtfsfLEFT[i] = self.amplitude*np.exp(-(c*t-x)**2/(2*sigma))
            #self.EtfsfLEFT[i] = self.amplitude*np.exp(-(c*t-x)**2/2*1)
        for i in range(len(self.EtfsfBOTTOM)):
            self.EtfsfTOP[i] = self.amplitude*np.exp(-(c*t-x)**2/(2*sigma))
            self.EtfsfBOTTOM[i] = self.amplitude*np.exp(-(c*t-x)**2/(2*sigma))
            x += grid.xDiscretization[i+self.tfsfleft]
        for i in range(len(self.EtfsfRIGHT)):
            self.EtfsfRIGHT[i] = self.amplitude*np.exp(-(c*t-x)**2/(2*sigma))
            #self.EtfsfRIGHT[i] = self.amplitude*np.exp(-(c*t-x)**2/2*1)

        x = xinit
        for i in range(len(self.HtfsfLEFT)):
            self.HtfsfLEFT[i] = self.amplitude/c*np.exp(-(c*t-x)**2/(2*sigma))
        for i in range(len(self.HtfsfBOTTOM)):
            self.HtfsfTOP[i] = self.amplitude/c*np.exp(-(c*t-x)**2/(2*sigma))
            self.HtfsfBOTTOM[i] = self.amplitude/c*np.exp(-(c*t-x)**2/(2*sigma))
            x += grid.xDiscretization[i+self.tfsfleft]
        for i in range(len(self.HtfsfRIGHT)):
            self.HtfsfRIGHT[i] = self.amplitude/c*np.exp(-(c*t-x)**2/(2*sigma))
        


"define objects"  
class object:
    def __init__(self, x, y, Dx, Dy, xrefinement, yrefinement, permittivity, permeability, conductivity):
        self.x = x
        self.y = y
        self.Dx = Dx
        self.Dy = Dy
        self.xrefinement = xrefinement
        self.yrefinement = yrefinement
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity
    
    def inside(self,x,y):
        return(self.x<=x<=self.x+self.Dx and self.y<=y<=self.y+self.Dy)
        
def UserInput():
    examp = input("Run example? (Y/N):")
    if examp=='Y':
        a = object(0.04,0.04,0.02,0.02,1.50,1.50,epsilon0*5,mu0*5,0)
        g = grid(0.1,0.1,0.001,0.001)
        g.generate_grid([a])
        g.generate_cE([a],0.000000000001)
        g.generate_cH([a],0.000000000001)
        g.generate_sE([a],0.000000000001)
        field = tfsf(10,90,10,90)
        fig, ax = plt.subplots()
        fdtd = FTDT(g,[a],field,0,0.000000000001,500,ax=ax)
        fdtd.run()

        my_anim = ArtistAnimation(fig, fdtd.movie, interval=10, repeat_delay=1000,blit=True,)
        plt.show()
        return()
    print("\n=====Grid initialization=====")
    w = input('Width (m): ')
    h = input("Height (m): ")
    dx = input("dX (m): ")
    dy = input("dY (m): ")
    g = grid(100,200,0.1,0.1)
    print("\n====Object initialization====")
    objects=[]
    moreobjects = True
    moreobjects = input("Do you wish to add an object? (Y/N)")=="Y"or"y"
    while moreobjects == True:
        x = float(input("X pos (m): "))
        y = float(input("Y pos (m): "))
        width = float(input("Width (m): "))
        height = float(input("Height (m): "))
        refX = float(input("Relative refinement along x-axis (1.0 for no refinement): "))
        refY = float(input("Relative refinement along y-axis (1.0 for no refinement): "))
        eps_r = float(input("Relative permittivity: "))
        mu_r = float(input("Relative permeability: "))
        sigma = float(input("Conductivity: "))
        objects.append(object(40,90,20,20,1.2,1.3,2.5,1.3,0.0))
        moreobjects = input("\nDo you wish to add another object? (Y/N)")=="Y"or"y"
    print("\n=========TFSF spacing=========")
    spacingx = int(input("Steps of spacing along the x-axis: "))
    spacingy = int(input("Steps of spacing along the y-axis: "))
    print(spacingx,spacingy,w,h,dx,dy)
    field = tfsf(30,1000-30,60,2000-60)
    print("\n=====Time discretization=====")
    dt = input("Timestep (ns): ")
    dt = float(dt)*0.000000001
    nt = int(input("Amount of timesteps: "))
    fig, ax = plt.subplots()
    g.generate_grid(objects)
    g.generate_cE(objects,dt)
    g.generate_cH(objects,dt)
    g.generate_sE(objects,dt)
    ftdt = FTDT(g,objects,field,0,0.0000000001,500,ax=ax)
    ftdt.run()
    my_anim = ArtistAnimation(fig, fdtd.movie, interval=10, repeat_delay=1000,blit=True,)
    plt.show()

UserInput()





    



