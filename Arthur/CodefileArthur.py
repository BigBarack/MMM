import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import time
import scipy as sc
matplotlib.use('Qt5Agg')

"Create a "
class grid:

    def __init__(self, SizeX, SizeY, dX, dY):
        self.SizeX = SizeX
        self.SizeY = SizeY
        self.dX = dX
        self.dY = dY


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
        Bz = np.zeros((len(xDiscretization),len(yDiscretization)))
        return(Ex,Ey,Bz,xDiscretization,xDualDiscretization,yDiscretization,yDualDiscretization)

"define objects"  
class object:
    def __init__(self, x, y, Dx, Dy, xrefinement, yrefinement):
        self.x = x
        self.y = y
        self.Dx = Dx
        self.Dy = Dy
        self.xrefinement = xrefinement
        self.yrefinement = yrefinement

a = object(5,5,2,3,2.0,2.0)
b = object(1,1,1.2,1.2,1.5,3.0)
g = grid(10,10,0.1,0.1)

u,u,u,x,u,y,u = grid.generate_grid(g,[a,b])
fig, ax = plt.subplots()

xpos = 0
ypos = 0
for i in x:
    xpos += i
    ax.axvline(x=xpos)
for i in y:
    ypos += i
    ax.axhline(y=ypos)
plt.show()    

        

