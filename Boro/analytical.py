import numpy as np
from scipy.special import hankel2
from scipy.special import jv


#For the analytical solution what are the parameters needed and at what point will we observe
#it doesnt seem to be time dependant
#it is dependant on pho which has a 
ka = 1.0
k  = 1.0 #wavevector
a  = 1.0 #Radius of cylinder
ka = k*a
phi =  np.pi/4
rho = 2.0  # Observation point
A= 1        # amplitude
N_max = int(np.ceil(ka + 10))  #amount of summations we area going to do

Ez_scat = 0.0
def nu(n):
    if n == 0:
          return 1
    return 2


for n in range(0, 2*N_max+1): # over around 
    
        an=(-1j)**nu(n)
        term=A*(  jv(n,k * rho)-  jv(n,ka)/hankel2(n,ka) * hankel2(n , k * rho) ) * np.cos( n * phi)
        termcomp = an * term
        Ez_scat += term

print("scattered Ez:", Ez_scat)