import numpy as np
from scipy.special import hankel2
from scipy.special import jv
import matplotlib.pyplot as plt

# Parameters
a = 1.0  # Radius of cylinder
phi = np.pi / 4  # Angle
rho = 2.0  # Observation point
A = 1  # Amplitude

# Frequency range (wavevector k)
k_values = np.linspace(0.1, 10, 500)  # Avoid k=0 to prevent division by zero
Hz_scat_values = []

# Function to compute nu(n)
def nu(n):
    if n == 0:
        return 1
    return 2

# Compute Hz_scat for each k
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
plt.figure(figsize=(10, 6))
plt.plot(k_values, Hz_scat_values, label="|Hz_scat|")
plt.title("Frequency Response of Hz_scat")
plt.xlabel("Wavevector k (proportional to frequency)")
plt.ylabel("|Hz_scat|")
plt.grid(True)
plt.legend()
plt.show()

