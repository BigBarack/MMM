import numpy as np
import matplotlib.pyplot as plt

Lx=5
Ly=5
N= 100
percentageold= 0.5
percentage= 1/2-percentageold/2
# Define the 2D Heaviside function
def heaviside_2d(x, y):

    H1 = np.heaviside( (1/2-percentage)*Lx - x, 1 ) * np.heaviside( ( 1/2-percentage )* Ly-x, 1)

    H2 = np.heaviside((1/2-percentage)*Lx-y, 1) * np.heaviside((1/2-percentage)*Ly-y, 1)

    H3 = np.heaviside(-(1/2+percentage)*Lx+x, 1) * np.heaviside(-(1/2+percentage)*Ly+x, 1)

    H4 = np.heaviside(-(1/2+percentage)*Lx+y, 1) * np.heaviside(-(1/2+percentage)*Ly+y, 1)

    combined = np.logical_or.reduce([H1, H2, H3, H4])
    return combined

# Create a grid
x = np.linspace(0, Lx,N)
y = np.linspace(0, Ly, N)
X, Y = np.meshgrid(x, y)

# Apply the Heaviside function     
H = heaviside_2d(X, Y)

# Plot the result
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, H, levels=1, colors=['lightgray', 'blue'])
plt.title("2D Heaviside Function")
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.colorbar(label='H(x, y)')
plt.tight_layout()
plt.show()
