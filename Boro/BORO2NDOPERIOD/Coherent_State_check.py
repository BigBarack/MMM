import numpy as np
import matplotlib.pyplot as plt



def main():
    # This is a placeholder for the main function.
    # You can add your code here to execute when the script is run.

    """
    Creates a harmonic potential well centered at a specific point (xc, yc) with a given radius.
    The potential increases quadratically as the distance from the center increases and updates self.V.

    :param X: x-coordinate matrix (e.g., self.Xc)
    :param Y: y-coordinate matrix (e.g., self.Yc)
    :return: Updates self.V with the potential values for the entire grid
    """

    # Define the simulation domain
    Lx, Ly = 10, 10  # Domain dimensions in cm
    Nx, Ny = 200, 200  # Number of grid points

    # Create a mesh grid for X and Y
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    m_e=9.10938356e-31  # Mass of electron in kg
    hbar=1.0545718e-34  # Reduced Planck's constant in J.s
    omega=1.0e15  # Angular frequency in rad/s
    # Initialize the potential array
    V = np.zeros((Ny, Nx))
    print("V shape:", V.shape)
    # Create a class instance to hold the domain and potential
    class Simulation:
        def __init__(self, Lx, Ly, X, Y, V):
            self.Lx = Lx
            self.Ly = Ly
            self.Xc = X
            self.Yc = Y
            self.V = V
            
            self.psi_r = np.zeros((Ny, Nx) )  # Initialize the wave function
            self.psi_i = np.zeros((Ny, Nx))  # Initialize the wave function

        def coherent_state(self, X, Y):
            """
            Creates a coherent state wave function in the form of a Gaussian wave packet.
            The wave function is centered at (xc, yc) with a given width and phase.

            :param X: x-coordinate matrix (e.g., self.Xc)
            :param Y: y-coordinate matrix (e.g., self.Yc)
            :return: Updates self.psi_r and self.psi_i with the real and imaginary parts of the wave function
            """
            # Parameters for the Gaussian wave packet
            xc, yc = 5, 5
        def V_well(self, X, Y):
            """
            Creates a harmonic potential well centered at a specific point (xc, yc) with a given radius.
            The potential increases quadratically as the distance from the center increases and updates self.V.

            :param X: x-coordinate matrix (e.g., self.Xc)
            :param Y: y-coordinate matrix (e.g., self.Yc)
            :return: Updates self.V with the potential values for the entire grid
            """
            # Center of the harmonic well
            xc, yc = 5, 5 # Centered in the simulation domain

            # Radius of the harmonic well
            radius = min(self.Lx, self.Ly) / 10  # Example: 1/10th of the smallest domain dimension

            # Spring constant for the harmonic potential
            k = 1e3  # Adjust this value as needed for the desired strength of the potential

            # Calculate the squared distance from the center
            r_squared = (X - xc) ** 2 + (Y - yc) ** 2

            # Apply the harmonic potential within the circle
            potential = np.zeros_like(self.V)
            print(r_squared <= radius ** 2)
            potential = 0.5 * k * r_squared #* np.where(r_squared <= radius ** 2, 1, 0)


            # Update self.V with the calculated potential
            self.V = potential

            
            return potential
        

        
    
        def update_wave_function(self, dt, hbar=1.0, m=1.0):
            """
            Update the wave function using the time-dependent Schrödinger equation.
            This uses a finite difference method to approximate the evolution.

                :param dt: Time step for the update
                :param hbar: Reduced Planck's constant (default: 1.0)
                :param m: Mass of the particle (default: 1.0)
                """
                # Compute the Laplacian of the wave function (finite difference approximation)
            dx = self.Lx / (self.Xc.shape[1] - 1)
            dy = self.Ly / (self.Yc.shape[0] - 1)

            laplacian_real = (
                (np.roll(self.psi_r, -1, axis=1) - 2 * self.psi_r + np.roll(self.psi_r, 1, axis=1)) / dx**2 +
                (np.roll(self.psi_r, -1, axis=0) - 2 * self.psi_r + np.roll(self.psi_r, 1, axis=0)) / dy**2
            )
            laplacian_imag = (
                (np.roll(self.psi_i, -1, axis=1) - 2 * self.psi_i + np.roll(self.psi_i, 1, axis=1)) / dx**2 +
                (np.roll(self.psi_i, -1, axis=0) - 2 * self.psi_i + np.roll(self.psi_i, 1, axis=0)) / dy**2
            )

                # Update the real and imaginary parts of the wave function
            self.psi_r += dt * (hbar / (2 * m) * laplacian_imag - self.V * self.psi_i / hbar)
            self.psi_i -= dt * (hbar / (2 * m) * laplacian_real - self.V * self.psi_r / hbar)
            # Placeholder for wave function update logic
            # You can implement your own logic here
            return None
        
        def iterate(self, dt, steps):
            """
            Iterate the wave function update for a given number of steps.

            :param dt: Time step for the update
            :param steps: Number of time steps to iterate
            """
            for _ in range(steps):
                self.update_wave_function(dt)
                # Visualize the wave function
                plt.clf()
                plt.imshow(np.abs(self.psi_r + 1j * self.psi_i), extent=(0, self.Lx, 0, self.Ly), origin='lower', cmap='viridis')
                plt.colorbar(label='Wave Function Magnitude')
                plt.title('Wave Function Evolution')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.pause(0.01)
                
                # You can add logic to visualize or save the wave function here



    # Instantiate the simulation object
    sim = Simulation(Lx, Ly, X, Y, V)
    sim.psi_r = m_e * omega / (np.pi * hbar) * np.exp(-m_e * omega / (2 * hbar) * (X**2 + Y**2))  # Initial wave function
    # Call the V_well function to calculate the potential
    sim.V_well(sim.Xc, sim.Yc)
    sim.iterate(dt=0.01, steps=100)
    pass



if __name__ == "__main__":
    main()  