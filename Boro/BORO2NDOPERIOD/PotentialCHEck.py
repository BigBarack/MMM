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
            potential = 0.5 * k * r_squared * np.where(r_squared <= radius ** 2, 1, 0)


            # Update self.V with the calculated potential
            self.V = potential

            print()
            fig, ax = plt.subplots(figsize=(8, 8))
            contour = ax.contourf(self.Xc, self.Yc, self.V, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax, label="Potential")
            ax.set_title("Harmonic Well")
            ax.set_xlabel("x [cm]")
            ax.set_ylabel("y [cm]")
            plt.show()
            return potential



    # Instantiate the simulation object
    sim = Simulation(Lx, Ly, X, Y, V)

    # Call the V_well function to calculate the potential
    sim.V_well(sim.Xc, sim.Yc)
  
    pass



if __name__ == "__main__":
    main()  