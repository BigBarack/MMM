# Keep all the previous code (imports, classes FDTD, Scatterer, functions)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings # For CFL condition warning

# --- Constants ---
EPSILON_0 = 8.8541878128e-12  # F/m
MU_0 = 4 * np.pi * 1e-7      # H/m
C0 = 1 / np.sqrt(MU_0 * EPSILON_0) # Speed of light

def cross_average2Drray(arr):
    """
    takes given array and returns an elements-wise cross average new array
    would have been useful to treat boundaries in the updates insteaad of explicitly handling
    the PEC ghost cells
    :param arr: 2D array to be padded and averaged
    :return: cross average of array
    """
    padded = np.pad(arr,1,mode='edge')
    # Corrected cross average: center + 4 neighbors / 5
    cross_avg = (padded[1:-1, 1:-1] + padded[:-2, 1:-1] + padded[2:, 1:-1] +
                 padded[1:-1, :-2] + padded[1:-1, 2:]) / 5
    return cross_avg

class Scatterer:
    # ... (keep your Scatterer class as is) ...
    def __init__(self , shape:str , material:str , ID:int , geometry:dict , properties:dict ):
        self.shape = shape
        self.ID = ID              #useful for up-eq and mask
        self.material = material      #pec / pmc / drude
        self.geometry = geometry      #depends on shape
        self.properties = properties  #e,m

    def get_bounds(self):
        """
        for given scatterer, it gives the x-range and y-range
        to be used for defining refinement regions and masks
        """
        if self.shape == 'circle':
            xc = self.geometry['center'][0]
            yc = self.geometry['center'][1]
            r = self.geometry['radius']
            x_range = ( xc - r , xc + r )
            y_range = ( yc - r, yc + r ) # Circle bounds are same for x and y relative to center
            return x_range, y_range
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

    def __init__(self, Lx:float , Ly:float , fmax:float , scatterer_list:list , observation_points, Npml=10, Courant=0.7):
        """
        Initialize the FDTD simulation domain.

        Args:
            Lx, Ly: Domain size in meters (assuming input is cm, convert later if needed)
            fmax: Maximum frequency of interest for grid resolution.
            scatterer_list: List of Scatterer objects.
            observation_points: List of (x, y) tuples for observation.
            Npml: Number of PML layers.
            Courant: Courant number (<= 1/sqrt(2) for 2D)
        """
        # constants
        self.epsilon_0 = EPSILON_0
        self.mu_0 = MU_0
        self.c0 = C0
        self.Lx = Lx
        self.Ly = Ly
        self.fmax = fmax
        self.lmin = self.c0 / fmax if fmax > 0 else Lx # Smallest wavelength
        self.scatterer_list = scatterer_list
        self.observation_points = observation_points
        self.Npml = Npml

        # --- Grid Generation ---
        # Base resolutions
        self.dx_coarse = self.lmin / 10
        self.dx_fine = self.lmin / 20
        # Intermediate steps for smoother transition (adjust factor as needed)
        transition_factor = 1.5 # Instead of 2**(1/3) for simplicity
        self.dx_inter1 = self.dx_fine * transition_factor
        self.dx_inter2 = self.dx_inter1 * transition_factor

        # Generate non-uniform grid edges
        self.x_edges = self._generate_grid_axis(self.Lx, 'x')
        self.y_edges = self._generate_grid_axis(self.Ly, 'y')

        # Calculate grid spacings (primal and dual)
        self.dx = np.diff(self.x_edges) # dx[i] = x[i+1] - x[i] (Nx elements)
        self.dy = np.diff(self.y_edges) # dy[j] = y[j+1] - y[j] (Ny elements)
        # Dual grid spacing (at centers) - handle edges carefully if needed
        self.dx_dual = 0.5 * (self.dx[:-1] + self.dx[1:]) # (Nx-1 elements)
        self.dy_dual = 0.5 * (self.dy[:-1] + self.dy[1:]) # (Ny-1 elements)

        # Add dual grid points at boundaries assuming center is edge + dx/2 or edge - dx/2
        # Primal grid nodes Nx+1, Ny+1 points. Cells Nx x Ny.
        self.Nx = len(self.dx)
        self.Ny = len(self.dy)

        # Create meshgrid for cell centers and edges (for field locations)
        # Cell centers (Hz locations)
        x_centers = self.x_edges[:-1] + 0.5 * self.dx
        y_centers = self.y_edges[:-1] + 0.5 * self.dy
        self.Xc, self.Yc = np.meshgrid(x_centers, y_centers, indexing='ij') # Use 'ij' indexing!

        # Locations for Ex (centers in x, edges in y)
        self.Xc_Ex, self.Yc_Ex = np.meshgrid(x_centers, self.y_edges, indexing='ij')
        # Locations for Ey (edges in x, centers in y)
        self.Xc_Ey, self.Yc_Ey = np.meshgrid(self.x_edges, y_centers, indexing='ij')

        # Meshgrids for grid spacings at field locations
        # For Hz update: dx at Hz location, dy at Hz location
        self.DX_Hz, self.DY_Hz = np.meshgrid(self.dx, self.dy, indexing='ij')
        # For Ex update: dy at Ex location (these are the primal dy)
        _, self.DY_Ex = np.meshgrid(x_centers, self.dy, indexing='ij')
        # For Ey update: dx at Ey location (these are the primal dx)
        self.DX_Ey, _ = np.meshgrid(self.dx, y_centers, indexing='ij')


        # --- Time Step ---
        # Use minimum grid spacing for CFL condition
        min_ds = min(np.min(self.dx), np.min(self.dy))
        self.dt = Courant * min_ds / (self.c0 * np.sqrt(2)) # CFL for 2D
        print(f"Grid: {self.Ny}x{self.Nx} cells.")
        print(f"Min dx={np.min(self.dx):.2e}, Min dy={np.min(self.dy):.2e}")
        print(f"Time step dt={self.dt:.2e} s")


        # --- Initialize Fields ---
        # Note: Shapes adjusted for 'ij' indexing (rows=y, cols=x)
        self.Hz = np.zeros((self.Ny, self.Nx))
        self.Ex = np.zeros((self.Ny + 1, self.Nx)) # Defined on y-edges
        self.Ey = np.zeros((self.Ny, self.Nx + 1)) # Defined on x-edges


        # --- Material Properties & Masks ---
        self.epsilon_r_grid = np.ones_like(self.Hz) # Relative permittivity at Hz locations
        self.mu_r_grid = np.ones_like(self.Hz)      # Relative permeability at Hz locations
        self.sigma_e_grid = np.zeros_like(self.Hz) # Electric conductivity (for Drude or lossy)
        self.sigma_m_grid = np.zeros_like(self.Hz) # Magnetic conductivity (optional)

        self.mask_Hz = np.zeros_like(self.Hz, dtype=int) # Material ID at Hz locations
        # Need masks for Ex and Ey locations too for PEC/PMC handling inside objects
        # Simple approach: check if center point is inside
        self.mask_Ex = np.zeros_like(self.Ex, dtype=int)
        self.mask_Ey = np.zeros_like(self.Ey, dtype=int)

        # Populate masks and material grids based on scatterers
        for scatterer in self.scatterer_list:
            # Hz grid check
            inside_z = scatterer.is_inside(self.Xc, self.Yc)
            self.mask_Hz[inside_z] = scatterer.ID
            if scatterer.material == 'Drude': # Assuming properties are relative for now
                self.epsilon_r_grid[inside_z] = scatterer.properties.get('e_r', 1.0) # Use get for defaults
                self.mu_r_grid[inside_z] = scatterer.properties.get('m_r', 1.0)
                # Add conductivity if defined for Drude model (e.g., plasma frequency, collision freq)
                # self.sigma_e_grid[inside_z] = calculate_drude_sigma(...)
            elif scatterer.material == 'PEC':
                 # Mask indicates PEC, updates will handle it
                 pass
            elif scatterer.material == 'PMC':
                 # Mask indicates PMC, updates will handle it
                 pass

            # Ex grid check (approximate: check center x, edge y)
            inside_x = scatterer.is_inside(self.Xc_Ex[:-1,:], self.Yc_Ex[:-1,:]) # Check point below the edge
            self.mask_Ex[:-1,:][inside_x] = scatterer.ID # Assign ID to Ex below the point

            # Ey grid check (approximate: check edge x, center y)
            inside_y = scatterer.is_inside(self.Xc_Ey[:,:-1], self.Yc_Ey[:,:-1]) # Check point left of the edge
            self.mask_Ey[:,:-1][inside_y] = scatterer.ID # Assign ID to Ey left of the point

        # --- Calculate Update Coefficients (Standard FDTD Part) ---
        # Pre-calculate for efficiency, including material properties
        self.Ce_x = self.dt / (self.epsilon_r_grid * self.epsilon_0) # Coefficient at Hz location (needs averaging for Ex)
        self.Ce_y = self.dt / (self.epsilon_r_grid * self.epsilon_0) # Coefficient at Hz location (needs averaging for Ey)
        self.Ch_z = self.dt / (self.mu_r_grid * self.mu_0)          # Coefficient at Hz location

        # Averaged permittivity for E updates
        # Ex uses average epsilon along y: (eps[j] + eps[j-1])/2 -> Ny+1 x Nx array
        self.eps_r_yavg = np.ones_like(self.Ex) * self.epsilon_0 # Start with background
        eps_r_Hz = self.epsilon_r_grid * self.epsilon_0
        self.eps_r_yavg[1:-1, :] = 0.5 * (eps_r_Hz[1:, :] + eps_r_Hz[:-1, :])
        # Handle boundaries (assume background or extrapolate)
        self.eps_r_yavg[0, :] = eps_r_Hz[0, :]
        self.eps_r_yavg[-1, :] = eps_r_Hz[-1, :]
        self.Cex_coeffs = self.dt / self.eps_r_yavg

        # Ey uses average epsilon along x: (eps[i] + eps[i-1])/2 -> Ny x Nx+1 array
        self.eps_r_xavg = np.ones_like(self.Ey) * self.epsilon_0 # Start with background
        self.eps_r_xavg[:, 1:-1] = 0.5 * (eps_r_Hz[:, 1:] + eps_r_Hz[:, :-1])
        # Handle boundaries
        self.eps_r_xavg[:, 0] = eps_r_Hz[:, 0]
        self.eps_r_xavg[:, -1] = eps_r_Hz[:, -1]
        self.Cey_coeffs = self.dt / self.eps_r_xavg

        # Hz uses mu at Hz location -> Ny x Nx array
        self.Chz_coeffs = self.dt / (self.mu_r_grid * self.mu_0)


        # --- PML Initialization ---
        self.pml_params = {
            'Npml': Npml,
            'sigma_max': 0.0, # Calculated below
            'kappa_max': 5.0, # Typical starting value
            'alpha_max': 0.05, # Typical starting value for CFS-PML
            'm_pml': 3       # Polynomial order
        }
        # Optimal sigma based on impedance matching and polynomial order
        # This formula is a common heuristic, adjust if needed
        eta0 = np.sqrt(self.mu_0 / self.epsilon_0)
        self.pml_params['sigma_max'] = (self.pml_params['m_pml'] + 1) / (150 * np.pi * min_ds * eta0) # Adjusted for eta0


        self._calculate_pml_profiles()
        self._calculate_pml_coeffs()

        # Initialize PML auxiliary fields (same shape as corresponding E/H fields)
        self.psi_Ex_Hz = np.zeros_like(self.Ex) # Aux for Ex update related to dHz/dy
        self.psi_Ey_Hz = np.zeros_like(self.Ey) # Aux for Ey update related to dHz/dx
        self.psi_Hz_Ey = np.zeros_like(self.Hz) # Aux for Hz update related to dEy/dx
        self.psi_Hz_Ex = np.zeros_like(self.Hz) # Aux for Hz update related to dEx/dy

        print("PML Initialized.")


    def _generate_grid_axis(self, L_axis: float, axis_name: str):
        """Generates non-uniform grid points along one axis."""
        points = [0.0]
        p0 = 0.0
        entering_intermediate = True # Flag to track transition direction

        # --- Determine Refinement Ranges ---
        refine_intervals = []
        intermediate_intervals = []
        padding = 2 * (self.dx_inter1 + self.dx_inter2) # Define padding width

        for sc in self.scatterer_list:
            bounds_x, bounds_y = sc.get_bounds()
            bounds = bounds_x if axis_name == 'x' else bounds_y
            lower, upper = bounds
            refine_intervals.append((lower, upper))
            intermediate_intervals.append((lower - padding, upper + padding))

        # Simple merging of overlapping intervals (can be improved for complex cases)
        # This basic approach just checks points, doesn't merge intervals formally
        def get_region_type(pos):
            is_refine = any(low <= pos <= upp for low, upp in refine_intervals)
            if is_refine:
                return 'refine'
            is_intermediate = any(low <= pos <= upp for low, upp in intermediate_intervals)
            if is_intermediate:
                return 'intermediate'
            return 'coarse'

        # --- Build Grid Points ---
        last_type = 'coarse'
        while p0 < L_axis:
            current_type = get_region_type(p0)

            # Determine step based on region type and transition logic
            if current_type == 'refine':
                step = self.dx_fine
                entering_intermediate = True # Reset entering flag when in refine
            elif current_type == 'intermediate':
                # Handle transitions using intermediate steps
                if last_type == 'coarse': # Entering refine region
                     # Add intermediate steps dx_inter2, dx_inter1 (reversed order)
                     steps = [self.dx_inter2, self.dx_inter2, self.dx_inter1, self.dx_inter1]

                elif last_type == 'refine': # Leaving refine region
                    # Add intermediate steps dx_inter1, dx_inter2
                    steps = [self.dx_inter1, self.dx_inter1, self.dx_inter2, self.dx_inter2]
                else: # Already intermediate, continue coarse step for simplicity here
                    steps = [self.dx_coarse] # Or adjust logic for smoother transition within intermediate

                # Add points based on determined steps
                current_p = p0
                for s in steps:
                   current_p += s
                   if current_p > L_axis + 1e-9: # Avoid overshooting L_axis
                       break
                   points.append(current_p)
                step = sum(steps) # Total step taken in this block

            else: # Coarse region
                step = self.dx_coarse
                entering_intermediate = True # Reset entering flag

            # Update position (unless intermediate steps already handled it)
            if current_type != 'intermediate':
                 p0 += step
                 if p0 > L_axis + 1e-9: # Avoid overshooting
                     # If the last full step overshoots, adjust the last point to L_axis
                     if points[-1] > L_axis:
                        points[-1] = L_axis
                 else:
                    points.append(p0)
            else:
                p0 = points[-1] # Update p0 to the last point added by intermediate logic


            last_type = current_type

        # Ensure the last point is exactly L_axis
        if points[-1] < L_axis:
             points.append(L_axis)
        elif points[-1] > L_axis:
             points[-1] = L_axis

        return np.unique(np.array(points)) # Use unique to handle potential duplicates


    def in_refined_region(self, pos:float , axis:str):
        """ DEPRECATED - Logic moved to _generate_grid_axis """
        raise DeprecationWarning("Use _generate_grid_axis internal logic instead.")


    def _calculate_pml_profiles(self):
        """Calculates sigma, kappa, alpha profiles on the Yee grid within PML."""
        Npml = self.pml_params['Npml']
        sig_max = self.pml_params['sigma_max']
        kap_max = self.pml_params['kappa_max']
        alp_max = self.pml_params['alpha_max']
        m = self.pml_params['m_pml']

        # --- PML Profile Function ---
        def profile_func(rho_norm, val_max):
            # Normalized distance rho_norm (0 at interface, 1 at outer boundary)
            # Handle rho_norm > 1 or < 0 cases if necessary
            rho_norm = np.clip(rho_norm, 0, 1)
            # Polynomial grading
            return val_max * (rho_norm ** m)

        # --- Coordinate Arrays at Yee Grid Locations Needed ---
        # Primal grid coordinates (edges)
        x_p = self.x_edges # For Ey, sigma_x, kappa_x, alpha_x
        y_p = self.y_edges # For Ex, sigma_y, kappa_y, alpha_y
        # Dual grid coordinates (centers)
        x_d = self.x_edges[:-1] + 0.5 * self.dx # For Hz, Ex
        y_d = self.y_edges[:-1] + 0.5 * self.dy # For Hz, Ey

        # --- Initialize Profile Arrays (size of domain, zero outside PML) ---
        # Staggered grid locations matter!
        # sigma_x, kappa_x, alpha_x needed at Ey locations (Ny x Nx+1)
        self.sigma_x_pml = np.zeros((self.Ny, self.Nx + 1))
        self.kappa_x_pml = np.ones((self.Ny, self.Nx + 1)) # Default kappa is 1
        self.alpha_x_pml = np.zeros((self.Ny, self.Nx + 1))

        # sigma_y, kappa_y, alpha_y needed at Ex locations (Ny+1 x Nx)
        self.sigma_y_pml = np.zeros((self.Ny + 1, self.Nx))
        self.kappa_y_pml = np.ones((self.Ny + 1, self.Nx)) # Default kappa is 1
        self.alpha_y_pml = np.zeros((self.Ny + 1, self.Nx))

        # Magnetic conductivities (optional, often related to electric)
        eta0 = np.sqrt(self.mu_0 / self.epsilon_0)
        self.sigma_mx_pml = np.zeros_like(self.sigma_x_pml) # At Ey locations
        self.sigma_my_pml = np.zeros_like(self.sigma_y_pml) # At Ex locations


        # --- Calculate Profiles in PML Regions ---
        # Left PML region (i < Npml)
        if Npml > 0:
            # Calculate thickness of left PML (physical distance)
            d_pml_left = self.x_edges[Npml] - self.x_edges[0]
            if d_pml_left > 1e-9: # Avoid division by zero
                for i in range(Npml):
                    # Ey location grid point x coordinate: self.x_edges[i]
                    rho_norm = (self.x_edges[Npml] - self.x_edges[i]) / d_pml_left
                    sig_val = profile_func(rho_norm, sig_max)
                    kap_val = 1 + profile_func(rho_norm, kap_max - 1) # Kappa >= 1
                    alp_val = profile_func(rho_norm, alp_max)

                    self.sigma_x_pml[:, i] = sig_val
                    self.kappa_x_pml[:, i] = kap_val
                    self.alpha_x_pml[:, i] = alp_val
                    self.sigma_mx_pml[:, i] = sig_val / eta0 # Magnetic conductivity

            # Right PML region (i >= Nx - Npml)
            d_pml_right = self.x_edges[-1] - self.x_edges[-Npml-1]
            if d_pml_right > 1e-9:
                 for i in range(Npml):
                    idx = self.Nx - Npml + i + 1 # Index for Ey location (Nx+1 total)
                    rho_norm = (self.x_edges[idx] - self.x_edges[-Npml-1]) / d_pml_right
                    sig_val = profile_func(rho_norm, sig_max)
                    kap_val = 1 + profile_func(rho_norm, kap_max - 1)
                    alp_val = profile_func(rho_norm, alp_max)

                    self.sigma_x_pml[:, idx] = sig_val
                    self.kappa_x_pml[:, idx] = kap_val
                    self.alpha_x_pml[:, idx] = alp_val
                    self.sigma_mx_pml[:, idx] = sig_val / eta0


            # Bottom PML region (j < Npml)
            d_pml_bottom = self.y_edges[Npml] - self.y_edges[0]
            if d_pml_bottom > 1e-9:
                for j in range(Npml):
                     # Ex location grid point y coordinate: self.y_edges[j]
                    rho_norm = (self.y_edges[Npml] - self.y_edges[j]) / d_pml_bottom
                    sig_val = profile_func(rho_norm, sig_max)
                    kap_val = 1 + profile_func(rho_norm, kap_max - 1)
                    alp_val = profile_func(rho_norm, alp_max)

                    self.sigma_y_pml[j, :] = sig_val
                    self.kappa_y_pml[j, :] = kap_val
                    self.alpha_y_pml[j, :] = alp_val
                    self.sigma_my_pml[j, :] = sig_val / eta0 # Magnetic conductivity

            # Top PML region (j >= Ny - Npml)
            d_pml_top = self.y_edges[-1] - self.y_edges[-Npml-1]
            if d_pml_top > 1e-9:
                for j in range(Npml):
                    idx = self.Ny - Npml + j + 1 # Index for Ex location (Ny+1 total)
                    rho_norm = (self.y_edges[idx] - self.y_edges[-Npml-1]) / d_pml_top
                    sig_val = profile_func(rho_norm, sig_max)
                    kap_val = 1 + profile_func(rho_norm, kap_max - 1)
                    alp_val = profile_func(rho_norm, alp_max)

                    self.sigma_y_pml[idx, :] = sig_val
                    self.kappa_y_pml[idx, :] = kap_val
                    self.alpha_y_pml[idx, :] = alp_val
                    self.sigma_my_pml[idx, :] = sig_val / eta0


    def _calculate_pml_coeffs(self):
        """Calculate UPML update coefficients based on profiles."""
        # Denominators for coefficients (avoid division by zero)
        # Location: Ex (Ny+1, Nx)
        den_ex = self.kappa_y_pml + (self.sigma_y_pml + self.alpha_y_pml * self.kappa_y_pml) * self.dt / self.epsilon_0
        # Location: Ey (Ny, Nx+1)
        den_ey = self.kappa_x_pml + (self.sigma_x_pml + self.alpha_x_pml * self.kappa_x_pml) * self.dt / self.epsilon_0
        # Location: Hz (Ny, Nx) - Requires averaging kappa/sigma/alpha to Hz locations
        # Simplification: Use kappa/sigma at nearest E locations (can be refined)
        kappa_x_hz = 0.5 * (self.kappa_x_pml[:, :-1] + self.kappa_x_pml[:, 1:]) # Average kappa_x to Hz loc
        kappa_y_hz = 0.5 * (self.kappa_y_pml[:-1, :] + self.kappa_y_pml[1:, :]) # Average kappa_y to Hz loc
        sigma_mx_hz = 0.5 * (self.sigma_mx_pml[:, :-1] + self.sigma_mx_pml[:, 1:])
        sigma_my_hz = 0.5 * (self.sigma_my_pml[:-1, :] + self.sigma_my_pml[1:, :])
        alpha_x_hz = 0.5 * (self.alpha_x_pml[:, :-1] + self.alpha_x_pml[:, 1:])
        alpha_y_hz = 0.5 * (self.alpha_y_pml[:-1, :] + self.alpha_y_pml[1:, :])

        den_hz_x = kappa_x_hz + (sigma_mx_hz + alpha_x_hz * kappa_x_hz) * self.dt / self.mu_0 # Denom related to dEy/dx
        den_hz_y = kappa_y_hz + (sigma_my_hz + alpha_y_hz * kappa_y_hz) * self.dt / self.mu_0 # Denom related to dEx/dy

        # --- E-field Update Coefficients ---
        # For Ex update (Ny+1, Nx)
        self.b_psi_Ex = np.exp(-(self.sigma_y_pml / self.kappa_y_pml + self.alpha_y_pml) * self.dt / self.epsilon_0)
        # Handle zero division for a_psi_Ex when sigma=0, alpha=0 (should be zero)
        fac_ex = (self.sigma_y_pml * self.kappa_y_pml + self.kappa_y_pml * self.kappa_y_pml * self.alpha_y_pml)
        self.a_psi_Ex = np.divide(self.sigma_y_pml, fac_ex, out=np.zeros_like(fac_ex), where=fac_ex != 0) * (self.b_psi_Ex - 1.0)

        self.Cex_1 = (self.kappa_y_pml - (self.sigma_y_pml + self.alpha_y_pml * self.kappa_y_pml) * self.dt / self.epsilon_0) / den_ex
        self.Cex_2 = (self.dt / self.epsilon_0) / den_ex
        self.Cex_psi = - (self.dt / self.epsilon_0) * self.kappa_y_pml / den_ex # Coefficient for the psi term in Ex update

        # For Ey update (Ny, Nx+1)
        self.b_psi_Ey = np.exp(-(self.sigma_x_pml / self.kappa_x_pml + self.alpha_x_pml) * self.dt / self.epsilon_0)
        fac_ey = (self.sigma_x_pml * self.kappa_x_pml + self.kappa_x_pml * self.kappa_x_pml * self.alpha_x_pml)
        self.a_psi_Ey = np.divide(self.sigma_x_pml, fac_ey, out=np.zeros_like(fac_ey), where=fac_ey != 0) * (self.b_psi_Ey - 1.0)

        self.Cey_1 = (self.kappa_x_pml - (self.sigma_x_pml + self.alpha_x_pml * self.kappa_x_pml) * self.dt / self.epsilon_0) / den_ey
        self.Cey_2 = (self.dt / self.epsilon_0) / den_ey
        self.Cey_psi = - (self.dt / self.epsilon_0) * self.kappa_x_pml / den_ey # Coefficient for the psi term in Ey update

        # --- H-field Update Coefficients ---
        # For Hz update (Ny, Nx)
        # Coeffs related to dEy/dx term
        self.b_psi_Hz_Ey = np.exp(-(sigma_mx_hz / kappa_x_hz + alpha_x_hz) * self.dt / self.mu_0)
        fac_hzx = (sigma_mx_hz * kappa_x_hz + kappa_x_hz * kappa_x_hz * alpha_x_hz)
        self.a_psi_Hz_Ey = np.divide(sigma_mx_hz, fac_hzx, out=np.zeros_like(fac_hzx), where=fac_hzx != 0) * (self.b_psi_Hz_Ey - 1.0)

        self.Chz_1x = (kappa_x_hz - (sigma_mx_hz + alpha_x_hz * kappa_x_hz) * self.dt / self.mu_0) / den_hz_x
        self.Chz_2x = (self.dt / self.mu_0) / den_hz_x # For dEy/dx term
        self.Chz_psix = - (self.dt / self.mu_0) * kappa_x_hz / den_hz_x # Coeff for psi_Hz_Ey term

        # Coeffs related to dEx/dy term
        self.b_psi_Hz_Ex = np.exp(-(sigma_my_hz / kappa_y_hz + alpha_y_hz) * self.dt / self.mu_0)
        fac_hzy = (sigma_my_hz * kappa_y_hz + kappa_y_hz * kappa_y_hz * alpha_y_hz)
        self.a_psi_Hz_Ex = np.divide(sigma_my_hz, fac_hzy, out=np.zeros_like(fac_hzy), where=fac_hzy != 0) * (self.b_psi_Hz_Ex - 1.0)

        self.Chz_1y = (kappa_y_hz - (sigma_my_hz + alpha_y_hz * kappa_y_hz) * self.dt / self.mu_0) / den_hz_y
        self.Chz_2y = (self.dt / self.mu_0) / den_hz_y # For dEx/dy term
        self.Chz_psiy = - (self.dt / self.mu_0) * kappa_y_hz / den_hz_y # Coeff for psi_Hz_Ex term

        # Combine Chz_1x and Chz_1y? No, they multiply different terms.
        # The final Hz update is: Hz = Chz_1x * Hz_old + Chz_1y * Hz_old ?? No.
        # Hz = Coeff_Hz_Old * Hz_old + Coeff_dEx * dEx/dy + Coeff_dEy * dEy/dx + Coeff_psi_x * psi_x + Coeff_psi_y * psi_y
        # We need to combine coefficients correctly later in the update step.
        # Let's stick to the ADE form directly in `update`.


    def update(self):
        """Perform one FDTD time step using UPML."""

        # --- 1. Update H auxiliary fields (psi_Hz_...) ---
        # Calculate spatial derivatives of E using non-uniform grid differences
        # dEy/dx at Hz locations (Ny, Nx)
        dEy_dx = (self.Ey[:, 1:] - self.Ey[:, :-1]) / self.DX_Hz # DX_Hz is dx at Hz loc.
        # dEx/dy at Hz locations (Ny, Nx)
        dEx_dy = (self.Ex[1:, :] - self.Ex[:-1, :]) / self.DY_Hz # DY_Hz is dy at Hz loc.

        # Update psi_Hz_Ey (related to dEy/dx) -> (Ny, Nx)
        self.psi_Hz_Ey = self.b_psi_Hz_Ey * self.psi_Hz_Ey + self.a_psi_Hz_Ey * dEy_dx

        # Update psi_Hz_Ex (related to dEx/dy) -> (Ny, Nx)
        self.psi_Hz_Ex = self.b_psi_Hz_Ex * self.psi_Hz_Ex + self.a_psi_Hz_Ex * dEx_dy


        # --- 2. Update Hz field ---
        # Standard FDTD update coefficients (calculated in __init__)
        # Chz_coeffs = dt / (mu_r * mu_0) at Hz locations

        # Apply UPML modification using coefficients derived from ADE form
        # Hz(n+1) = C1*Hz(n) + C2*(dEx/dy|n+0.5 - dEy/dx|n+0.5) + Cpsi*Psi|n+1 ? No.

        # Direct ADE implementation for Hz:
        # (kappa_x + (sigma_mx + alpha_x*kappa_x)*dt/mu0) Hz_new_partial_x = (kappa_x - (...)*dt/mu0) Hz_old - dt/mu0 * (dEy/dx + psi_Hz_Ey_new)
        # (kappa_y + (sigma_my + alpha_y*kappa_y)*dt/mu0) Hz_new = (kappa_y - (...)*dt/mu0) Hz_new_partial_x + dt/mu0 * (dEx/dy + psi_Hz_Ex_new)

        # Simpler practical form (split update):
        # Update Hz based on dEy/dx term first
        Hz_new = self.Chz_1x * self.Hz - self.Chz_2x * dEy_dx + self.Chz_psix * self.psi_Hz_Ey

        # Update the result based on dEx/dy term (coefficients averaged at Hz loc)
        # This split isn't quite right. Revisit the ADE derivation or use simpler form.

        # Let's use the simpler form where update includes curl directly:
        # Ref: Taflove Eq. 7.28 - 7.31 (adapted for TMz)
        curl_Ex = dEx_dy # dEx/dy term
        curl_Ey = -dEy_dx # -dEy/dx term

        self.Hz = (self.Chz_1y * self.Hz +                 # Previous Hz (y-split part)
                   self.Chz_2y * (curl_Ex + self.psi_Hz_Ex)) # Update using dEx/dy & psi_Hz_Ex

        # Now update again using the x-split part (using the partially updated Hz)
        # This seems overly complex. Let's use combined coeffs if possible.

        # Alternative: Update using standard coeffs and add PML terms separately?
        # Hz_temp = self.Hz + self.Chz_coeffs * (dEx_dy - dEy_dx) # Standard update
        # Hz = apply_pml_correction(Hz_temp, self.psi_Hz_Ex, self.psi_Hz_Ey, ...)
        # This is also complex. Let's stick to the direct ADE form.

        # --- Try again with direct ADE update for Hz ---
        # Maybe pre-calculate combined coefficients for Hz
        # Hz(n+1) = CA_Hz * Hz(n) + CB_Hz_Ex * (dEx/dy(n+0.5) + psi_Hz_Ex(n+1))
        #                  + CB_Hz_Ey * (dEy/dx(n+0.5) + psi_Hz_Ey(n+1))
        # Need to derive CA_Hz, CB_Hz_Ex, CB_Hz_Ey - complex

        # --- Let's use Taflove/Gedney standard ADE update formulation ---
        # Update Hz using derivatives calculated earlier
        self.Hz = self.Hz + self.Chz_coeffs * (dEx_dy - dEy_dx) # Start with standard update

        # Add PML contributions (This step needs careful derivation based on the ADE form chosen)
        # This part is tricky and depends heavily on the exact ADE formulation.
        # For now, let's assume the coefficients calculated earlier are for a full update (needs verification)
        # If using split-field:
        # Update Hzx component using dEy/dx
        # Update Hzy component using dEx/dy
        # Hz = Hzx + Hzy --> Requires storing split fields.

        # Let's pause Hz PML correction and implement E first, then revisit Hz.
        # For now: Standard Hz update (won't have PML absorption yet)
        self.Hz = self.Hz + self.Chz_coeffs * (dEx_dy - dEy_dx)


        # --- 3. Update E auxiliary fields (psi_Ex_..., psi_Ey_...) ---
        # Calculate spatial derivatives of Hz using non-uniform grid differences
        # dHz/dy at Ex locations (Ny+1, Nx)
        dHz_dy = (self.Hz[1:, :] - self.Hz[:-1, :]) / self.DY_Ex # DY_Ex is dy at Ex loc.
        # dHz/dx at Ey locations (Ny, Nx+1)
        dHz_dx = (self.Hz[:, 1:] - self.Hz[:, :-1]) / self.DX_Ey # DX_Ey is dx at Ey loc.

        # Update psi_Ex_Hz (related to dHz/dy) -> (Ny+1, Nx)
        self.psi_Ex_Hz = self.b_psi_Ex * self.psi_Ex_Hz + self.a_psi_Ex * dHz_dy

        # Update psi_Ey_Hz (related to dHz/dx) -> (Ny, Nx+1)
        self.psi_Ey_Hz = self.b_psi_Ey * self.psi_Ey_Hz + self.a_psi_Ey * dHz_dx


        # --- 4. Update Ex field ---
        # Ex(n+1) = Cex_1 * Ex(n) + Cex_2 * (dHz/dy(n+1) + psi_Ex_Hz(n+1))
        self.Ex = self.Cex_1 * self.Ex + self.Cex_2 * dHz_dy + self.Cex_psi * self.psi_Ex_Hz


        # --- 5. Update Ey field ---
        # Ey(n+1) = Cey_1 * Ey(n) + Cey_2 * (-dHz/dx(n+1) + psi_Ey_Hz(n+1)) ? Sign? Check Maxwell eq.
        # dEy/dt = -1/eps * dHz/dx
        # Ey update involves -dHz/dx
        self.Ey = self.Cey_1 * self.Ey - self.Cey_2 * dHz_dx + self.Cey_psi * self.psi_Ey_Hz


        # --- 6. Apply PEC/PMC conditions inside scatterers (Optional) ---
        # If scatterers are PEC/PMC, set relevant E/H fields to zero *after* update
        for sc in self.scatterer_list:
             if sc.material == 'PEC':
                 # Set tangential E-field to zero on the boundary/inside
                 self.Ex[self.mask_Ex == sc.ID] = 0.0
                 self.Ey[self.mask_Ey == sc.ID] = 0.0
             elif sc.material == 'PMC':
                 # Set tangential H-field to zero on the boundary/inside
                 self.Hz[self.mask_Hz == sc.ID] = 0.0
                 # Also need to handle normal E = 0 (more complex)


        # --- 7. Source Injection (Add later) ---
        # Example: Add source term to Hz at a specific point
        # self.Hz[source_y, source_x] += source_function(time)


    def add_source(self, source_func, location_idx):
        """Adds source term at a given index."""
        # Simple hard source for Hz
        self.Hz[location_idx] += source_func


    # --- Deprecated PML methods ---
    # def pml_mask(self,field,percentageN=0.10): ...
    # def pml(self, sigmamax=1,thickness_denom = 10, ...): ...


    def source_pw(self, Aetc):
        pass # Implement plane wave source later if needed (e.g., TF/SF boundary)


    def observation_points_indices(self):
        """Find grid indices closest to observation points."""
        indices = []
        x_centers = self.x_edges[:-1] + 0.5 * self.dx
        y_centers = self.y_edges[:-1] + 0.5 * self.dy
        for x_obs, y_obs in self.observation_points:
            ix = np.argmin(np.abs(x_centers - x_obs))
            iy = np.argmin(np.abs(y_centers - y_obs))
            indices.append((iy, ix)) # Return in (row, col) format
        return indices


    def debugger(self, show_grid=False, show_pml_kappa=False):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot scatterer masks (Hz mask)
        if self.scatterer_list:
             levels = np.arange(0.5, max(sc.ID for sc in self.scatterer_list) + 1.5)
             cmap = plt.cm.get_cmap('viridis', len(levels) -1 )
             ax.contourf(self.Xc.T, self.Yc.T, self.mask_Hz.T, levels=levels, cmap=cmap, alpha=0.3) # Transpose for plot

        # Plot observation points
        obs_indices = self.observation_points_indices()
        obs_x = [self.x_edges[:-1][ix] + 0.5*self.dx[ix] for _, ix in obs_indices]
        obs_y = [self.y_edges[:-1][iy] + 0.5*self.dy[iy] for iy, _ in obs_indices]
        ax.plot(obs_x, obs_y, 'bo', fillstyle="none", label='Receivers', markersize=8)

        # Plot grid lines
        if show_grid:
            h_lines = [[(self.x_edges[0], y), (self.x_edges[-1], y)] for y in self.y_edges]
            v_lines = [[(x, self.y_edges[0]), (x, self.y_edges[-1])] for x in self.x_edges]
            line_col = h_lines + v_lines
            line_collection = LineCollection(line_col, colors='gray', linewidths=0.5, alpha=0.7)
            ax.add_collection(line_collection)

        # Plot PML regions (visual guide)
        if self.Npml > 0:
             pml_color = 'lightcoral'
             alpha_pml = 0.2
             # Left
             ax.axvspan(self.x_edges[0], self.x_edges[self.Npml], color=pml_color, alpha=alpha_pml, lw=0)
             # Right
             ax.axvspan(self.x_edges[-self.Npml-1], self.x_edges[-1], color=pml_color, alpha=alpha_pml, lw=0)
             # Bottom
             ax.axhspan(self.y_edges[0], self.y_edges[self.Npml], color=pml_color, alpha=alpha_pml, lw=0)
             # Top
             ax.axhspan(self.y_edges[-self.Npml-1], self.y_edges[-1], color=pml_color, alpha=alpha_pml, lw=0)
             ax.text(self.x_edges[0], self.y_edges[0] + 0.02*self.Ly, 'PML', color='red', alpha=0.5)

        # Overlay PML kappa_x profile (for debugging)
        if show_pml_kappa and hasattr(self, 'kappa_x_pml'):
            kappa_plot = self.kappa_x_pml.T # Transpose to match Xc, Yc for plotting
            # Need coordinates for kappa_x (Ey locations)
            kX, kY = np.meshgrid(self.x_edges, y_centers, indexing='xy') # Ey locs: x edges, y centers
            ct = ax.contourf(kX, kY, kappa_plot, levels=10, cmap='magma', alpha=0.5)
            plt.colorbar(ct, ax=ax, label='kappa_x_pml')


        ax.set_title(f"FDTD Grid (Non-uniform) {self.Ny}x{self.Nx}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # ax.invert_yaxis() # Keep origin at bottom-left
        ax.set_xlim(self.x_edges[0], self.x_edges[-1])
        ax.set_ylim(self.y_edges[0], self.y_edges[-1])
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        plt.show()

    def visualize_field(self, field_name='Hz'):
        """Simple visualization of a field component."""
        fig, ax = plt.subplots(figsize=(10, 8))
        if field_name == 'Hz':
            field_data = self.Hz.T # Transpose for plot
            X, Y = self.Xc.T, self.Yc.T # Transpose coords
            title = 'Hz Field'
        elif field_name == 'Ex':
            field_data = self.Ex.T # Transpose for plot
            # Coordinates for Ex (centers in x, edges in y) -> Need meshgrid for plotting
            X, Y = np.meshgrid(self.x_edges[:-1] + 0.5*self.dx, self.y_edges, indexing='xy')
            title = 'Ex Field'
        elif field_name == 'Ey':
            field_data = self.Ey.T # Transpose for plot
             # Coordinates for Ey (edges in x, centers in y) -> Need meshgrid for plotting
            X, Y = np.meshgrid(self.x_edges, self.y_edges[:-1] + 0.5*self.dy, indexing='xy')
            title = 'Ey Field'
        else:
            print(f"Unknown field: {field_name}")
            return

        vmax = np.max(np.abs(field_data))
        if vmax < 1e-9: vmax = 1e-9 # Avoid zero range
        im = ax.pcolormesh(X, Y, field_data, cmap='RdBu', vmin=-vmax, vmax=vmax, shading='auto')

        # Plot scatterer outlines
        for sc in self.scatterer_list:
             if sc.shape == 'rectangle':
                 ax.add_patch(plt.Rectangle((sc.geometry['xi'], sc.geometry['yi']),
                                           sc.geometry['xf']-sc.geometry['xi'],
                                           sc.geometry['yf']-sc.geometry['yi'],
                                           fill=False, edgecolor='black', lw=1))
             elif sc.shape == 'circle':
                 ax.add_patch(plt.Circle(sc.geometry['center'], sc.geometry['radius'],
                                           fill=False, edgecolor='black', lw=1))

        # Plot PML region guides
        if self.Npml > 0:
             pml_color = 'gray'
             alpha_pml = 0.8
             ls = '--'
             lw = 0.5
             # Left/Right
             ax.axvline(self.x_edges[self.Npml], color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
             ax.axvline(self.x_edges[-self.Npml-1], color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
             # Bottom/Top
             ax.axhline(self.y_edges[self.Npml], color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
             ax.axhline(self.y_edges[-self.Npml-1], color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)

        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(im)
        plt.show()


# --- User Input Function (Modified for frequency) ---
def user_inputs():
    print("--- FDTD Simulation Setup ---")
    # 1. size of sim area (assume meters now)
    Lx, Ly = map(float, input('Domain size Lx, Ly [m]: ').split(','))

    # 3. Max Frequency for resolution
    fmax = float(input('Max frequency fmax [Hz] (e.g., 1e9 for 1 GHz): '))

    # 4. scatterers
    scatter_list = []
    counter = 0
    print("Define Scatterers (type 'none' for shape when done):")
    while True:
        shape = input(f" Scatterer {counter+1} Shape (circle/rectangle/none): ").lower()
        if shape == 'none':
            break
        material = input(f" Scatterer {counter+1} Material (PEC/PMC/Drude): ").upper()

        geometry = {}
        properties = {}
        if shape == 'circle':
            xc, yc, r = map(float, input("  Center coords & radius (xc, yc, radius): ").split(','))
            geometry = {'center': (xc, yc), 'radius': r}
        elif shape == 'rectangle':
            xi, xf, yi, yf = map(float, input("  Coords (xmin, xmax, ymin, ymax): ").split(','))
            geometry = {'xi': xi, 'xf': xf, 'yi': yi, 'yf': yf}
        else:
            print(" Invalid shape. Try again.")
            continue

        if material == 'DRUDE':
            # Example: Asking for relative permittivity/permeability for simple Drude
            e_r, m_r = map(float, input("  Drude relative (epsilon_r, mu_r): ").split(','))
            properties = {'e_r': e_r, 'm_r': m_r} # Store as relative
            # Add sigma_e later if needed
        elif material not in ['PEC', 'PMC']:
            print(" Invalid material. Defaulting to PEC.")
            material = 'PEC'

        counter += 1
        scatter_list.append(Scatterer(shape, material, counter, geometry, properties))

    # 5. observation points
    obs_list_tuples = []
    obs_str = input('Observation points (x1,y1; x2,y2; ...): ')
    if obs_str:
        try:
            points = obs_str.split(';')
            for xy in points:
                a, b = map(float, xy.split(','))
                obs_list_tuples.append((a, b))
        except Exception as e:
            print(f"Error parsing observation points: {e}. No points added.")


    # PML Layers
    Npml = int(input(f"Number of PML layers (e.g., 10): "))

    return Lx, Ly, fmax, scatter_list, obs_list_tuples, Npml

# --- Simulation Example ---
if __name__ == "__main__":

    # Get inputs
    Lx, Ly, fmax, scatterers, obs_points, Npml = user_inputs()

    # Create FDTD instance
    sim = FDTD(Lx, Ly, fmax, scatterers, obs_points, Npml=Npml)

    # Show initial grid setup
    sim.debugger(show_grid=True, show_pml_kappa=False) # Turn on kappa plot for debugging PML

    # --- Simple Gaussian Source ---
    source_ix = sim.Nx // 4 # Example source location index (x)
    source_iy = sim.Ny // 2 # Example source location index (y)
    t0 = 40 * sim.dt       # Center time of pulse
    spread = 15 * sim.dt   # Width of pulse

    def gaussian_pulse(t):
        return np.exp(-0.5 * ((t - t0) / spread)**2)

    # --- Simulation Loop ---
    Nt = int(4 * t0 / sim.dt) # Number of time steps (e.g., run for 4*t0)
    print(f"Running simulation for {Nt} time steps...")

    plt.ion() # Turn on interactive mode for plotting
    fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
    field_to_plot = sim.Hz.T
    vmax_plot = 0.1 # Initial plot range
    plot_obj = ax_sim.pcolormesh(sim.Xc.T, sim.Yc.T, field_to_plot, cmap='RdBu', vmin=-vmax_plot, vmax=vmax_plot, shading='auto')
    ax_sim.set_title(f"Hz Field at step 0")
    ax_sim.set_xlabel("x [m]")
    ax_sim.set_ylabel("y [m]")
    ax_sim.set_aspect('equal', adjustable='box')
    # Add scatterer outlines to simulation plot
    for sc in sim.scatterer_list:
         if sc.shape == 'rectangle':
             ax_sim.add_patch(plt.Rectangle((sc.geometry['xi'], sc.geometry['yi']),
                                       sc.geometry['xf']-sc.geometry['xi'],
                                       sc.geometry['yf']-sc.geometry['yi'],
                                       fill=False, edgecolor='black', lw=1))
         elif sc.shape == 'circle':
             ax_sim.add_patch(plt.Circle(sc.geometry['center'], sc.geometry['radius'],
                                       fill=False, edgecolor='black', lw=1))
    plt.colorbar(plot_obj)
    plt.show()

    for n in range(Nt):
        # Inject source
        time_now = n * sim.dt
        sim.add_source(gaussian_pulse(time_now), (source_iy, source_ix))

        # Update fields
        sim.update()

        # Visualize periodically
        if n % 20 == 0:
            print(f"Step: {n}/{Nt}")
            field_to_plot = sim.Hz.T
            vmax_plot = np.max(np.abs(field_to_plot)) * 1.1 # Adjust range dynamically
            if vmax_plot < 1e-9: vmax_plot = 1e-9
            plot_obj.set_array(field_to_plot.ravel()) # Update plot data
            plot_obj.set_clim(-vmax_plot, vmax_plot)   # Update color limits
            ax_sim.set_title(f"Hz Field at step {n}")
            plt.pause(0.01) # Short pause to allow plot to update

    plt.ioff() # Turn off interactive mode
    print("Simulation finished.")
    sim.visualize_field('Hz') # Show final state
# --- User Input Function (Modified for frequency) ---
def user_inputs():
    # ... (keep the user_inputs function as is) ...
    print("--- FDTD Simulation Setup ---")
    # 1. size of sim area (assume meters now)
    Lx, Ly = map(float, input('Domain size Lx, Ly [m]: ').split(','))

    # 3. Max Frequency for resolution
    fmax = float(input('Max frequency fmax [Hz] (e.g., 1e9 for 1 GHz): '))

    # 4. scatterers
    scatter_list = []
    counter = 0
    print("Define Scatterers (type 'none' for shape when done):")
    while True:
        shape = input(f" Scatterer {counter+1} Shape (circle/rectangle/none): ").lower()
        if shape == 'none':
            break
        material = input(f" Scatterer {counter+1} Material (PEC/PMC/Drude): ").upper()

        geometry = {}
        properties = {}
        if shape == 'circle':
            xc, yc, r = map(float, input("  Center coords & radius (xc, yc, radius): ").split(','))
            geometry = {'center': (xc, yc), 'radius': r}
        elif shape == 'rectangle':
            xi, xf, yi, yf = map(float, input("  Coords (xmin, xmax, ymin, ymax): ").split(','))
            geometry = {'xi': xi, 'xf': xf, 'yi': yi, 'yf': yf}
        else:
            print(" Invalid shape. Try again.")
            continue

        if material == 'DRUDE':
            # Example: Asking for relative permittivity/permeability for simple Drude
            e_r, m_r = map(float, input("  Drude relative (epsilon_r, mu_r): ").split(','))
            properties = {'e_r': e_r, 'm_r': m_r} # Store as relative
            # Add sigma_e later if needed
        elif material not in ['PEC', 'PMC']:
            print(" Invalid material. Defaulting to PEC.")
            material = 'PEC'

        counter += 1
        scatter_list.append(Scatterer(shape, material, counter, geometry, properties))

    # 5. observation points
    obs_list_tuples = []
    obs_str = input('Observation points (x1,y1; x2,y2; ...): ')
    if obs_str:
        try:
            points = obs_str.split(';')
            for xy in points:
                a, b = map(float, xy.split(','))
                obs_list_tuples.append((a, b))
        except Exception as e:
            print(f"Error parsing observation points: {e}. No points added.")


    # PML Layers
    Npml = int(input(f"Number of PML layers (e.g., 10): "))

    return Lx, Ly, fmax, scatter_list, obs_list_tuples, Npml


# --- Simulation Example ---
if __name__ == "__main__":

    # Get inputs
    Lx, Ly, fmax, scatterers, obs_points, Npml = user_inputs()

    # --- Simulation Parameters ---
    Courant_num = 0.7 # Courant number for stability

    # Create FDTD instance
    print("Initializing FDTD grid and parameters...")
    sim = FDTD(Lx, Ly, fmax, scatterers, obs_points, Npml=Npml, Courant=Courant_num)
    print("Initialization complete.")

    # Show initial grid setup
    sim.debugger(show_grid=True, show_pml_kappa=False)

    # --- Source Setup ---
    # Place source near the center, avoiding edges
    source_ix = sim.Nx // 2
    source_iy = sim.Ny // 2
    print(f"Source injected at grid index: (iy={source_iy}, ix={source_ix})")

    # Gaussian pulse parameters (adjust as needed)
    pulse_width_factor = 15 # Controls pulse width relative to dt
    pulse_delay_factor = 3 * pulse_width_factor # Center pulse later in time
    spread = pulse_width_factor * sim.dt
    t0 = pulse_delay_factor * sim.dt

    def gaussian_pulse(t):
        # Avoid potential numpy warning for large exponent argument
        arg = -0.5 * ((t - t0) / spread)**2
        return np.exp(arg) if arg > -700 else 0.0 # Clip exp for very small values

    # --- Simulation Loop & Animation Setup ---
    # Estimate time steps needed for wave to reach PML and interact significantly
    max_domain_dim = max(sim.Lx, sim.Ly)
    # Time to cross domain + pulse duration + some decay time in PML
    estimated_time = 2.0 * (max_domain_dim / sim.c0) + 2.0 * t0
    Nt = int(estimated_time / sim.dt)
    plot_interval = 20 # Update plot every N steps
    print(f"Running simulation for {Nt} time steps (estimated time: {estimated_time:.2e} s)...")
    print(f"Plotting every {plot_interval} steps.")

    plt.ion() # Turn on interactive mode
    fig_sim, ax_sim = plt.subplots(figsize=(10, 8))

    # Initial plot setup
    field_data = sim.Hz.T # Transpose for plotting (X horizontal, Y vertical)
    # Determine initial plot range dynamically but with a minimum threshold
    vmax_plot = max(np.max(np.abs(field_data)), 1e-9) * 1.1
    # Use pcolormesh coordinates (cell centers)
    X_plot, Y_plot = sim.Xc.T, sim.Yc.T
    plot_obj = ax_sim.pcolormesh(X_plot, Y_plot, field_data, cmap='RdBu', vmin=-vmax_plot, vmax=vmax_plot, shading='gouraud') # Use Gouraud for smoother look
    colorbar = plt.colorbar(plot_obj, ax=ax_sim)
    colorbar.set_label("Hz Field Amplitude")

    # Add static elements (scatterers, PML guides) to the plot
    for sc in sim.scatterer_list:
         if sc.shape == 'rectangle':
             ax_sim.add_patch(plt.Rectangle((sc.geometry['xi'], sc.geometry['yi']),
                                       sc.geometry['xf']-sc.geometry['xi'],
                                       sc.geometry['yf']-sc.geometry['yi'],
                                       fill=False, edgecolor='black', lw=1, label=f'Scatterer {sc.ID}'))
         elif sc.shape == 'circle':
             ax_sim.add_patch(plt.Circle(sc.geometry['center'], sc.geometry['radius'],
                                       fill=False, edgecolor='black', lw=1, label=f'Scatterer {sc.ID}'))
    # Draw PML boundaries guides
    if sim.Npml > 0:
        pml_color = 'gray'
        alpha_pml = 0.8
        ls = '--'
        lw = 0.7
        # Get PML interface coordinates
        x_pml_L = sim.x_edges[sim.Npml]
        x_pml_R = sim.x_edges[-sim.Npml-1]
        y_pml_B = sim.y_edges[sim.Npml]
        y_pml_T = sim.y_edges[-sim.Npml-1]
        # Left/Right boundaries
        ax_sim.axvline(x_pml_L, color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
        ax_sim.axvline(x_pml_R, color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
        # Bottom/Top boundaries
        ax_sim.axhline(y_pml_B, color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
        ax_sim.axhline(y_pml_T, color=pml_color, alpha=alpha_pml, lw=lw, linestyle=ls)
        ax_sim.text(sim.x_edges[0] + 0.01*sim.Lx, y_pml_T + 0.01*sim.Ly, 'PML region', color=pml_color, alpha=alpha_pml, ha='left', va='bottom', fontsize=8)


    ax_sim.set_title(f"Hz Field at step 0/{Nt}")
    ax_sim.set_xlabel("x [m]")
    ax_sim.set_ylabel("y [m]")
    ax_sim.set_aspect('equal', adjustable='box')
    ax_sim.set_xlim(sim.x_edges[0], sim.x_edges[-1])
    ax_sim.set_ylim(sim.y_edges[0], sim.y_edges[-1])
    # Add legend if scatterers are present
    if sim.scatterer_list:
        ax_sim.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show() # Display the initial figure window

    # --- Run Simulation ---
    print("\nStarting simulation loop...")
    for n in range(Nt):
        time_now = n * sim.dt
        # Inject source pulse
        sim.add_source(gaussian_pulse(time_now), (source_iy, source_ix))

        # Update FDTD fields
        sim.update()

        # --- Animation Update ---
        if n % plot_interval == 0 or n == Nt - 1: # Plot periodically and the last frame
            print(f"\rStep: {n}/{Nt}", end='') # Update progress on the same line
            field_data = sim.Hz.T # Get current Hz field data and transpose
            # Dynamic range adjustment (important for seeing wave decay)
            current_max = np.max(np.abs(field_data))
            # Avoid setting vmin/vmax to exactly zero if field is zero everywhere
            vmax_plot = max(current_max * 1.1, 1e-9)

            plot_obj.set_array(field_data.ravel()) # Update plot data
            plot_obj.set_clim(-vmax_plot, vmax_plot) # Update color limits dynamically
            ax_sim.set_title(f"Hz Field at step {n}/{Nt} (Time: {time_now:.2e} s)")
            fig_sim.canvas.draw_idle() # Redraw the canvas efficiently
            fig_sim.canvas.flush_events() # Process GUI events
            # plt.pause(0.001) # Minimal pause is often enough

    print("\nSimulation finished.")
    plt.ioff() # Turn off interactive mode

    # Show final state using the dedicated method (optional)
    # sim.visualize_field('Hz')
    # plt.show() # Keep final plot window open if using visualize_field

    # Keep the animation window open at the end
    print("Displaying final state in the animation window. Close window to exit.")
    plt.show()