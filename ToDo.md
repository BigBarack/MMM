# FDTD Project - Implementation Plan & To-Do List

## ✅ PHASE 1: Core FDTD Setup (Free Space)

### 1. Gridding Input

- Support **non-uniform** dx and dy.
    
- Allow user to input `dx`, `dy` as lists.
    
- Compute `Nx`, `Ny`, average dx/dy if needed.
    
- Determine field array shapes:
    
    - `Ex.shape = (Nx, Ny-1)`
        
    - `Ey.shape = (Nx-1, Ny)`
        
    - `Hz.shape = (Nx, Ny)`
        

### 2. Time Step Stability Check

- Suggest `dt` from CFL condition.
    
- Warn user if their `dt` is unstable.
    

### 3. Initialize Constants & Arrays

- Constants: `eps0`, `mu0`, `c0`
    
- Arrays: `Ex`, `Ey`, `Hz`, `eps_r`, `mu_r`, etc.
    

### 4. Write Basic Update Equations

- Implement TE update equations for free space.
    
- Use `np.select` for material-specific logic later.
    

### 5. Plane Wave Source

- Read Chapter 5 from material.
    
- Use Total Field / Scattered Field (TF/SF) technique.
    
- Inject wave over a region.
    

### 6. Visualization

- Visualize `Hz` using matplotlib.
    
- Optional: animate or save frames.
    

---

## ✅ PHASE 2: User Input & Interaction

### 7. User Input Prompts

- Wrap input requests into functions.
    
- Gather:
    
    - Grid sizes & spacings
        
    - Time step
        
    - Source parameters
        
    - Scatterer definitions
        
    - Observation points
        

### 8. Scatterer Geometry Input

- Define scatterer shapes & types:
    
    - Options: rectangle, circle, custom
        
    - Assign material: `'free'`, `'drude'`, `'pec'`, `'pmc'`
        
- Create a label mask for domain.
    

---

## ✅ PHASE 3: Add Physics & Complexity

### 9. Handle Scatterer Types

- Use `np.select` to assign update equations:
    
    - PEC → Dirichlet
        
    - PMC → Neumann
        
    - Drude → ADE
        

### 10. Drude Media with ADE

- Introduce polarization/current density fields.
    
- Update fields accordingly.
    

### 11. Boundary Conditions

- Start with Mur absorbing boundaries.
    
- Prepare for adding PML.
    

### 12. Perfectly Matched Layers (PML)

- Add graded material zones at domain boundaries.
    
- Possibly use split-field method.
    

### 13. Observation Points

- Track `Hz` (or other fields) at specified coordinates.
    
- Save for analysis or plotting.
    

---

## ✅ PHASE 4: Extras & Cleanup

### 14. Warnings & Checks

- CFL condition
    
- Sharp grid variation warning
    
- Scatterer overlap checks
    

### 15. Code Organization

- Refactor into modular functions or classes.
    

### 16. Run Experiments

- Validate with known setups:
    
    - PEC box
        
    - Drude slab
        
    - PMC wall
        

---

## 🔽 Required User Inputs

1. Size of simulation area
    
2. Non-uniform gridding (dx, dy lists or options) + time-step
    
3. Plane wave parameters (direction, frequency, amplitude)
    
4. Scatterer geometry & material (Drude, PEC, PMC)
    
5. Observation point coordinates


# ✅ Validation

To ensure the correctness of the implementation, the following validation steps should be performed:

1. **Visual Inspection**  
   Perform a visual, time-domain inspection of the plane wave propagating through the simulation domain. Ensure the wave behaves as expected (e.g., consistent speed, direction, reflection/transmission at boundaries or scatterers).

2. **Observation Points & Analytical Comparison**  
   Record numerical field values at well-chosen observation point(s) and compare them with their analytical counterparts.  
   > ⚠️ When using FFTs to analyze frequency response, make sure to **restrict the frequency domain response** to the **bandwidth of the plane wave** to avoid division by values close to zero.

3. **PMC Cylinder Scattering**  
   Simulate scattering from a circular **PMC** cylinder. An analytical frequency-domain solution is available and can be used for direct comparison.

4. **Dielectric Object Scattering**  
   Perform additional validation using scattering off **well-chosen dielectric objects**, and compare the simulation output to expected behavior (qualitative or quantitative).
