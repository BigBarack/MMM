import numpy as np
def main():
    # Example usage
    global dx ,dy 
    dx =0.1
    dy =0.1
    
    field = np.random.rand(10, 10)  # Example field
    result = lap2d4th(field)
    print(result)




def lap2d4th(field):
        global dx,dy
        """
        4th order laplacian in 2D
        :param field: field to be laplacianed
        :return: laplacian of field
        :taken from 3.130 course

        this would mean that the field are larger by two in both directions. since i need to do plus one and plus two
        the shapes would be in total Ny+2,Ny+2 but i want to make it into Nx,Ny thus there needs to be a
        field[2:-2,2:-2] would result in (Nx,Ny) that field will be shifted around and for regular field updates it would be better
        to take this
        """
        #gonna print all different field shapes
        #print(field[4:, 2:-2].shape)
        #print(16*field[3:-1, 2:-2].shape)
        #print(-30*field[2:-2, 2:-2].shape)
        #print(16*field[1:-3, 2:-2].shape)
        #print(-field[:-4, 2:2].shape)
        #print(field[2:-2, 2:-2].shape)
        fieldxcomp = -field[4:, 2:-2] + 16*field[3:-1, 2:-2] - 30*field[2:-2, 2:-2] + 16*field[1:-3, 2:-2] - field[:-4, 2:-2]
        fieldycomp = -field[2:-2, 4:] + 16*field[2:-2, 3:-1] - 30*field[2:-2, 2:-2] + 16*field[2:-2, 1:-3] - field[2:-2, :-4]
        lap=fieldxcomp/(12*dx**2) + fieldycomp/(12*dy**2)

        
        """
        quick question do we need to reset the values after we take the laplacian in the field indices that are not in the H-field? 
        """
        return fieldxcomp / (dx ** 2)












if __name__ == "__main__":
    main()  