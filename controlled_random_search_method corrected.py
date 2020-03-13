from math import cos, sin, tan, pi
import matplotlib.pyplot as plot
import numpy as np
import pylab
import random

#n represents the number of variables in the function
n = 2
#N is a predetermined number of trial points chosen at random over the domain V and stored in an array_A
N_stored_points = n*25
a = -2
b = 2

#Rosenbrock function (banana function) global minimum at (1,1) a = -2 b = 2
objective_function = lambda x, y: 100*((y - x**2)**2) + (1 - x**2)

# ??? function optimum at -308.63418551 a = -4 b = 4
# fx_y = (x**2 + y - 11)**2 + (x+y**2 - 7)**2
#objective_function = lambda x, y: (x**2 + y - 11)**2 + (x+y**2 - 7)**2

# the dimension corresponsdes to the number of variables
# dimension = 2
#Rastrigin function global optimum at (0,0) a = -5.12 b = 5.12
# rastrigin_function = lambda x, y: (x**2 - 10*np.cos(2*pi*x)) + (y**2 - 10*np.cos(2*pi*y))
# objective_function = lambda x, y: dimension*(10) + (rastrigin_function(x, y))

#Schaffer Function N. 2 global minimum at (0,0) a = -100 b = 100 
# objective_function = lambda x, y: 0.5 + (pow(np.sin(x**2 - y**2), 2) - 0.5)/(pow(1 + 0.001*(x**2 +y**2), 2))

#a circle constrained by another smaller circle with a global optimum at (0, -2)
#objective_function = lambda x, y, z: (2*x + 2*z*x)**2 + (2*y + 2*z*y -10)**2 + (x**2 + y**2 -4)**2

array_A = np.random.uniform(a, b, (N_stored_points, n))

Point_Mx = np.max(array_A[:,0].tolist())
Point_My = np.max(array_A[:,1].tolist())
#Point_Mz = np.max(array_A[:,2].tolist())
Point_Lx = np.min(array_A[:,0].tolist())
Point_Ly = np.min(array_A[:,1].tolist())
#Point_Lz = np.min(array_A[:,2].tolist())

#should be selected randomly from a certain set of trial points
Points_P = ""

#point of greatest value of the current N points stored (add this back in when you have a 3 dimensional function: , Point_Mz , Point_Lz)
Function_at_M = objective_function(Point_Mx, Point_My)
Function_at_L = objective_function(Point_Lx, Point_Ly)
# print(Function_at_M)
# print(Function_at_L)


step = 0
#add these back in when you have a 3 dimensional function: , Point_Mz , Point_Lz
while(np.linalg.norm(np.array([Point_Mx, Point_My]) - np.array([Point_Lx, Point_Ly])) >= pow(10, -6)):
#determine if P is within domain V 
# if no the program should choose R-Rn+1 and calculate centroid_G and Points_P 
# until P is within domain V 
    Rloop = True
    #add this back for 3 dimensions: , array_A[:,2]
    Function_at_array_A = objective_function(array_A[:,0], array_A[:,1])

    Point_Mx = array_A[np.argmax(Function_at_array_A),0]
    Point_My =  array_A[np.argmax(Function_at_array_A),1]
    #Point_Mz =  array_A[np.argmax(Function_at_array_A),2]
    
    Point_Lx = array_A[np.argmin(Function_at_array_A),0]
    Point_Ly = array_A[np.argmin(Function_at_array_A),1]
   #Point_Lz = array_A[np.argmin(Function_at_array_A),2]
    print("first loop")
    while(Rloop):
        print("inner loop")
        #Should choose n+1 random points, R-Rn+1,
        #  and it creates a collection of points called a simplex 
        #add this back for 3 dimensions: , Point_Lz
        R_1 = np.array([Point_Lx, Point_Ly])
        
        Rint = np.argmin(Function_at_array_A)
        Rint2 = np.argmin(Function_at_array_A)
        Rint3 = np.argmin(Function_at_array_A)

        while(Rint == np.argmin(Function_at_array_A) or Rint2 == np.argmin(Function_at_array_A) or Rint3 == np.argmin(Function_at_array_A)):
            Rint = np.random.randint(N_stored_points)
            Rint2 = np.random.randint(N_stored_points)
            Rint3 = np.random.randint(N_stored_points) 
        #add back for 3 dimensions: , array_A[Rint,2] , array_A[Rint2,2] , array_A[Rint3,2]
        R_2 =  [array_A[Rint,0], array_A[Rint,1]]
        R_3 = [array_A[Rint2,0], array_A[Rint2,1]]
        R_n1 = [array_A[Rint3,0], array_A[Rint3,1]]
        print(R_1)
        print(R_2)
        print(R_3)
        simplex = [R_1, R_2, R_3]
        print(simplex)

        #this should take all the values from the simplex, find the average, and assign it to the centroid
        centroid_G = np.mean(simplex, axis=0)
        print("Cent", centroid_G)

        #a new trial point P is chosen from a set of trial points
        # and is calculated with respect to both the centroid and the point R_n1
        Points = np.array(2*centroid_G - R_n1)
        # print(Points)
        Points_P = Points
        print("P", Points_P)
        #add back for 3 dimensions: , Points_P[2]
        Function_at_P = objective_function(Points_P[0], Points_P[1])
        #add this back for 3 dimensions: , Point_Mz
        Function_at_M = objective_function(Point_Mx, Point_My)
        print("F(P)", Function_at_P)
        print("F(M)", Function_at_M)
        # print(Function_at_P > Function_at_M)
        #add back for 3 dimensions:  or abs(Points[2]) > b
        if(abs(Points_P[0]) > b or abs(Points[1]) > b):
            Rloop =  True
            #rerun loop to find random numbers
        else: 
            if(Function_at_P > Function_at_M):
                Rloop = True
                #rerun loop to find random numbers
            else:
                # print("Here")
                Rloop = False
                #don't break and rerun the loop to find random numbers 
                # instead overwrite Points_P to be the X_max and Y_max in array A
                array_A[np.argmax(Function_at_array_A),0] = Points_P[0]
                array_A[np.argmax(Function_at_array_A),1] = Points_P[1]
                #array_A[np.argmax(Function_at_array_A),2] = Points_P[2]
                #add back for 3 dimensions: , array_A[:,2]
                Function_at_array_A = objective_function(array_A[:,0], array_A[:,1])

                Point_Mx = array_A[np.argmax(Function_at_array_A),0]
                Point_My =  array_A[np.argmax(Function_at_array_A),1]
                #Point_Mz =  array_A[np.argmax(Function_at_array_A),2]
        print(Points_P)
        # Rloop=False
    # break
    #add back for 3 dimensions: , array_A[np.argmax(Function_at_array_A ),2] , Points_P[2]
    print("PointsM", [array_A[np.argmax(Function_at_array_A ),0],array_A[np.argmax(Function_at_array_A ),1]], objective_function(Points_P[0], Points_P[1]))

    step = step+1
    print("step", step)
    # print("Outer Loop")
#add back for 3 dimensions: , array_A[np.argmin(Function_at_array_A ),2]
print("PointsL", [array_A[np.argmin(Function_at_array_A ),0],array_A[np.argmin(Function_at_array_A ),1]])
#add back for 3 dimensions: ,array_A[np.argmax(Function_at_array_A ),2]
print("PointsM", [array_A[np.argmax(Function_at_array_A ),0],array_A[np.argmax(Function_at_array_A ),1]])
#add back for 3 dimensions: , array_A[:,2]
print(objective_function(array_A[:,0], array_A[:,1]))
#add back in for 3 dimensions:  - ((Point_Mz - Point_Lz)**2)
print(np.sqrt(abs(((Point_Mx - Point_Lx)**2) - ((Point_My - Point_Ly)**2))))





# x = np.linspace(-2, 2,100)
# y = x 

# XPoints, YPoints  = np.meshgrid(x,y)

# ZPoints = np.log(0.0001 +100*((YPoints - XPoints**2)**2) + (1 - XPoints**2))

# pylab.xlim([-2,2])
# pylab.ylim([-2,2])

# contours = plot.contour(XPoints, YPoints, ZPoints)

# plot.clabel(contours, inline=1, fontsize=10)
# plot.show()

# cp = plot.contour(vector_x_n, vector_x_n1)
# plot.clabel(cp, inline=1, fontsize=10)
# plot.xlabel('X1')
# plot.ylabel('X2')
# plot.show()
