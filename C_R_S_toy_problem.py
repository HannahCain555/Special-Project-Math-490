from math import cos, sin, tan, pi
import matplotlib.pyplot as plot
import numpy as np
import pylab
import random

#n represents the number of variables in the function
n = 1
#N is a predetermined number of trial points chosen at random over the domain V and stored in an array_A
N_stored_points = n*9
a = 0
b = 1

#Overlap cost of the classes, taken from data collected by the Math department in order
# will be calc 1 calc 2 and linear algebra in columns 1, 2, and 3 respectively 
# and the same is true for the rows.
Overlap_Cost =  np.array([[0,0,11],[0,0,7],[11,7,0]])
print(Overlap_Cost)

array_A = np.around(np.random.random_sample((N_stored_points, n)))
array_A = np.reshape(array_A, [3,3])

#this array represents the cost of classes overlapping. The first column is calc 1, 
# the second is calc 2, the third is Linear Algebra and the rows are in the same order.
Overlap_Cost =  np.array(([[0, 0, 15], [0, 0,7], [15,7,0]]))

objective_function = 0
#new objective function will sum up all the values for the number of classes we are offering 
def objective_function(x, y, z):
    for l in range(0,3):
        for j in range(0,3):
            for i in range(0,3):
                objective_function = objective_function + Overlap_Cost[j,l] * array_A[i,j] * array_A[i,l]
    return objective_function(x, y, z)

Point_Mx = np.max(array_A[:,0].tolist())
Point_My = np.max(array_A[:,1].tolist())
Point_Mz = np.max(array_A[:,2].tolist())
Point_Lx = np.min(array_A[:,0].tolist())
Point_Ly = np.min(array_A[:,1].tolist())
Point_Lz = np.min(array_A[:,2].tolist())

#should be selected randomly from a certain set of trial points
Points_P = ""

#point of greatest value of the current N points stored 
Function_at_M = objective_function(Point_Mx, Point_My, Point_Mz)
Function_at_L = objective_function(Point_Lx, Point_Ly, Point_Lz)
# print(Function_at_M)
# print(Function_at_L)


step = 0
while(np.linalg.norm(np.array([Point_Mx, Point_My, Point_Mz]) - np.array([Point_Lx, Point_Ly, Point_Lz])) >= pow(10, -6)):
#determine if P is within domain V 
# if no the program should choose R-Rn+1 and calculate centroid_G and Points_P 
# until P is within domain V 
    Rloop = True
    Function_at_array_A = objective_function(array_A[:,0], array_A[:,1], array_A[:,2])

    Point_Mx = array_A[np.argmax(Function_at_array_A),0]
    Point_My =  array_A[np.argmax(Function_at_array_A),1]
    Point_Mz =  array_A[np.argmax(Function_at_array_A),2]
    
    Point_Lx = array_A[np.argmin(Function_at_array_A),0]
    Point_Ly = array_A[np.argmin(Function_at_array_A),1]
    Point_Lz = array_A[np.argmin(Function_at_array_A),2]

    while(Rloop):
        #Should choose n+1 random points, R-Rn+1,
        #  and it creates a collection of points called a simplex 
        R_1 = np.array([Point_Lx, Point_Ly, Point_Lz])
        
        Rint = np.argmin(Function_at_array_A)
        Rint2 = np.argmin(Function_at_array_A)
        Rint3 = np.argmin(Function_at_array_A)

        while(Rint == np.argmin(Function_at_array_A) or Rint2 == np.argmin(Function_at_array_A) or Rint3 == np.argmin(Function_at_array_A)):
            Rint = np.random.randint(N_stored_points)
            Rint2 = np.random.randint(N_stored_points)
            Rint3 = np.random.randint(N_stored_points) 

        R_2 =  [array_A[Rint,0], array_A[Rint,1], array_A[Rint,2]]
        R_3 = [array_A[Rint2,0], array_A[Rint2,1], array_A[Rint2,2]]
        R_n1 = [array_A[Rint3,0], array_A[Rint3,1], array_A[Rint3,2]]
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
        Points_P = np.around(Points)
        print("P", Points_P)
        Function_at_P = objective_function(Points_P[0], Points_P[1], Points_P[2])
        Function_at_M = objective_function(Point_Mx, Point_My, Point_Mz)
        print("F(P)", Function_at_P)
        print("F(M)", Function_at_M)
        # print(Function_at_P > Function_at_M)
        if(abs(Points_P[0]) > b or abs(Points[1]) > b or abs(Points[2]) > b or np.sum(array_A, axis = 0) != 3):
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
                array_A[np.argmax(Function_at_array_A),2] = Points_P[2]
                Function_at_array_A = objective_function(array_A[:,0], array_A[:,1], array_A[:,2])

                Point_Mx = array_A[np.argmax(Function_at_array_A),0]
                Point_My =  array_A[np.argmax(Function_at_array_A),1]
                Point_Mz =  array_A[np.argmax(Function_at_array_A),2]
        print(Points_P)
        # Rloop=False
    # break
    print("PointsM", [array_A[np.argmax(Function_at_array_A ),0],array_A[np.argmax(Function_at_array_A ),1], array_A[np.argmax(Function_at_array_A ),2]], objective_function(Points_P[0], Points_P[1], Points_P[2]))

    step = step+1
    print("step", step)
    # print("Outer Loop")
print("PointsL", [array_A[np.argmin(Function_at_array_A ),0],array_A[np.argmin(Function_at_array_A ),1], array_A[np.argmin(Function_at_array_A ),2]])
print("PointsM", [array_A[np.argmax(Function_at_array_A ),0],array_A[np.argmax(Function_at_array_A ),1],array_A[np.argmax(Function_at_array_A ),2]])
print(objective_function(array_A[:,0], array_A[:,0], array_A[:,2]))
print(np.sqrt(abs(((Point_Mx - Point_Lx)**2) - ((Point_My - Point_Ly)**2) - ((Point_Mz - Point_Lz)**2))))





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
