import numpy as np
import matplotlib.pyplot as plot
import pylab
from math import sin, cos, tan  

x_n = 0.1
y_n = -1
vector_x_n = np.array([x_n, y_n])
vector_x_n1 = np.array([0, 0])
alpha = 0.5

def objective_function(vector_x_n):
    fx_y = 100*((vector_x_n[1] - vector_x_n[0]**2)**2) + (1 - vector_x_n[0])**2
    return(fx_y)
def gradient_derivative_x(vector_x_n):
    fprime_x_n = 2*(200*(vector_x_n[0])**3 - 200*vector_x_n[0]*vector_x_n[1] + vector_x_n[0] - 1 )
    return(fprime_x_n)

def gradient_derivative_y(y_n):
    fprime_y_n = 200*vector_x_n[1] - 200*(vector_x_n[0]**2)
    return(fprime_y_n)
print(vector_x_n)
gradient = np.array([gradient_derivative_x(vector_x_n), gradient_derivative_y(vector_x_n)])
point_storeSD = vector_x_n
F_storeSD = objective_function(vector_x_n)
vector_x_n1 = vector_x_n - alpha*(gradient*objective_function(vector_x_n))

while((np.linalg.norm(vector_x_n1 - vector_x_n) >= pow(10, -6)) & (objective_function(vector_x_n) >= pow(10,-6))):
    update = np.array([gradient_derivative_x(vector_x_n), gradient_derivative_y(vector_x_n)])
    
    vector_x_n1 = vector_x_n - alpha*(update)
    #print("x", vector_x_n, "f(x)", objective_function(vector_x_n), objective_function(vector_x_n1))
    #print("grad", update, alpha)

    
    if( objective_function(vector_x_n)>  objective_function(vector_x_n1)):
        alpha = 1.0
        vector_x_n = vector_x_n1
        vector_x_n1 = vector_x_n - alpha*(update)
        point_storeSD = np.vstack((point_storeSD, vector_x_n))
        F_storeSD = np.vstack((F_storeSD, objective_function(vector_x_n)))
        #print('accept')
    else:
        alpha = alpha/2
        