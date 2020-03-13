from math import cos, sin, tan
import matplotlib.pyplot as plot
import numpy as np
import pylab

x_n = 0.01
y_n = -1
vector_x_n = np.array([[x_n, y_n]])
vector_x_n1 = np.array([[0, 0]])
alpha = 0.05
precision = pow(10, -6)


# fx_y = 100*((y - x**2)**2) + (1 - x**2)
objective_function = lambda x, y: 100*((y - x**2)**2) + (1 - x)**2

# fprime_x_n = -400*x*y + 400*(x**3) - 2*x
gradient_derivative_x = lambda x, y: 2*(200*x**3 - 200*x*y+ x - 1 )

# fprime_y_n = 200*y - 200*(x**2)
gradient_derivative_y = lambda x, y: (200 * y - 200 * (x**2))

second_partial_derivative_x = lambda x, y: (-400*y + 1200*(x**2))

second_partial_derivative_y = 200

#Both these mixed derivatives's names are defined by their order of operations.
#Thus _x_y means the mixed derivative dx of the partial derivative in terms of y.
#similarly _y_x means the mixed derivative dy of the partial derivative in terms of x.
mixed_partial_derivative_x_y = lambda x: -400*x

mixed_partial_derivative_y_x = lambda x: -400*x

def hessian_matix_inverse(vector_x_n):
    
    hessian_matrix = np.matrix(
        [[second_partial_derivative_x(vector_x_n[0,0], vector_x_n[0,1]), mixed_partial_derivative_y_x(vector_x_n[0,0])],
                                 [mixed_partial_derivative_x_y(vector_x_n[0,0]), second_partial_derivative_y]])
    #hessian_matrix_inverse = np.eye(2)
    hessian_matrix_inverse = np.linalg.inv(hessian_matrix)
    
    return(hessian_matrix_inverse)

# gradient = np.array([gradient_derivative_x(vector_x_n[0],vector_x_n[1]), gradient_derivative_y(vector_x_n[0], vector_x_n[1])])
gradient = lambda x: ([gradient_derivative_x(x[0,0], x[0,1]), gradient_derivative_y(x[0,0],x[0,1])])

print(vector_x_n[0])
gradient(vector_x_n)
# print(gradient(vector_x_n[0], vector_x_n[1]))
# print(np.transpose(gradient(vector_x_n[0], vector_x_n[1])))

point_storeNM = vector_x_n
F_storeNM = objective_function(vector_x_n[0,0], vector_x_n[0,1])
print( (objective_function(vector_x_n[0,0], vector_x_n[0,1])))
while((np.linalg.norm(vector_x_n1 - vector_x_n) >= pow(10, -6)) & (objective_function(vector_x_n[0,0], vector_x_n[0,1]) >= pow(10,-6))):
    
    update = np.matmul(hessian_matix_inverse(vector_x_n), gradient(vector_x_n))
    vector_x_n1 = vector_x_n - alpha*(update)
    #print("x", vector_x_n, "f(x)", objective_function(vector_x_n), objective_function(vector_x_n1))
    

    if( objective_function(vector_x_n[0,0], vector_x_n[0,1])>  objective_function(vector_x_n1[0,0], vector_x_n1[0,1])):
        alpha = 1.0
        vector_x_n = vector_x_n1
        update = np.matmul(hessian_matix_inverse(vector_x_n), gradient(vector_x_n))

        vector_x_n1 = (vector_x_n - alpha*(update))
        point_storeNM = np.vstack((point_storeNM, vector_x_n))
        F_storeNM = np.vstack((F_storeNM, objective_function(vector_x_n[0,0], vector_x_n[0,1])))
    else:
        alpha = alpha/2
        