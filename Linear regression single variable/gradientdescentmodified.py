from costfunction import cost_function

import numpy as np

def gradient_descent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   this function updates theta by 
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = y.shape[0] # number of training examples
    J_history = np.zeros([num_iters, 1]);

    for iter in range(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be usefusl to print out the values
        #       of the cost function (compute_cost) and gradient here.
        #
        
        h_of_theta = np.dot(X, theta)
        theta= theta - alpha*(1/m)*np.sum((h_of_theta-y)*X)

       



        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = cost_function(X, y, theta);

    return theta, J_history



    
