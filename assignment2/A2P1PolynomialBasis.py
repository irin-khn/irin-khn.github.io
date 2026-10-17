## # ---------------------------------------------------
# Irina Khan
# SYDE 572 Assignment 2 Part 1
#
# Objective: 
# Predict the median house value from a large dataset
# that contains various features describing a town,
# using a polynomial basis function with degrees 1
# to 7 for the linear regression functions below.
#
# Please note- the dataset has a LARGE portion of
# racial prejudice. The purpose of this assignment is
# to demonstrate mathematical principles about linear
# basis function regression. The author claims NO
# ACCURACY for the conclusions drawn from the 
# function mapping generated from this script to real
# world applications.
# ---------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

TRAINING_SETS = 100
FEATURES = 13

# The following code imports the data set 'housing.data'
with open('housing.data', 'r') as f:
    lines = f.readlines()[:]

# Get every individual number from the data file
numbers = []
for line in lines:
    numbers.extend(line.split())

# Boston dataset has 14 values per data point, but formatted as 11 data points
# in one row and three data points in the next, alternating down
data = np.array(numbers, dtype=float).reshape(-1, 14)

training_features = data[:TRAINING_SETS, 0:13]
y_target = data[:TRAINING_SETS,13]

testing_features = data[TRAINING_SETS:, 0:13]
y_test = data[TRAINING_SETS:, 13]

training_errors = []
test_errors = []

def basis_poly(data, degree ,num_features):

    # Make 1D data into a 2D column this helps with avoiding an error when
    # only 1 feature is being fed into the 'for' loop below
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)


    # Number of rows of the data array (aka the number of training sets), 
    # it will become the number of columns in the Phi function matrix.
    # This is important because there are two different sets of data
    # being fed into this Phi matrix generating function, 100 sets of 
    # training data, and ~400 sets of test data
    N = data.shape[0]

    # There are 1 + 13d rows, where d is the degree of the polynomial 
    # basis function. (the plus 1 term is for bias)
    phi_matrix = np.zeros((1 + num_features*degree, N)) 

    # Bias basis function
    phi_matrix[0, :] = 1

    
    # The loop will continue to add x^2, x^3, etc. to rows
    # further down in the phi_matrix depending on the degree
    # of the polynomial basis, using the integer counter 'row'
    row = 1
    for d in range(1, degree + 1):
        for i in range(num_features): 
            phi_matrix[row, :] = data[:, i] ** d
            row += 1
        
    return phi_matrix

def predictions_and_weight(phi_matrix, y_target):

    w = np.linalg.pinv(phi_matrix.T) @ y_target
    y_hat = phi_matrix.T @ w

    return y_hat, w

# ---------------------------------------------------
# Linear regression with first-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [14 x 100] from training data for first-order polynomial
phi_train_firstdegree = basis_poly(training_features, 1, FEATURES)
y_hat_train_firstdegree, w_firstdegree = predictions_and_weight(phi_train_firstdegree, y_target)

# Phi function matrix from testing data for first-order polynomial
phi_test_firstdegree = basis_poly(testing_features, 1, FEATURES)

# Predicted y-values with w_firstdegree
y_hat_test_firstdegree = phi_test_firstdegree.T @ w_firstdegree

# Calculating the RMS values:
rms_train_1 = np.sqrt(np.mean((y_target - y_hat_train_firstdegree)**2))
training_errors.append(rms_train_1)
rms_test_1 = np.sqrt(np.mean((y_test - y_hat_test_firstdegree)**2))
test_errors.append(rms_test_1)

# ---------------------------------------------------
# Linear regression with second-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [27 x 100] from training data for second-order polynomial
phi_train_seconddegree = basis_poly(training_features, 2, FEATURES)
y_hat_train_seconddegree, w_seconddegree = predictions_and_weight(phi_train_seconddegree, y_target)

# Phi function matrix from testing data for second-order polynomial
phi_test_seconddegree = basis_poly(testing_features, 2, FEATURES)

# Predicted y-values with w_seconddegree
y_hat_test_seconddegree = phi_test_seconddegree.T @ w_seconddegree

# Calculating the RMS values:
rms_train_2 = np.sqrt(np.mean((y_target - y_hat_train_seconddegree)**2))
training_errors.append(rms_train_2)
rms_test_2 = np.sqrt(np.mean((y_test - y_hat_test_seconddegree)**2))
test_errors.append(rms_test_2)

# ---------------------------------------------------
# Linear regression with third-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [40 x 100] from training data for third-order polynomial
phi_train_thirddegree = basis_poly(training_features, 3, FEATURES)
y_hat_train_thirddegree, w_thirddegree = predictions_and_weight(phi_train_thirddegree, y_target)

# Phi function matrix from testing data for third-order polynomial
phi_test_thirddegree = basis_poly(testing_features, 3, FEATURES)

# Predicted y-values with w_thirddegree
y_hat_test_thirddegree = phi_test_thirddegree.T @ w_thirddegree

# Calculating the RMS values:
rms_train_3 = np.sqrt(np.mean((y_target - y_hat_train_thirddegree)**2))
training_errors.append(rms_train_3)
rms_test_3 = np.sqrt(np.mean((y_test - y_hat_test_thirddegree)**2))
test_errors.append(rms_test_3)

# ---------------------------------------------------
# Linear regression with fourth-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [53 x 100] from training data for fourth-order polynomial
phi_train_fourthdegree = basis_poly(training_features, 4, FEATURES)
y_hat_train_fourthdegree, w_fourthdegree = predictions_and_weight(phi_train_fourthdegree, y_target)

# Phi function matrix from testing data for fourth-order polynomial
phi_test_fourthdegree = basis_poly(testing_features, 4, FEATURES)

# Predicted y-values with w_fourthdegree
y_hat_test_fourthdegree = phi_test_fourthdegree.T @ w_fourthdegree

# Calculating the RMS values:
rms_train_4 = np.sqrt(np.mean((y_target - y_hat_train_fourthdegree)**2))
training_errors.append(rms_train_4)
rms_test_4 = np.sqrt(np.mean((y_test - y_hat_test_fourthdegree)**2))
test_errors.append(rms_test_4)

# ---------------------------------------------------
# Linear regression with fifth-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [66 x 100] from training data for fifth-order polynomial
phi_train_fifthdegree = basis_poly(training_features, 5, FEATURES)
y_hat_train_fifthdegree, w_fifthdegree = predictions_and_weight(phi_train_fifthdegree, y_target)

# Phi function matrix from testing data for fifth-order polynomial
phi_test_fifthdegree = basis_poly(testing_features, 5, FEATURES)

# Predicted y-values with w_fifthdegree
y_hat_test_fifthdegree = phi_test_fifthdegree.T @ w_fifthdegree

# Calculating the RMS values:
rms_train_5 = np.sqrt(np.mean((y_target - y_hat_train_fifthdegree)**2))
training_errors.append(rms_train_5)
rms_test_5 = np.sqrt(np.mean((y_test - y_hat_test_fifthdegree)**2))
test_errors.append(rms_test_5)

# ---------------------------------------------------
# Linear regression with sixth-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [79 x 100] from training data for sixth-order polynomial
phi_train_sixthdegree = basis_poly(training_features, 6, FEATURES)
y_hat_train_sixthdegree, w_sixthdegree = predictions_and_weight(phi_train_sixthdegree, y_target)

# Phi function matrix from testing data for sixth-order polynomial
phi_test_sixthdegree = basis_poly(testing_features, 6, FEATURES)

# Predicted y-values with w_sixthdegree
y_hat_test_sixthdegree = phi_test_sixthdegree.T @ w_sixthdegree

# Calculating the RMS values:
rms_train_6 = np.sqrt(np.mean((y_target - y_hat_train_sixthdegree)**2))
training_errors.append(rms_train_6)
rms_test_6 = np.sqrt(np.mean((y_test - y_hat_test_sixthdegree)**2))
test_errors.append(rms_test_6)

# ---------------------------------------------------
# Linear regression with seventh-order polynomial basis 
# function
# --------------------------------------------------- 

# Phi function matrix [92 x 100] from training data for seventh-order polynomial
phi_train_seventhdegree = basis_poly(training_features, 7, FEATURES)
y_hat_train_seventhdegree, w_seventhdegree = predictions_and_weight(phi_train_seventhdegree, y_target)

# Phi function matrix from testing data for seventh-order polynomial
phi_test_seventhdegree = basis_poly(testing_features, 7, FEATURES)

# Predicted y-values with w_seventhdegree
y_hat_test_seventhdegree = phi_test_seventhdegree.T @ w_seventhdegree

# Calculating the RMS values:
rms_train_7 = np.sqrt(np.mean((y_target - y_hat_train_seventhdegree)**2))
training_errors.append(rms_train_7)
rms_test_7 = np.sqrt(np.mean((y_test - y_hat_test_seventhdegree)**2))
test_errors.append(rms_test_7)

# ---------------------------------------------------
# PART A: Creating the plot of training error and testing error
# against the polynomial degree used in the linear # regression:
# ---------------------------------------------------
degrees = range(1, 8)
print(training_errors)
print(test_errors)

plt.semilogy(degrees, training_errors, label="RMS Error of Training Data")
plt.semilogy(degrees, test_errors, label="RMS Error of Test Data")

plt.xlabel("Polynomial Degree")
plt.ylabel("RMS Error")
plt.title("Training Error and Test Error (in RMS Error) vs. Polynomial Degree")
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.minorticks_on()
plt.legend()
plt.show()

# ---------------------------------------------------
# PART B: Printing out the weights of the first-order
# polynomial for all 13 features
# ---------------------------------------------------

# Note that at index 0 of the list 'w_firstdegree', the weight is 
# for the bias/intercept of the polynomial.
# The 6th feature occurs at index 6, aka the seventh weight, and
# the 8th feature occurs at index 8, aka the ninth weight.
print(w_firstdegree)


# ---------------------------------------------------
# PART C: Creating the visualization of three different polynomial 
# basis functions of the 2nd feature of the dataset (aka ZN, 
# proportion of residential land zoned for lots over 25,000 sq.ft.)
# ---------------------------------------------------

def polynomial_regression_1d_vis(degree):

    # The second feature of the dataset, aka data on the proportion of residential land zones
    # [ZN] is the only feature that will be visualized against our various-ordered polynomial basis functions.
    ZN_data = data[:TRAINING_SETS, 1]
    ZN_testing_data = data[TRAINING_SETS:, 1]

    ZN_y_target = data[:TRAINING_SETS, 13]
    ZN_y_test = data[TRAINING_SETS:, 13]

    phi_train = basis_poly(ZN_data, degree, 1)
    y_hat_train, w_ZN = predictions_and_weight(phi_train, ZN_y_target)

    phi_test = basis_poly(ZN_testing_data, degree, 1)
    y_hat_test = phi_test.T @ w_ZN

    # We are feeding the spacing of the polynomial functions through the 
    # polynomial Phi matrix generator, so that all of the values of x 
    # along the linspace will map onto the polynomial equation
    x_plot = np.linspace(np.min(ZN_data), np.max(ZN_data), 500)
    phi_plot = basis_poly(x_plot, degree, 1)
    y_plot = phi_plot.T @ w_ZN

    # VISUALIZATION SECTION OF PART C BELOW!

    plt.figure(figsize=(8,6))

    plt.scatter(ZN_data, ZN_y_target, label="Training data")
    plt.scatter(ZN_testing_data, ZN_y_test, label="Test data")

    plt.plot(x_plot, y_plot, label=f"Degree {degree} polynomial")
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.xlabel("Proportion of residential land zoned for lots over 25,000 sq.ft (%)")
    plt.ylabel("Median House Value ($1000s)")
    plt.title(f"Training/Test Data vs. Degree {degree} Polynomial Regression of \n% Residential Land Zones (ZN) on Median House Value (MEDV)")
    plt.legend()
    plt.show()

# Generating the three different plots for Part 1c) of assignment 2:
polynomial_regression_1d_vis(2)
polynomial_regression_1d_vis(5)
polynomial_regression_1d_vis(7)

# ---------------------------------------------------
# PART D: Implementing a ridge regression (with cross validation) 
# using the first 100 data points and only the 2nd feature 
# to fit an eighth-order polynomial.
# Lambda values used: 0, 0.01, 0.1, 1, 10, 100, 1000
# ---------------------------------------------------

def ridge_regression(rows, Phi, y_val, hp_lambda):
    I = np.eye(rows)
    I[0, 0] = 0

    w_fold = (np.linalg.inv(Phi @ Phi.T + hp_lambda * I)) @ Phi @ y_val
    y_pred = Phi.T @ w_fold

    return y_pred, w_fold

# It seems that just 'lambda' is a special keyword, so its now called 
# 'hp_lambda' for hyperparameter lambda. (not Harry Potter)
def polynomial_regression_reg(hp_lambda):
    # The second feature of the dataset, aka data on the proportion of residential land zones
    # [ZN] is the only feature that will be visualized against our various-ordered polynomial basis functions.

    # The following two arrays will be further subdivided into 10 groups of 10 datasets, 
    # in order to create a 10-fold cross validation method.
    ZN_data = data[:TRAINING_SETS, 1]
    ZN_y_target = data[:TRAINING_SETS, 13]

    # The remaining ~400 dataset points in the following 2 arrays 
    # will be used as our testing data again.
    ZN_testing_data = data[TRAINING_SETS:, 1]
    ZN_y_test = data[TRAINING_SETS:, 13]


    # According to cross-validation, these 2 arrays will hold 10 datapoints
    ZN_folded_test_data = []
    ZN_folded_test_y = []
    

    folds_data = []
    folds_target = []

    lambda_rms = 0

    for i in range(10):
        folds_data.append(ZN_data[(i*10):(i*10+10)])
        folds_target.append(ZN_y_target[(i*10):(i*10+10)])

    for i in range(10):
        # The following 2 arrays will hold 90 datapoints
        ZN_folded_train_data = []
        ZN_folded_train_y = []

        for k in range(10):
            # Here, the kth fold is the test data, and the remaining 9 folds act
            # as the training data.
            if k == i:
                continue
            else:
                ZN_folded_train_data.extend(folds_data[k])
                ZN_folded_train_y.extend(folds_target[k])

        #The following generates a Phi function matrix off of training data (90 datasets)
        phi_train_per_fold = basis_poly(ZN_folded_train_data, 8, 1)
        y_hat_train_per_fold, w_fold = ridge_regression(phi_train_per_fold.shape[0], phi_train_per_fold, ZN_folded_train_y, hp_lambda)

        #The following generates a Phi function matrix off of our test data fold (10 datasets)
        phi_validation = basis_poly(folds_data[i], 8, 1)
        y_hat_validation = phi_validation.T @ w_fold

            
        #Calculating the validation RMS values:
        rms_validation = np.sqrt(np.mean((folds_target[i] - y_hat_validation)**2))
        lambda_rms += rms_validation

    lambda_rms /= 10
    return lambda_rms

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
rms_of_lambdas = []

for hp_lambda in lambdas:
    rms = polynomial_regression_reg(hp_lambda)
    rms_of_lambdas.append(rms)
    print("Lambda:", hp_lambda)
    print("The corresonding cross-validated RMS is: ", rms)

plt.axhline(y=rms_of_lambdas[0], color='green', linestyle='--', label='RMS value when lambda = 0')

# The first lambda value (0) is not plotted on the semilogx, since it is plotted as a line above.
plt.semilogx(lambdas[1:], rms_of_lambdas[1:], marker='o', color='blue', label='RMS values of lambda')
plt.xlabel('Lambda values ')
plt.ylabel('RMS of Eighth Degree Polynomial Basis')
plt.title('Plot of Cross-validated RMS values for Eighth Degree Polynomial \nBasis vs. Hyperparameter Lambda Values')
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.legend()

lambda_0 = np.linspace(0.01, 1000, 1000)
y_lambda_0 = rms_of_lambdas[0] * lambda_0

plt.show()