## # ---------------------------------------------------
# Irina Khan
# SYDE 572 Assignment 2 Part 1
#
# Objective: 
# Predict the median house value from a large dataset
# that contains various features describing a town,
# using a gaussian basis function with with a range of
# basis functions from 5 to 95 functions. 
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

# A seed number is chosen so we can preserve the same
# 'pattern' of randomness:
np.random.seed(95) # Ka-chow!

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

def basis_gaussian(data, centers ,a_vals):

    # Make data be in a numpy array format:
    data = np.asarray(data)
    centers = np.asarray(centers)
    a_vals = np.asarray(a_vals)

    # Number of rows of the data array (aka the number of training sets), 
    # it will become the number of columns in the Phi function matrix.
    # This is important because there are two different sets of data
    # being fed into this Phi matrix generating function, 100 sets of 
    # training data, and ~400 sets of test data
    N = data.shape[0]
    K = centers.shape[0]

    # There are K rows, where K is the number of desired basis functions.
    # There are no bias terms, so it is NOT K+1
    phi_matrix = np.zeros((K, N)) 

    # The structure of the gaussian function is coefficient 'a' multiplied
    # with the squared norm of the difference in our data point and the center
    # of the gaussian function, put to the power of e. 
    for k in range(K):
        for n in range(N):
            difference = (np.linalg.norm(data[n] - centers[k])) ** 2

            # Each feature has its own 'a' value that is used to construct the 'k' 
            # gaussian basis functions
            exponent = np.sum(a_vals * difference)

            phi_matrix[k, n] = np.exp(exponent)
         
    return phi_matrix

def predictions_and_weight(phi_matrix, y_target):

    w = np.linalg.pinv(phi_matrix.T) @ y_target
    y_hat = phi_matrix.T @ w

    return y_hat, w

# Main part of the code where it calls on the previous basis_gaussian function and the 
# predictions_and_weight function

# Since the 'a' is the coefficient (-1 / (2 sigma^2)), multiple 'a' values
# are calculated PER feature, so that they are defined by the standard deviation
# of each feature's data spread.

a_vals_gaussian_def = []
random_order_training_sets = np.random.permutation(TRAINING_SETS)

for i in range(FEATURES):
    mean_sum = 0
    mean = 0
    square_diff = 0
    for j in range(TRAINING_SETS):
        mean_sum += training_features[j][i]
    
    mean = mean_sum / TRAINING_SETS

    for k in range(TRAINING_SETS):
        square_diff += (training_features[k][i] - mean)**2

    sigma = (square_diff/(TRAINING_SETS-1))**0.5

    print("Sigma of feature" + str(i+1) +" is " + str(sigma))

    a = -1 / (2 * (sigma**2))   

    # The data points for the 4th feature are all 0 for the first 100 training points. 
    # a will be written as infinity since the fraction above has sigma (which would be 0)
    # as the denominator. The following at least sets the 'a' value of the 4th feature,
    # and thus, the Gaussian basis functions, as zero.
    if (sigma == 0):
        a = 0
    a_vals_gaussian_def.append(a)

print(a_vals_gaussian_def)


training_rms_errors = []
test_rms_errors = []

# ---------------------------------------------------
# Linear regression with 5  basis functions
# --------------------------------------------------- 

K = 5
centers_5 = training_features[random_order_training_sets[:K]]

phi_train_5 = basis_gaussian(training_features, centers_5, a_vals_gaussian_def)
phi_test_5 = basis_gaussian(testing_features, centers_5, a_vals_gaussian_def)

y_hat_train_5, w_5_fxns = predictions_and_weight(phi_train_5, y_target)
y_hat_test_5 = phi_test_5.T @ w_5_fxns

training_rms_5 = np.sqrt(np.mean((y_target - y_hat_train_5)**2))
test_rms_5 = np.sqrt(np.mean((y_test - y_hat_test_5)**2))

training_rms_errors.append(training_rms_5)
test_rms_errors.append(test_rms_5)

print(phi_train_5.shape)
print(phi_test_5.shape)
print(training_rms_5)
print(test_rms_5)

# ---------------------------------------------------
# Linear regression with 15  basis functions
# --------------------------------------------------- 

K = 15
centers_15 = training_features[random_order_training_sets[:K]]

phi_train_15 = basis_gaussian(training_features, centers_15, a_vals_gaussian_def)
phi_test_15 = basis_gaussian(testing_features, centers_15, a_vals_gaussian_def)

y_hat_train_15, w_15_fxns = predictions_and_weight(phi_train_15, y_target)
y_hat_test_15 = phi_test_15.T @ w_15_fxns

training_rms_15 = np.sqrt(np.mean((y_target - y_hat_train_15)**2))
test_rms_15 = np.sqrt(np.mean((y_test - y_hat_test_15)**2))

training_rms_errors.append(training_rms_15)
test_rms_errors.append(test_rms_15)

print(phi_train_15.shape)
print(phi_test_15.shape)
print(training_rms_15)
print(test_rms_15)

# ---------------------------------------------------
# Linear regression with 25  basis functions
# --------------------------------------------------- 

K = 25
centers_25 = training_features[random_order_training_sets[:K]]

phi_train_25 = basis_gaussian(training_features, centers_25, a_vals_gaussian_def)
phi_test_25 = basis_gaussian(testing_features, centers_25, a_vals_gaussian_def)

y_hat_train_25, w_25_fxns = predictions_and_weight(phi_train_25, y_target)
y_hat_test_25 = phi_test_25.T @ w_25_fxns

training_rms_25 = np.sqrt(np.mean((y_target - y_hat_train_25)**2))
test_rms_25 = np.sqrt(np.mean((y_test - y_hat_test_25)**2))

training_rms_errors.append(training_rms_25)
test_rms_errors.append(test_rms_25)

print(phi_train_25.shape)
print(phi_test_25.shape)
print(training_rms_25)
print(test_rms_25)

# ---------------------------------------------------
# Linear regression with 35  basis functions
# --------------------------------------------------- 

K = 35
centers_35 = training_features[random_order_training_sets[:K]]

phi_train_35 = basis_gaussian(training_features, centers_35, a_vals_gaussian_def)
phi_test_35 = basis_gaussian(testing_features, centers_35, a_vals_gaussian_def)

y_hat_train_35, w_35_fxns = predictions_and_weight(phi_train_35, y_target)
y_hat_test_35 = phi_test_35.T @ w_35_fxns

training_rms_35 = np.sqrt(np.mean((y_target - y_hat_train_35)**2))
test_rms_35 = np.sqrt(np.mean((y_test - y_hat_test_35)**2))

training_rms_errors.append(training_rms_35)
test_rms_errors.append(test_rms_35)

print(phi_train_35.shape)
print(phi_test_35.shape)
print(training_rms_35)
print(test_rms_35)

# ---------------------------------------------------
# Linear regression with 45  basis functions
# --------------------------------------------------- 

K = 45
centers_45 = training_features[random_order_training_sets[:K]]

phi_train_45 = basis_gaussian(training_features, centers_45, a_vals_gaussian_def)
phi_test_45 = basis_gaussian(testing_features, centers_45, a_vals_gaussian_def)

y_hat_train_45, w_45_fxns = predictions_and_weight(phi_train_45, y_target)
y_hat_test_45 = phi_test_45.T @ w_45_fxns

training_rms_45 = np.sqrt(np.mean((y_target - y_hat_train_45)**2))
test_rms_45 = np.sqrt(np.mean((y_test - y_hat_test_45)**2))

training_rms_errors.append(training_rms_45)
test_rms_errors.append(test_rms_45)

print(phi_train_45.shape)
print(phi_test_45.shape)
print(training_rms_45)
print(test_rms_45)

# ---------------------------------------------------
# Linear regression with 55  basis functions
# --------------------------------------------------- 

K = 55
centers_55 = training_features[random_order_training_sets[:K]]

phi_train_55 = basis_gaussian(training_features, centers_55, a_vals_gaussian_def)
phi_test_55 = basis_gaussian(testing_features, centers_55, a_vals_gaussian_def)

y_hat_train_55, w_55_fxns = predictions_and_weight(phi_train_55, y_target)
y_hat_test_55 = phi_test_55.T @ w_55_fxns

training_rms_55 = np.sqrt(np.mean((y_target - y_hat_train_55)**2))
test_rms_55 = np.sqrt(np.mean((y_test - y_hat_test_55)**2))

training_rms_errors.append(training_rms_55)
test_rms_errors.append(test_rms_55)

print(phi_train_55.shape)
print(phi_test_55.shape)
print(training_rms_55)
print(test_rms_55)

# ---------------------------------------------------
# Linear regression with 65  basis functions
# --------------------------------------------------- 

K = 65
centers_65 = training_features[random_order_training_sets[:K]]

phi_train_65 = basis_gaussian(training_features, centers_65, a_vals_gaussian_def)
phi_test_65 = basis_gaussian(testing_features, centers_65, a_vals_gaussian_def)

y_hat_train_65, w_65_fxns = predictions_and_weight(phi_train_65, y_target)
y_hat_test_65 = phi_test_65.T @ w_65_fxns

training_rms_65 = np.sqrt(np.mean((y_target - y_hat_train_65)**2))
test_rms_65 = np.sqrt(np.mean((y_test - y_hat_test_65)**2))

training_rms_errors.append(training_rms_65)
test_rms_errors.append(test_rms_65)

print(phi_train_65.shape)
print(phi_test_65.shape)
print(training_rms_65)
print(test_rms_65)

# ---------------------------------------------------
# Linear regression with 75  basis functions
# --------------------------------------------------- 

K = 75
centers_75 = training_features[random_order_training_sets[:K]]

phi_train_75 = basis_gaussian(training_features, centers_75, a_vals_gaussian_def)
phi_test_75 = basis_gaussian(testing_features, centers_75, a_vals_gaussian_def)

y_hat_train_75, w_75_fxns = predictions_and_weight(phi_train_75, y_target)
y_hat_test_75 = phi_test_75.T @ w_75_fxns

training_rms_75 = np.sqrt(np.mean((y_target - y_hat_train_75)**2))
test_rms_75 = np.sqrt(np.mean((y_test - y_hat_test_75)**2))

training_rms_errors.append(training_rms_75)
test_rms_errors.append(test_rms_75)

print(phi_train_75.shape)
print(phi_test_75.shape)
print(training_rms_75)
print(test_rms_75)

# ---------------------------------------------------
# Linear regression with 85  basis functions
# --------------------------------------------------- 

K = 85
centers_85 = training_features[random_order_training_sets[:K]]

phi_train_85 = basis_gaussian(training_features, centers_85, a_vals_gaussian_def)
phi_test_85 = basis_gaussian(testing_features, centers_85, a_vals_gaussian_def)

y_hat_train_85, w_85_fxns = predictions_and_weight(phi_train_85, y_target)
y_hat_test_85 = phi_test_85.T @ w_85_fxns

training_rms_85 = np.sqrt(np.mean((y_target - y_hat_train_85)**2))
test_rms_85 = np.sqrt(np.mean((y_test - y_hat_test_85)**2))

training_rms_errors.append(training_rms_85)
test_rms_errors.append(test_rms_85)

print(phi_train_85.shape)
print(phi_test_85.shape)
print(training_rms_85)
print(test_rms_85)

# ---------------------------------------------------
# Linear regression with 95  basis functions
# --------------------------------------------------- 

K = 95
centers_95 = training_features[random_order_training_sets[:K]]

phi_train_95 = basis_gaussian(training_features, centers_95, a_vals_gaussian_def)
phi_test_95 = basis_gaussian(testing_features, centers_95, a_vals_gaussian_def)

y_hat_train_95, w_95_fxns = predictions_and_weight(phi_train_95, y_target)
y_hat_test_95 = phi_test_95.T @ w_95_fxns

training_rms_95 = np.sqrt(np.mean((y_target - y_hat_train_95)**2))
test_rms_95 = np.sqrt(np.mean((y_test - y_hat_test_95)**2))

training_rms_errors.append(training_rms_95)
test_rms_errors.append(test_rms_95)

print(phi_train_95.shape)
print(phi_test_95.shape)
print(training_rms_95)
print(test_rms_95)




# ---------------------------------------------------
# Creating the plot of training error and testing error
# against the polynomial degree used in the linear # regression:
#---------------------------------------------------
k_values = [5,15,25,35,45,55,65,75,85,95]


plt.plot(k_values, training_rms_errors, label="Training RMS")
plt.plot(k_values, test_rms_errors, label="Test RMS")

plt.xlabel("Number of Basis Functions")
plt.ylabel("RMS Error")
plt.title("Plot of RMS Errors of the Training Data and the Testing \nData against the number of Gaussian Basis Functions")

plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.legend()
plt.show()