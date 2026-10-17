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

# The following code imports the Boston data set from a .txt file, skipping the
# first 22 lines of description
with open('boston.txt', 'r') as f:
    lines = f.readlines()[22:]

# Get every individual number from the .txt fileir
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

def basis_gaussian(data, centers ,a):

    # Make data be in a numpy array format:
    data = np.asarray(data)
    centers = np.asarray(centers)


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
            distance_2 = (np.linalg.norm(data[n] - centers[k])) ** 2

            phi_matrix[k, n] = np.exp(a * distance_2)
         
    return phi_matrix

def predictions_and_weight(phi_matrix, y_target):

    w = np.linalg.pinv(phi_matrix.T) @ y_target
    y_hat = phi_matrix.T @ w

    return y_hat, w

# Main part of the code where it calls on the previous basis_gaussian function and the 
# predictions_and_weight function

# QUESTION - WHAT IS A???????????????????????????????????
# NOTE IS A THE FOLLOWING:
# sigma = 10
# a = -1 / (2 * sigma**2)

# Arbitrarily chose -1 for the value of 'a' just to make sure the code is functioning.
a = -1
random_order_training_sets = np.random.permutation(TRAINING_SETS)

training_rms_errors = []
test_rms_errors = []

# ---------------------------------------------------
# Linear regression with 5  basis functions
# --------------------------------------------------- 

K = 5
centers_5 = training_features[random_order_training_sets[:K]]

phi_train_5 = basis_gaussian(training_features, centers_5, a)
phi_test_5 = basis_gaussian(testing_features, centers_5, a)

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

phi_train_15 = basis_gaussian(training_features, centers_15, a)
phi_test_15 = basis_gaussian(testing_features, centers_15, a)

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

phi_train_25 = basis_gaussian(training_features, centers_25, a)
phi_test_25 = basis_gaussian(testing_features, centers_25, a)

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

phi_train_35 = basis_gaussian(training_features, centers_35, a)
phi_test_35 = basis_gaussian(testing_features, centers_35, a)

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

phi_train_45 = basis_gaussian(training_features, centers_45, a)
phi_test_45 = basis_gaussian(testing_features, centers_45, a)

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

phi_train_55 = basis_gaussian(training_features, centers_55, a)
phi_test_55 = basis_gaussian(testing_features, centers_55, a)

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

phi_train_65 = basis_gaussian(training_features, centers_65, a)
phi_test_65 = basis_gaussian(testing_features, centers_65, a)

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

phi_train_75 = basis_gaussian(training_features, centers_75, a)
phi_test_75 = basis_gaussian(testing_features, centers_75, a)

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

phi_train_85 = basis_gaussian(training_features, centers_85, a)
phi_test_85 = basis_gaussian(testing_features, centers_85, a)

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

phi_train_95 = basis_gaussian(training_features, centers_95, a)
phi_test_95 = basis_gaussian(testing_features, centers_95, a)

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



# print("Max phi test K=5:", np.max(phi_test_5))
# print("Max phi test K=15:", np.max(phi_test_15))
# print("Max phi test K=25:", np.max(phi_test_25))

def ridge_regression(rows, Phi, y_val, hp_lambda):
    I = np.eye(rows)

    w_fold = (np.linalg.pinv(Phi @ Phi.T + hp_lambda * I)) @ Phi @ y_val

    return w_fold


training_features = data[:TRAINING_SETS, 0:13]
y_target = data[:TRAINING_SETS,13]

testing_features = data[TRAINING_SETS:, 0:13]
y_test = data[TRAINING_SETS:, 13]

# It seems that just 'lambda' is a special keyword, so its now called 
# 'hp_lambda' for hyperparameter lambda. (not Harry Potter)
def gaussian_regression_reg(hp_lambda, centers, a):
    # The following regularizing function works on all 13 features of the dataset. 

    folds_data = []
    folds_target = []

    lambda_rms = 0

    # The following loop splits up the training data into 10 different folds
    for i in range(10):
        folds_data.append(training_features[(i*10):(i*10+10)])
        folds_target.append(y_target[(i*10):(i*10+10)])

    
    for i in range(10):
        # The following 2 arrays will hold 90 datapoints
        folded_train_data = []
        folded_train_y = []

        for k in range(10):
                # Here, the kth fold is the test data, and the remaining 9 folds act
                # as the training data.
                if k == i:
                    continue
                else:
                    folded_train_data.extend(folds_data[k])
                    folded_train_y.extend(folds_target[k])
    
        #The following generates a Phi function matrix off of training data (90 datasets)
        phi_train_per_fold = basis_gaussian(folded_train_data, centers, a)
        w_fold = ridge_regression(phi_train_per_fold.shape[0], phi_train_per_fold, folded_train_y, hp_lambda)
    
        #The following generates a Phi function matrix off of our test data fold (10 datasets)
        phi_validation = basis_gaussian(folds_data[i], centers, a)
        y_hat_validation = phi_validation.T @ w_fold
    
                
        #Calculating the validation RMS values:
        rms_validation = np.sqrt(np.mean((folds_target[i] - y_hat_validation)**2))
        lambda_rms += rms_validation
    
    lambda_rms /= 10
    return lambda_rms    


lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
rms_of_lambdas = []

for hp_lambda in lambdas:           # I THINK that the 95 functions has the best performance, so thats what we're optimizing lambda for.
    rms = gaussian_regression_reg(hp_lambda, centers_95, -1)
    rms_of_lambdas.append(rms)
    print("Lambda:", hp_lambda)
    print("The corresonding cross-validated RMS is: ", rms)

print(gaussian_regression_reg(0,centers_5,-1))