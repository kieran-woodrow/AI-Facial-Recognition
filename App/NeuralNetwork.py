
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
if len(sys.argv) != 6:
    print(
        "Usage: python ex.py test_set.npy test_label.npy train_set.npy train_label.npy <learning rate>"
    )
    sys.exit(1)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# NORMALIZE THE DATA AND LOAD IT IN
test_set = np.load(sys.argv[1])
test_label = np.load(sys.argv[2])
train_set = np.load(sys.argv[3])
train_label = np.load(sys.argv[4])
learn_rate = float(sys.argv[5])

y_train = []
y_train_b = []
for i in range(train_label.shape[0]):
    for j in range(10):
        y_train.append(0)
    y_train_b.append(y_train)
    y_train = []


y_train_b = np.array(y_train_b)
y_train_b.shape

for i in range(train_label.shape[0]):
    y_train_b[i][train_label[i] - 1] = 1

y_train = y_train_b
y_train = y_train.T
y_train.shape

y_train_a = []
y_train_b = []
for i in range(test_label.shape[0]):
    for j in range(10):
        y_train_a.append(0)
    y_train_b.append(y_train_a)
    y_train_a = []


y_train_b = np.array(y_train_b)
y_train_b.shape

for count in range(test_label.shape[0]):
    y_train_b[count][test_label[count] - 1] = 1

y_test = y_train_b
y_test = y_test.T
#----------------------------------------------------------------------------------
#THERE THREE FUNCTIONS ARE USED TO MAKE VALUES BETWEEN 0 AND 1
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def grad_tanh(x):
    return 1 - np.tanh(x) ** 2
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#THIS FUNCTION FINDS THE NUMBER OF COMPONENTS
def numberOfComponents(people_list):
    pca = PCA().fit(people_list.data)
    explained_var = pca.explained_variance_ratio_.cumsum()
    explained_numpy = np.array(explained_var)

    a = 0
    for x in explained_numpy:
        a = a + 1

        if x > 0.95:
            print(" The top K number of components are: ", a, " and variance is ", x)
            break
    # print(a)
    return a
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#THIS FUNCTION WILL PORINT ALL THE INFO TO A FILE
def printStuffToFile(W_1, b_1, W_2, b_2, train_a, test_a):
    f = open("output.txt", "w")
    print_string = "W_1 \n [" + np.array2string(W_1) + "] \n \n"
    f.write(print_string)
    print_string = "b_1 \n [" + np.array2string(b_1) + "]\n \n"
    f.write(print_string)
    print_string = "W_2 \n [" + np.array2string(W_2) + "]\n \n"
    f.write(print_string)
    print_string = "b_2 \n [" + np.array2string(b_2) + "]\n \n"
    f.write(print_string)
    print_string = "Training set accuracy : " + str(train_a) + "\n \n"
    f.write(print_string)
    print_string = "Testing set accuracy : " + str(test_a) + "\n \n"
    f.write(print_string)
    f.close()

    return
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#THIS IS THE MAIN FUNCTION
def neural_network():

    pca = PCA(n_components=numberOfComponents(test_set))
    pca = pca.fit(train_set)
    X_pca_train = pca.transform(train_set)
    X_pca_test = pca.transform(test_set)
    X_pca_train.shape
    reducedTraingSet = X_pca_train
    reducedTestSet = X_pca_test
    scaler = StandardScaler()
    scaler.fit(reducedTraingSet)
    scaled_data_train = scaler.transform(reducedTraingSet)
    scaled_data_train = scaled_data_train.T
    scaler.fit(reducedTestSet)
    scaled_data_test = scaler.transform(reducedTestSet)
    scaled_data_test = scaled_data_test.T
    n_h = 250
    N = scaled_data_train.shape[1]
    D = scaled_data_train.shape[0]

    print(D)

    W_1 = np.random.randn(n_h, D) * learn_rate
    b_1 = np.random.randn(n_h, 1) * learn_rate
    W_2 = np.random.randn(10, n_h) * learn_rate
    b_2 = np.random.randn(1, 1) * learn_rate

    for iteration in range(500):
        Z_1 = np.matmul(W_1, scaled_data_train) + b_1
        A_1 = tanh(Z_1)
        Z_2 = np.matmul(W_2, A_1) + b_2
        A_2 = sigmoid(Z_2)  
        cost = (1 / N) * np.sum(
            -y_train * np.log(A_2) - (1 - y_train) * np.log(1 - A_2)
        )

        dZ_2 = A_2 - y_train
        dZ_1 = np.matmul(W_2.T, dZ_2) * grad_tanh(Z_1)
        dW_2 = (1 / N) * np.matmul(dZ_2, A_1.T)
        dW_1 = (1 / N) * np.matmul(dZ_1, scaled_data_train.T)
        db_2 = (1 / N) * np.sum(dZ_2, axis=1, keepdims=True)
        db_1 = (1 / N) * np.sum(dZ_1, axis=1, keepdims=True)
        W_1 = W_1 - learn_rate * dW_1
        W_2 = W_2 - learn_rate * dW_2
        b_1 = b_1 - learn_rate * db_1
        b_2 = b_2 - learn_rate * db_2

        Z_1 = np.matmul(W_1, scaled_data_train) + b_1
        A_1 = tanh(Z_1)
        Z_2 = np.matmul(W_2, A_1) + b_2
        A_2 = sigmoid(Z_2)
        y_train_pred = (A_2 > 0.5).astype(np.int_).flatten()
        train_accuracy = accuracy_score(y_train.flatten(), y_train_pred)

        Z_1 = np.matmul(W_1, scaled_data_test) + b_1
        A_1 = tanh(Z_1)
        Z_2 = np.matmul(W_2, A_1) + b_2
        A_2 = sigmoid(Z_2)
        y_test_pred = (A_2 > 0.5).astype(np.int_).flatten()
        test_accuracy = accuracy_score(y_test.flatten(), y_test_pred)

        print(
            "Iteration: {}".format(iteration),
            " Cost: {:02f}".format(cost),
            " Training accuracy : {:02f}".format(train_accuracy),
            " Test accuracy: {:02f}".format(test_accuracy),
        )
        
        # IF TEST ACCURACY IS MORE THAN 90 PERCENT
        if test_accuracy >= 0.90:
            if train_accuracy >= 0.90:
                print(
                    "The model Accuracy has reached 90% on both training and test data. No further improvements are required to prevent overfitting"
                )
                break

    printStuffToFile(W_1, b_1, W_2, b_2, train_accuracy, test_accuracy)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#CALL MAIN FUNCTION
neural_network()
#----------------------------------------------------------------------------------
