import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

def read_stuff(data, label):
    data_table = pd.read_csv(data,header=None)
    labels_df = pd.read_csv(label,header=None)

    return data_table,labels_df

def pd_to_numpy_converter(data_table,labels_df):
    data_vector=data_table.values
    label_vector= labels_df.values

    return data_vector,label_vector


def sample_function(No_of_sample, data_vector,label_vector):
    index = random.sample(range(1000),No_of_sample)

    sampled_train_vector_1=[]
    sampled_train_label_vector_1=[]
    for i in index:
        sampled_train_vector_1+=[data_vector[i]]
        sampled_train_label_vector_1+=[label_vector[i]]
    sampled_train_vector=np.asarray(sampled_train_vector_1)
    sampled_train_vector.reshape(1,-1)
    sampled_train_label_vector=np.asarray(sampled_train_label_vector_1)
    sampled_train_label_vector.reshape(1,-1)

    return sampled_train_vector,sampled_train_label_vector

def Id_matrix_generator(data_vector):
    Identity_matrix = np.identity(data_vector.shape[1])

    return Identity_matrix

def w_calculator(data_vector,label_vector,Identity_matrix,lambda_parameter):
    data_vector_transpose=np.transpose(data_vector)

    product = np.add(np.multiply(int(lambda_parameter),Identity_matrix) ,np.dot(data_vector_transpose,data_vector))
    inverse_matrix=np.linalg.inv(product)

    w_vector = np.dot(inverse_matrix,np.dot(data_vector_transpose,label_vector))

    return w_vector

def predictor(w_vector, data_vector):
    predicted_values=np.dot(data_vector,w_vector)
    return predicted_values

def mse_calculator(predicted_values,label_vector):
    error= np.subtract(predicted_values,label_vector)
    square_error=np.square(error)
    sum_of_square_error=np.sum(square_error)
    mse = sum_of_square_error/label_vector.shape[0]
    return mse

def graph_plotting(sample_list,mse_list_7,mse_list_80,mse_list_130):
    plt.xlabel('size of data')
    plt.ylabel('mean square error')
    plt.title('mse VS data size')
    plt.plot(sample_list, mse_list_7,'r',label="lambda=7")
    plt.plot(sample_list, mse_list_80, 'g',label="lambda=80")
    plt.plot(sample_list, mse_list_130, 'b',label="lambda=130")
    plt.legend()
    plt.show()

def main_function_1(train_data,train_labels,test_data,test_labels,lambda_value):
    #read_train_data
    train_data_table, train_labels_df = read_stuff(train_data, train_labels)
    train_data_vector, train_label_vector = pd_to_numpy_converter(train_data_table, train_labels_df)


    # read_test_data
    test_data_table, test_labels_df = read_stuff(test_data, test_labels)
    test_data_vector, test_label_vector = pd_to_numpy_converter(test_data_table, test_labels_df)
    sample_list=[]
    for i in range(50,950,75):
        sample_list.append(i)
    mse_list_1=[]
    mse_list=[]
    for sample_number in sample_list:
        temp=0
        for iterations in range(0,15):
            sampled_train_vector, sampled_train_label_vector=sample_function(sample_number, train_data_vector,train_label_vector)
            Identity_matrix = Id_matrix_generator(sampled_train_vector)
            w_vector = w_calculator(sampled_train_vector, sampled_train_label_vector, Identity_matrix,lambda_value )
            test_predicted_values = predictor(w_vector, test_data_vector)
            test_mse_ = mse_calculator(test_predicted_values, test_label_vector)
            temp+=test_mse_
        mse_list_1.append(temp)
    for element in mse_list_1:
        mse_list.append(element/15)

    return mse_list,sample_list



mse_list_7, sample_list_7=main_function_1("train-1000-100.csv","trainR-1000-100.csv","test-1000-100.csv","testR-1000-100.csv",7)
mse_list_80, sample_list_80=main_function_1("train-1000-100.csv","trainR-1000-100.csv","test-1000-100.csv","testR-1000-100.csv", 80)
mse_list_130, sample_list_130=main_function_1("train-1000-100.csv","trainR-1000-100.csv","test-1000-100.csv","testR-1000-100.csv", 130)

graph_plotting(sample_list_7, mse_list_7, mse_list_80, mse_list_130)



