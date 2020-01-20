import pandas as pd
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import timeit
import sys

def read_stuff(data, label):
    data_table = pd.read_csv(data,sep=",",header=None)
    labels_df = pd.read_csv(label,sep=",",header=None)

    return data_table,labels_df

def data_shape(data_table,labels_df):
    shape_of_data=data_table.shape
    shape_of_labels=labels_df.shape

    return shape_of_data,shape_of_labels

def pd_to_numpy_converter(data_table,labels_df):
    data_vector=data_table.values
    label_vector= labels_df.values

    return data_vector,label_vector

def split_data(data_table,label_df,folds=10):
    fold_rows=round(data_table.shape[0]/folds)
    data_folds=[]
    label_folds=[]
    fold=0
    for element in range (0,folds):
        data_folds_1=data_table[fold:fold+fold_rows]
        label_folds_1=label_df[fold:fold+fold_rows]
        fold=fold+fold_rows
        data_folds.append(data_folds_1)
        label_folds.append(label_folds_1)

    return data_folds,label_folds

def Id_matrix_generator(data_vector):
    Identity_matrix = np.identity(data_vector.shape[1])

    return Identity_matrix

def w_calculator(data_vector,label_vector,Identity_matrix,lambda_parameter):

    product = np.add(np.multiply(lambda_parameter, Identity_matrix),np.dot(np.transpose(data_vector), data_vector))
    inverse_matrix=np.linalg.inv(product)
    w_vector = np.dot(inverse_matrix, np.dot(np.transpose(data_vector), label_vector))

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

def graph_plotting(lambda_array,mse_temp):
    plt.xlabel('lambda_parameter')
    plt.ylabel('mean square error')
    plt.title('mse VS lambda_parameters')
    plt.plot(lambda_array, mse_temp,'r')
    plt.show()

def main_function_1(train_data,train_labels,test_data,test_labels):
    data_table, labels_df = read_stuff(train_data, train_labels)
    data_table_test,labels_df_test=read_stuff(test_data,test_labels)
    data_vector, label_vector=pd_to_numpy_converter(data_table,labels_df)
    test_vector, test_label_vector = pd_to_numpy_converter(data_table_test,labels_df_test)
    data_vector_split, label_vector_split=split_data(data_vector,label_vector,folds=10)
    Identity_matrix = Id_matrix_generator(data_vector)

    lambda_array=[]

    mse_temp=[]

    for lambda_parameter in range(0, 150):

        arb=0

        for i in range(0,10):
            data_vector_temp_1=np.vstack(([data_vector_split[j] for j in range(0,10) if i!=j]))
            label_vector_temp_1=np.vstack(([label_vector_split[j] for j in range(0,10) if i!=j]))
            w_vector = w_calculator(data_vector_temp_1, label_vector_temp_1, Identity_matrix, lambda_parameter)
            test_predicted_values = predictor(w_vector, data_vector_split[i])
            mse = mse_calculator(test_predicted_values, label_vector_split[i])
            arb+=mse
        mse_temp.append(arb/10)
        lambda_array.append(lambda_parameter)
    min_error=min(mse_temp)
    location_of_minimum_error=mse_temp.index(min(mse_temp))
    correspnding_lambda_parameter=lambda_array[location_of_minimum_error]
    w_vector = w_calculator(data_vector_temp_1, label_vector_temp_1, Identity_matrix, correspnding_lambda_parameter)
    test_predicted_values = predictor(w_vector, test_vector)
    mse = mse_calculator(test_predicted_values, test_label_vector)
    print("min error is", min_error, " and its lambda value is ", location_of_minimum_error)
    return min_error, correspnding_lambda_parameter


if __name__=="__main__":

    t1=timeit.default_timer()
    main_function_1(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    t2=timeit.default_timer()
    t=t2-t1
    print("time",t)
