import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_stuff(data, label):
    data_table = pd.read_csv(data,header=None)
    labels_df = pd.read_csv(label,header=None)

    return data_table,labels_df

def data_shape(data_table,labels_df):
    shape_of_data=data_table.shape
    shape_of_labels=labels_df.shape

    return shape_of_data,shape_of_labels

def pd_to_numpy_converter(data_table,labels_df):
    data_vector=data_table.values
    label_vector= labels_df.values

    return data_vector,label_vector

def Id_matrix_generator(data_vector):
    Identity_matrix = np.identity(data_vector.shape[1])

    return Identity_matrix


def w_calculator(data_vector,label_vector,Identity_matrix,lambda_parameter=5):
    data_vector_transpose=np.transpose(data_vector)
    product = np.add(np.multiply(lambda_parameter,Identity_matrix) ,np.dot(data_vector_transpose,data_vector))
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

def graph_plotting(lambda_parameters,train_mse,test_mse):
    plt.xlabel('lambda_values')
    plt.ylabel('mean square error')
    plt.title('train and test error VS lambda')
    plt.plot(lambda_parameters, train_mse,'r',label='train mse')
    plt.plot(lambda_parameters, test_mse,'b',label='test mse')
    plt.legend(loc='lower right')
    plt.show()

def main_function_1(train_data,train_labels,test_data,test_labels):
    #read_train_data
    train_data_table, train_labels_df = read_stuff(train_data, train_labels)
    shape_of_train_data, shape_of_train_labels = data_shape(train_data_table, train_labels_df)
    train_data_vector, train_label_vector = pd_to_numpy_converter(train_data_table, train_labels_df)
    Identity_matrix = Id_matrix_generator(train_data_vector)

    # read_test_data
    test_data_table, test_labels_df = read_stuff(test_data, test_labels)
    shape_of_test_data, shape_of_test_labels = data_shape(test_data_table, test_labels_df)
    test_data_vector, test_label_vector = pd_to_numpy_converter(test_data_table, test_labels_df)
    train_mse_error=[]
    test_mse_error=[]
    lambda_parameters=[]
    for lambda_v in range(0,150):
        # get w
        w_vector = w_calculator(train_data_vector, train_label_vector, Identity_matrix, lambda_v)
        training_predicted_values=predictor(w_vector, train_data_vector)
        test_predicted_values=predictor(w_vector, test_data_vector)
        training_mse=mse_calculator(training_predicted_values,train_label_vector)
        train_mse_error+=[training_mse]
        test_mse=mse_calculator(test_predicted_values,test_label_vector)
        test_mse_error+=[test_mse]
        lambda_parameters.append(lambda_v)

    graph_plotting(lambda_parameters, train_mse_error, test_mse_error)


if __name__=="__main__":
    print("please enter the file names in order")
    print("1.train data","\n","2.train label","\n","3.test data","\n","4.test label","\n")
    main_function_1(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
