import pandas as pd
import numpy as np
from random import randrange
import timeit
import sys

def read_stuff(data, label):
    data_table = pd.read_csv(data,header=None)
    labels_df = pd.read_csv(label,header=None)

    return data_table,labels_df

def pd_to_numpy_converter(data_table,labels_df):
    data_vector=data_table.values
    label_vector= labels_df.values

    return data_vector,label_vector

def covar_matrix(data_vector,b_value):
    co_matrix_1=np.dot(np.transpose(data_vector),data_vector)
    lambda_matrix=np.multiply(b_value,co_matrix_1)
    return lambda_matrix


def Id_matrix_generator(data_vector):
    Identity_matrix = np.identity(data_vector.shape[1])
    return Identity_matrix

def sn_matrix(a_value,Identity_matrix,b_value,data_vector):
    alpha_multiply=np.multiply(a_value,Identity_matrix)
    lambda_matrix=covar_matrix(data_vector,b_value)
    final_sum=np.add(alpha_multiply,lambda_matrix)
    final_sn=np.linalg.inv(final_sum)

    return final_sn

def mn_matrix(final_sn,b_value,data_vector,label_vector):
    initial_product= final_sn.dot(np.transpose(data_vector)).dot(label_vector)
    final_mn=np.multiply(b_value,initial_product)

    return final_mn

def get_gamma(lambda_matrix,a_value):
    gamma_value=0
    eigen_lambda_matrix=np.linalg.eigvals(lambda_matrix)
    for x in eigen_lambda_matrix:
                gamma_value+=(x/(a_value+x))

    return gamma_value

def reiterated_a_and_b_value(gamma_value,final_mn,data_vector,label_vector):
    denominator_product=np.dot(np.transpose(final_mn),final_mn)
    new_a_value=gamma_value/denominator_product
    product_1=np.dot(data_vector,final_mn)
    difference_predicted_actual=np.square(np.subtract(label_vector,product_1))
    b_inverse=(np.sum(difference_predicted_actual))/(label_vector.shape[0] - gamma_value)
    b_value_new=(1/b_inverse)
    return new_a_value,b_value_new

def predictor(w_vector,data_vector):
    predicted_values=np.dot(data_vector,w_vector)

    return predicted_values

def mse_calculator(predicted_values,label_vector):

    error= np.subtract(predicted_values,label_vector)

    square_error=np.square(error)

    sum_of_square_error=np.sum(square_error)
    mse = sum_of_square_error/label_vector.shape[0]
    return mse

def main_function_1(data,label,test,t_label):
    a_value = randrange(1, 10)
    b_value = randrange(1, 10)
    data_table, labels_df=read_stuff(data, label)
    test_data, test_label = read_stuff(test, t_label)

    data_vector, label_vector=pd_to_numpy_converter(data_table,labels_df)
    test_vector, test_label_vector = pd_to_numpy_converter(test_data,test_label)
    final_mse=[]
    for iterations in range(0,100):
        Identity_matrix=Id_matrix_generator(data_vector)
        lambda_matrix=covar_matrix(data_vector,b_value)
        final_sn=sn_matrix(a_value,Identity_matrix,b_value,data_vector)
        final_mn=mn_matrix(final_sn,b_value,data_vector,label_vector)
        gamma_value=get_gamma(lambda_matrix,a_value)
        a_value, b_value=reiterated_a_and_b_value(gamma_value,final_mn,data_vector,label_vector)
        predicted_values=predictor(final_mn,data_vector)
        mse_error=mse_calculator(predicted_values,label_vector)
        final_mse+=[mse_error]
        if iterations>=1:
            change_in_error=((final_mse[iterations]-final_mse[iterations-1])/final_mse[iterations-1])*100

            if (change_in_error*100)<0.000001:
                print("the final alpha, beta values are ",a_value," ",b_value)
                print("the corrresponding lambda value is",a_value/b_value)
                print("similarly, the gamma value ",gamma_value)
                print("number of iterations",iterations)


                break
    predicted_values = predictor(final_mn, test_vector)
    mse_error = mse_calculator(predicted_values, test_label_vector)
    print(mse_error)
    return change_in_error

if __name__=="__main__":
    print("please enter the file names in order")
    print("1.train data","\n","2.train label")
    t1=timeit.default_timer()
    main_function_1(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    t2=timeit.default_timer()
    t=t2-t1
    print("TIME ",t)