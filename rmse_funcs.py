import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import math

import day_classify as dc
#from rainfall_sep import Rainfall_Sep
#from rain_classify import rain_classifier
from conmat_classify import mat_rain_classifier
from month_classifier import Rainfall_Year
from monthly_mat_classifier import monthly_rain_classifier

def residual_finder(y_obs, y_pred):
    residual = y_obs - y_pred
    return residual

def euclidean_norm_mat(matr_a):
    sum_sub_list = []
    for i in range(0,len(matr_a)):
        for j in range(0,len(matr_a)):
            sum_sub_list.append(pow(matr_a[i,j], 2))
    sum_list = sum(sum_sub_list)
    euclid_norm = math.sqrt(sum_list)
    return euclid_norm

def euclidean_norm_mat_1(matr_a):
    sum_sub_list = []
    for i in range(0,len(matr_a)):
        sum_sub_list.append(pow(matr_a[i], 2))
    sum_list = sum(sum_sub_list)
    euclid_norm = math.sqrt(sum_list)
    return euclid_norm

def euclidean_norm(num):
    euc_list = []
    euc_list.append(pow(num, 2))
    #euc_list.append(pow(y_pred, 2))
    euc_sum = sum(euc_list)
    euclid_norm = math.sqrt(euc_sum)
    return euclid_norm
        
def rmse_finder(y_obs, y_pred):
    rmse = math.sqrt(pow(euclidean_norm(residual_finder(y_obs, y_pred)),2)/365)   # Divide by 4017 when including 2007, 3652 when not
    return rmse

def rmse_finder_ind(y_obs, y_pred):
    rmse = pow(euclidean_norm(residual_finder(y_obs, y_pred)),2)
    return rmse

def rmse_finder_ind_mat(y_obs, y_pred):
    rmse = pow(euclidean_norm_mat_1(residual_finder(y_obs, y_pred)),2)
    return rmse