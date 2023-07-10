import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
import netCDF4 as nc
import math
import scipy.stats as stats

import day_classify as dc
#from rainfall_sep import Rainfall_Sep
#from rain_classify import rain_classifier
from conmat_classify import mat_rain_classifier
from month_classifier import Rainfall_Year
from monthly_mat_classifier import monthly_rain_classifier
import rmse_funcs as rmfn
#Region 77.4684E, 12.7098N, 77.7129E, 13.1081N

#--------------------------------------HAL Airport Data-----------------------------------------

list01_11 = [ 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AQ', 'AR', 'AS','AT']  #2007 is 'AP'
list12_22 = ['AU', 'AV']#, 'AW', 'AX', 'AY', 'AZ']

bang_hal_df4 = pd.DataFrame()
bang_hal_df5 = pd.DataFrame()

for m in list01_11:
    bang_hal4 = pd.read_excel('HAL_Revised_1969.xlsx', usecols=m)
    bang_hal_df4 = pd.concat([bang_hal_df4, (bang_hal4[1:369])], axis = 0, ignore_index= True)
bang_hal_df4 = bang_hal_df4.stack().reset_index()
#print(bang_hal_df3.loc[110:130])

for n in list12_22:
    bang_hal5 = pd.read_excel('HAL_Revised_1969.xlsx', usecols=n)
    bang_hal_df5 = pd.concat([bang_hal_df5, (bang_hal5[1:368])], axis = 0, ignore_index= True)
bang_hal_df5 = bang_hal_df5.stack().reset_index()

bang_hal_df4.rename(columns = {0: 'Rainfall'}, inplace=True)
bang_hal_df5.rename(columns = {0: 'Rainfall'}, inplace=True)


bang_hal_df4['Rainfall'] = pd.to_numeric(bang_hal_df4['Rainfall'], errors = 'coerce')
df4_rarr_hal = bang_hal_df4['Rainfall'].array
df4_rarr_hal = np.insert(df4_rarr_hal, 0, 0.0)
#df4_rarr_hal = np.delete(df4_rarr_hal, 4017)

bang_hal_df5['Rainfall'] = pd.to_numeric(bang_hal_df5['Rainfall'], errors = 'coerce')
df5_rarr_hal = bang_hal_df5['Rainfall'].array
df5_rarr_hal = np.insert(df5_rarr_hal, 0, 0.0)
#df5_rarr_hal = np.delete(df5_rarr_hal, 3288)

#print(df4_rarr_hal[350:367])

#----------------------------------- IMERG Data ------------------------------------

list_year = (2002, 2003, 2004, 2005, 2006, 2008, 2009, 2010, 2011)

r_2001 = Rainfall_Year('Year_IMERG_2001.nc', leap = False)
r_2001.monthly_mean()
r_arr01 = np.array(r_2001.monthly_rain)
r_arr = np.array(r_arr01)

for inc_imerg in list_year:
    if inc_imerg % 4 == 0:
        r_imerg = Rainfall_Year('Year_IMERG_' + str(inc_imerg) + '.nc', leap = True)
        r_imerg.monthly_mean()
        r_arr_imerg = np.array(r_imerg.monthly_rain)
        r_arr = np.append(r_arr, r_arr_imerg)

    else:
        r_imerg = Rainfall_Year('Year_IMERG_' + str(inc_imerg) + '.nc')
        r_imerg.monthly_mean()
        r_arr_imerg = np.array(r_imerg.monthly_rain)
        r_arr = np.append(r_arr, r_arr_imerg)
#print(r_arr)


#======================== Finding RMSE =====================================

rmse_arr = np.zeros(1)
rmse_arr_ind = np.zeros(1)
resid_arr = np.zeros(1)
gpm_arr = np.zeros(1)
week_rmse = np.zeros(1)                       
week_list = np.zeros(1)                      
week_rmse_ind = np.zeros(1)                 
                                          
for num1 in range(0, 10):                
    num2 = num1 + 1
    if (num2/4 == 1 or num2/7 == 1):
        bang_hal4_iter = dc.week_rain_getter(df4_rarr_hal[(369*num1):(369*num2)], leap = True)
        week_list = np.append(week_list, bang_hal4_iter)
        bang_hal4_gpm = dc.week_rain_getter(r_arr[(369*num1):(369*num2)], leap = True)
        gpm_arr = np.append(gpm_arr, bang_hal4_gpm)
    else:
        bang_hal4_iter = dc.week_rain_getter(df4_rarr_hal[(369*num1):(369*num2)])
        week_list = np.append(week_list, bang_hal4_iter)
        bang_hal4_gpm = dc.week_rain_getter(r_arr[(369*num1):(369*num2)])
        gpm_arr = np.append(gpm_arr, bang_hal4_gpm)
week_list = np.delete(week_list, 0)
gpm_arr = np.delete(gpm_arr, 0)


for inc1 in range(0, 525):  #4017 when including 2007
    resid_arr = np.append(resid_arr, rmfn.residual_finder(week_list[inc1], gpm_arr[inc1]))
resid_arr = np.delete(resid_arr, 0)
print(max(resid_arr))

for inc2 in range(0,525):  #4017 when including 2007
    rmse_arr = np.append(rmse_arr, rmfn.rmse_finder(week_list[inc2], gpm_arr[inc2]))    
print(max(rmse_arr))
rmse_arr = np.delete(rmse_arr, 0)

for inc3 in range(0, 525):   #4017 when including 2007
    rmse_arr_ind = np.append(rmse_arr_ind, rmfn.rmse_finder_ind(week_list[inc3], gpm_arr[inc3]))
rmse_arr_ind = np.delete(rmse_arr_ind, 0)
rmse_fin = math.sqrt(sum(rmse_arr_ind)/525)
print(rmse_fin)

gpm_df = pd.DataFrame(gpm_arr)
week_df = pd.DataFrame(week_list)
gpm_df.rename(columns = {0:'Rainfall'}, inplace =True)
week_df.rename(columns = {0: 'Rainfall'}, inplace = True)


#===================================== Plotting ==================================


gpm_df.sort_values(by = ['Rainfall'], inplace = True)
week_df.sort_values(by = ['Rainfall'], inplace = True)

gpm_mean = np.mean(gpm_df['Rainfall'])
week_mean = np.mean(week_df['Rainfall'])

gpm_std = np.std(gpm_df['Rainfall'])
week_std = np.std(week_df['Rainfall'])

gpm_pdf = stats.norm.pdf(gpm_df['Rainfall'], gpm_mean, gpm_std)
week_pdf = stats.norm.pdf(week_df['Rainfall'], week_mean, week_std)

plt.plot(resid_arr, 'g', label = 'Residual = Station - IMERG' )
#plt.scatter(gpm_arr, week_list)

#plt.plot(gpm_df['Rainfall'], gpm_pdf, 'g' , label = 'IMERG')
#plt.plot(week_df['Rainfall'], week_pdf, 'r' , label = 'HAL Station')

#plt.hist(gpm_df['Rainfall'])

plt.title('PDF of HAL v/s IMERG')
plt.xlabel('Weekly Rainfall (mm)')
plt.ylabel('Probability')
plt.legend()
plt.grid()
plt.show()
