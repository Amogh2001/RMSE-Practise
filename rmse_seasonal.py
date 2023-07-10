import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import math
import scipy.stats as stats
from importlib import reload
plt=reload(plt)

import day_classify as dc
import rmse_funcs as rmfn
from conmat_classify import mat_rain_classifier
from month_classifier import Rainfall_Year
from monthly_mat_classifier import monthly_rain_classifier
import imd_nc_extractor as imd

#--------------------------------------GKVK Data-----------------------------------------

list01_11 = ['AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AN', 'AO', 'AP', 'AQ'] #include 'AM' for 2007
list12_22 = ['AR', 'AS','AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ']
list2 =['D','H', 'L', 'P', 'T','X', 'AB', 'AF', 'AJ', 'AN', 'AR']

bang_gkvk_df4 = pd.DataFrame()
bang_gkvk_df5 = pd.DataFrame()


for m in list01_11:
    bang_gkvk4 = pd.read_excel('GKVK_DailyRainfall_1.1972-2022.xlsx', usecols=m)
    bang_gkvk_df4 = pd.concat([bang_gkvk_df4, (bang_gkvk4[1:369])], axis = 0, ignore_index= True)
bang_gkvk_df4 = bang_gkvk_df4.stack().reset_index()
#print(bang_gkvk_df3.loc[110:130])

for n in list12_22:
    bang_gkvk5 = pd.read_excel('GKVK_DailyRainfall_1.1972-2022.xlsx', usecols=n)
    bang_gkvk_df5 = pd.concat([bang_gkvk_df5, (bang_gkvk5[1:369])], axis = 0, ignore_index= True)
bang_gkvk_df5 = bang_gkvk_df5.stack().reset_index()

bang_gkvk_df4.rename(columns = {0: 'Rainfall'}, inplace=True)
bang_gkvk_df5.rename(columns = {0: 'Rainfall'}, inplace=True)

bang_gkvk_df4['Rainfall'] = pd.to_numeric(bang_gkvk_df4['Rainfall'], errors = 'coerce')
df4_rarr = bang_gkvk_df4['Rainfall'].array
df4_rarr = np.insert(df4_rarr, 0, 0.0)
#df4_rarr = np.delete(df4_rarr, 4017)   #include when including 2007

bang_gkvk_df5['Rainfall'] = pd.to_numeric(bang_gkvk_df5['Rainfall'], errors = 'coerce')
df5_rarr = bang_gkvk_df5['Rainfall'].array
df5_rarr = np.insert(df5_rarr, 0, 0.0)
#df5_rarr = np.delete(df5_rarr, 3288)
#print(df1_rarr[730:740])

#--------------------------------------HAL Airport Data-----------------------------------------

list01_11 = [ 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AQ', 'AR', 'AS','AT']  #2007 is 'AP'
list12_22 = ['AU', 'AV']#, 'AW', 'AX', 'AY', 'AZ']
list2 =['D','H', 'L', 'P', 'T','X', 'AB', 'AF', 'AJ', 'AN', 'AR']

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

#--------------------------------------City Central College Data-----------------------------------------

list01_11 = [ 'AJ', 'AK', 'AL','AM', 'AN', 'AO', 'AQ', 'AR', 'AS','AT']
list12_22 = ['AU', 'AV']#, 'AW', 'AX', 'AY', 'AZ']
list2 =['D','H', 'L', 'P', 'T','X', 'AB', 'AF', 'AJ', 'AN', 'AR']

bang_ccc_df4 = pd.DataFrame()
bang_ccc_df5 = pd.DataFrame()


for m in list01_11:
    bang_ccc4 = pd.read_excel('/media/amogh01/One Touch/prac/City_Central_College_Revised_69-14.xlsx', usecols=m)
    bang_ccc_df4 = pd.concat([bang_ccc_df4, (bang_ccc4[1:368])], axis = 0, ignore_index= True)
bang_ccc_df4 = bang_ccc_df4.stack().reset_index()
#print(bang_ccc_df4[340:370])

for n in list12_22:
    bang_ccc5 = pd.read_excel('/media/amogh01/One Touch/prac/City_Central_College_Revised_69-14.xlsx', usecols=n)
    bang_ccc_df5 = pd.concat([bang_ccc_df5, (bang_ccc5[1:368])], axis = 0, ignore_index= True)
bang_ccc_df5 = bang_ccc_df5.stack().reset_index()

bang_ccc_df4.rename(columns = {0: 'Rainfall'}, inplace=True)
bang_ccc_df5.rename(columns = {0: 'Rainfall'}, inplace=True)


bang_ccc_df4['Rainfall'] = pd.to_numeric(bang_ccc_df4['Rainfall'], errors = 'coerce')
df4_rarr_ccc = bang_ccc_df4['Rainfall'].array
df4_rarr_ccc = np.insert(df4_rarr_ccc, 0, 0.0)
#df4_rarr_ccc = np.delete(df4_rarr_ccc, 4017)   #include when including 2007

bang_ccc_df5['Rainfall'] = pd.to_numeric(bang_ccc_df5['Rainfall'], errors = 'coerce')
df5_rarr_ccc = bang_ccc_df5['Rainfall'].array
df5_rarr_ccc = np.insert(df5_rarr_ccc, 0, 0.0)
#df5_rarr_ccc = np.delete(df5_rarr_ccc, 3288)

#===============================================  Finding Mean of 3 Stations ====================================

df4_gkvk = pd.DataFrame(df4_rarr)
df4_hal = pd.DataFrame(df4_rarr_hal)
df4_ccc = pd.DataFrame(df4_rarr_ccc)

comb_1 = [df4_gkvk, df4_hal]
df_com1 = pd.concat(comb_1, axis = 1, join = 'inner')

comb_2 = [df_com1, df4_ccc]
df_com2 = pd.concat(comb_2, axis = 1, join = 'inner')

df_com_sum = df_com2.sum(axis = 1)
df_com_mean = df_com_sum.div(3.0)

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
df_r_arr = pd.DataFrame(r_arr)
#============================================== RMSE ===============================================

com_mon = np.zeros(1)
imd_mon = np.zeros(1)
ccc_mon = np.zeros(1)
gkvk_mon = np.zeros(1)
hal_mon = np.zeros(1)

for num1 in range(0, 10):                
    num2 = num1 + 1
    if (num2/4 == 1 or num2/7 == 1):
        arr_com = dc.month_rain_getter(df_com_mean[(369*num1):(369*num2)], leap = True)
        com_mon = np.append(com_mon, arr_com)
        imd_arr = dc.month_rain_getter(r_arr[(369*num1):(369*num2)], leap = True)
        imd_mon = np.append(imd_mon, imd_arr)
        ccc_arr = dc.month_rain_getter(df4_rarr_ccc[(369*num1):(369*num2)], leap = True)
        ccc_mon = np.append(ccc_mon, ccc_arr)
        gkvk_arr = dc.month_rain_getter(df4_rarr[(369*num1):(369*num2)], leap = True)
        gkvk_mon = np.append(gkvk_mon, gkvk_arr)
        hal_arr = dc.month_rain_getter(df4_rarr_hal[(369*num1):(369*num2)], leap = True)
        hal_mon = np.append(hal_mon, hal_arr)
        
    else:
        arr_com = dc.month_rain_getter(df_com_mean[(369*num1):(369*num2)])
        com_mon = np.append(com_mon, arr_com)
        imd_arr = dc.month_rain_getter(r_arr[(369*num1):(369*num2)])
        imd_mon = np.append(imd_mon, imd_arr)
        ccc_arr = dc.month_rain_getter(df4_rarr_ccc[(369*num1):(369*num2)])
        ccc_mon = np.append(ccc_mon, ccc_arr)
        gkvk_arr = dc.month_rain_getter(df4_rarr[(369*num1):(369*num2)])
        gkvk_mon = np.append(gkvk_mon, gkvk_arr)
        hal_arr = dc.month_rain_getter(df4_rarr_hal[(369*num1):(369*num2)])
        hal_mon = np.append(hal_mon, hal_arr)

com_mon = np.delete(com_mon, 0)
imd_mon = np.delete(imd_mon, 0)  
ccc_mon = np.delete(ccc_mon, 0)
gkvk_mon = np.delete(gkvk_mon, 0)
hal_mon = np.delete(hal_mon, 0)


rf_seas_com = np.zeros(1)
rf_seas_gpm = np.zeros(1)
s_ccc_mon = np.zeros(1)
s_hal_mon = np.zeros(1)
s_gkvk_mon = np.zeros(1)

for num1 in range(0, 10):
    num2 = num1 + 1
    s_com_arr = dc.seasonal_getter(com_mon[(12*num1) : (12*num2)])
    rf_seas_com = np.append(rf_seas_com, s_com_arr)
    s_gpm_arr = dc.seasonal_getter(imd_mon[(12*num1) : (12*num2)])
    rf_seas_gpm = np.append(rf_seas_gpm, s_gpm_arr)
    s_ccc_arr = dc.seasonal_getter(df4_rarr_ccc[(12*num1):(12*num2)])
    s_ccc_mon = np.append(s_ccc_mon, s_ccc_arr)
    s_gkvk_arr = dc.seasonal_getter(df4_rarr[(12*num1):(12*num2)])
    s_gkvk_mon = np.append(s_gkvk_mon, s_gkvk_arr)
    s_hal_arr = dc.seasonal_getter(df4_rarr_hal[(12*num1):(12*num2)])
    s_hal_mon = np.append(s_hal_mon, s_hal_arr)

rf_seas_com = np.delete(rf_seas_com, 0)
rf_seas_gpm = np.delete(rf_seas_gpm, 0)
s_ccc_mon = np.delete(s_ccc_mon, 0)
s_hal_mon = np.delete(s_hal_mon, 0)
s_gkvk_mon = np.delete(s_gkvk_mon, 0)
#print(rf_seas_com[23])

s_com = dc.season_sum(rf_seas_com)
s_gpm = dc.season_sum(rf_seas_gpm)
s_ccc = dc.season_sum(s_ccc_mon)
s_gkvk = dc.season_sum(s_gkvk_mon)
s_hal = dc.season_sum(s_hal_mon)


gpm_arr = np.zeros(1)
week_rmse = np.zeros(1)                       
week_list = np.zeros(1)                      
week_rmse_ind = np.zeros(1)                 
                                          
for num1 in range(0, 10):                
    num2 = num1 + 1
    if (num2/4 == 1 or num2/7 == 1):
        bang_com_iter = dc.week_rain_getter(df4_gkvk[0][(369*num1):(369*num2)], leap = True)
        week_list = np.append(week_list, bang_com_iter)
        bang_com_gpm = dc.week_rain_getter(r_arr[(369*num1):(369*num2)], leap = True)
        gpm_arr = np.append(gpm_arr, bang_com_gpm)
    else:
        bang_com_iter = dc.week_rain_getter(df4_gkvk[0][(369*num1):(369*num2)])
        week_list = np.append(week_list, bang_com_iter)
        bang_com_gpm = dc.week_rain_getter(r_arr[(369*num1):(369*num2)])
        gpm_arr = np.append(gpm_arr, bang_com_gpm)
week_list = np.delete(week_list, 0)
gpm_arr = np.delete(gpm_arr, 0)



resid_arr = np.zeros(1)
rmse_arr = np.zeros(1)
rmse_arr_ind = np.zeros(1)        
         
for inc1 in range(0, 525):  #4017 when including 2007
    resid_arr = np.append(resid_arr, rmfn.residual_finder(week_list[inc1], gpm_arr[inc1]))
resid_arr = np.delete(resid_arr, 0)
print(max(resid_arr))

for inc2 in range(0, 525):  #4017 when including 2007
    rmse_arr = np.append(rmse_arr, rmfn.rmse_finder(week_list[inc2], gpm_arr[inc2]))    
print(max(rmse_arr))
rmse_arr = np.delete(rmse_arr, 0)

for inc3 in range(0, 525):   #4017 when including 2007
    rmse_arr_ind = np.append(rmse_arr_ind, rmfn.rmse_finder_ind(week_list[inc3], gpm_arr[inc3]))
rmse_arr_ind = np.delete(rmse_arr_ind, 0)
rmse_fin = math.sqrt(sum(rmse_arr_ind)/525)
print(rmse_fin)

plt.plot(resid_arr, 'g', label = 'Residual = GKVK Weekly Sum - IMERG Weekly Mean')
plt.title('Weekly Comparison using Residual of Bangalore Rain Measurements')
plt.ylabel('Residual of Weekly Rainfall (mm)')
plt.xlabel('Weeks')
plt.legend()
plt.grid()
plt.show()
