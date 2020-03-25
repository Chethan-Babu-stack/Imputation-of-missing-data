# -*- coding: utf-8 -*-
"""
Spyder Editor by Chethan

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics  as st
from scipy import stats
import math

# Import the Dataset which has no missing Data
original_no_missing = pd.read_csv('pisma_original.txt',sep='	',header=None)

size = original_no_missing.size

row = size/9


# Creating key of the dataset
key = []
for x in range(int(row)):
    key.append(x)
    
# Append the key's column to the original DataSet
original_no_missing[9] = key

# Read the Data with few missing values
original_with_missing = pd.read_csv('pisma_10_per_nan.txt',sep='	',header=None)

original_with_missing[9] = key

master_data = original_no_missing   # Original Dataset with no missing

master_data_with_missing = original_with_missing  # Original Dataset with missing data

# Defining the columns 
original_no_missing.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class', 'key']

master_data.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class', 'key']


master_data_with_missing.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class', 'key']

###################################

# Checking the Dataframe
print(master_data.head())
print(master_data.dtypes)
print(master_data.shape) #(767, 9)

# Count the number of missing data
count_no_of_missing_data = master_data_with_missing.isna().sum()
print('The Total Number of Mising Data:',count_no_of_missing_data)

temp_master_data = master_data_with_missing


# Store the missing data
miss_data = temp_master_data[pd.isnull(temp_master_data).any(axis=1)]

# Eliminate those rows which have missing data
data_x = master_data_with_missing.dropna()
master_data_copy = data_x

######################################################################################

# Selected features with the help of max_revelance for clustering
Selected_Features_mc = master_data_copy.iloc[:, [2, 5, 8]].values 

####################################################complteing feature selection###

# Hierarchical clustering
import scipy.cluster.hierarchy as sch
dendrogram_mc = sch.dendrogram(sch.linkage(Selected_Features_mc, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Paitent')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc_mc = hc.fit_predict(Selected_Features_mc)


# Visualising the clusters
plt.scatter(Selected_Features_mc[y_hc_mc == 0, 0], Selected_Features_mc[y_hc_mc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(Selected_Features_mc[y_hc_mc == 1, 0], Selected_Features_mc[y_hc_mc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Selected_Features_mc[y_hc_mc == 2, 0], Selected_Features_mc[y_hc_mc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

###########################completing clustering###########################

master_data_copy[11] = y_hc_mc


master_data_copy.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class', 'key', 'cluster']



###########################################completing clustering ########################
## 3 cluster/Group created
group1_hc = master_data_copy.groupby(['cluster']).get_group(0)

group2_hc = master_data_copy.groupby(['cluster']).get_group(1)

group3_hc = master_data_copy.groupby(['cluster']).get_group(2)

########################################completing Grouping"#################################
##################################################completeing of all groups##################
# Find the minimum distance
df1= miss_data
x_dis_list = []
x_dist_sq = []
for m in range(435):
    x_small = 9999
    df1 = miss_data.iloc[m,0:9]
    for n in range(333):
        df2 = master_data_copy.iloc[n,0:9]
        df3 = master_data_copy.iloc[n,0:11]
        x_dis = (df2 - df1)
        x_dis = x_dis.apply(pd.to_numeric, errors='coerce').fillna(-999)
        x_edu_dis = 0
        for x in range(9):
            if x_dis[x] == -999:
                continue
            #print(x_dis[x])
            x_edu_dis = x_edu_dis + x_dis[x]**2
            if x_edu_dis < 0:
                x_edu_dis = x_edu_dis * (-1)
            x_edu_dis = math.sqrt(x_edu_dis)
         
         
        if x_edu_dis < x_small:
            x_small = x_edu_dis
            x_element = df3
    x_dis_list.append(x_element) 
    x_dist_sq.append(x_small)     
        
########################################################complete minimum distance####
### Imputing missing data
from scipy import spatial

X_miss_data = miss_data
col = ['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class', 'key']

#x_temp = X_miss_data.iloc[0,0:11]


result_imp = []

for i in range(435):
    df_X_miss_data = X_miss_data.iloc[i,0:9]
    df_X_miss_data = df_X_miss_data.apply(pd.to_numeric, errors='coerce').fillna(-1)
    #result_imp = df_X_miss_data
#x_temp = X_miss_data.iloc[0,0:11]
    for x in range(9):
        if df_X_miss_data[x] == -1:
            index_temp = x
            temp_close_atribute = x_dis_list[i][index_temp]
            temp_imp_key = x_dis_list[i][9]
            temp_imp_cluster = x_dis_list[i][10]
            
            if temp_imp_cluster == 1:
                group1_hc_col_max = group1_hc[col[x]].max()
                group1_hc_col_min = group1_hc[col[x]].min()
                group1_hc_col_sd = st.stdev(group1_hc[col[x]])
                group1_hc_col_trim_mean = stats.trim_mean(group1_hc[col[x]], 0.25)
                
                if temp_close_atribute < group1_hc_col_trim_mean:
                    Temp_max = group1_hc_col_trim_mean
                    Temp_min = group1_hc_col_min 
                    dataSetI = df_X_miss_data.values
                    dataSetII = x_dis_list[i].values
                    dataSetI = dataSetI[0:9]
                    dataSetII = dataSetII[0:9]
                    result_simlar = spatial.distance.cosine(dataSetI, dataSetII) 
                    X1_impu = result_simlar*temp_close_atribute
                    #X2_impu = (group1_hc_col_trim_mean-temp_close_atribute)*result_simlar
                    X2_impu = ((group1_hc_col_max-group1_hc_col_min)/temp_close_atribute)*result_simlar
                    X1_impu = X1_impu + X2_impu
                    
                else:
                    Temp_min = group1_hc_col_trim_mean
                    Temp_max = group1_hc_col_max
                    dataSetI = df_X_miss_data.values
                    dataSetII = x_dis_list[i].values
                    dataSetI = dataSetI[0:9]
                    dataSetII = dataSetII[0:9]
                    result_simlar =  spatial.distance.cosine(dataSetI, dataSetII) 
                    X1_impu = result_simlar*temp_close_atribute
                    X2_impu = (temp_close_atribute - group1_hc_col_trim_mean)*result_simlar
                   # X1_impu = X1_impu + X2_impu
                        
            if temp_imp_cluster == 0:
                group1_hc_col_max = group2_hc[col[x]].max()
                group1_hc_col_min = group2_hc[col[x]].min()
                group1_hc_col_sd = st.stdev(group2_hc[col[x]])
                group1_hc_col_trim_mean = stats.trim_mean(group2_hc[col[x]], 0.25)
                
                if temp_close_atribute < group1_hc_col_trim_mean:
                    Temp_max = group1_hc_col_trim_mean
                    Temp_min = group1_hc_col_min 
                    dataSetI = df_X_miss_data.values
                    dataSetII = x_dis_list[i].values
                    dataSetI = dataSetI[0:9]
                    dataSetII = dataSetII[0:9]
                    result_simlar = spatial.distance.cosine(dataSetI, dataSetII) 
                    X1_impu = result_simlar*temp_close_atribute
                    #X2_impu = (group1_hc_col_trim_mean-temp_close_atribute)*result_simlar
                    X2_impu = ((group1_hc_col_max-group1_hc_col_min)/temp_close_atribute)*result_simlar
                    X1_impu = X1_impu + X2_impu
                else:
                    Temp_min = group1_hc_col_trim_mean
                    Temp_max = group1_hc_col_max
                    dataSetI = df_X_miss_data.values
                    dataSetII = x_dis_list[i].values
                    dataSetI = dataSetI[0:9]
                    dataSetII = dataSetII[0:9]
                    result_simlar = spatial.distance.cosine(dataSetI, dataSetII) 
                    X1_impu = result_simlar*temp_close_atribute
                    X2_impu = (temp_close_atribute - group1_hc_col_trim_mean)*result_simlar
                    #X1_impu = X1_impu + X2_impu
                            
            if temp_imp_cluster == 2:
                group1_hc_col_max = group3_hc[col[x]].max()
                group1_hc_col_min = group3_hc[col[x]].min()
                group1_hc_col_sd = st.stdev(group3_hc[col[x]])
                group1_hc_col_trim_mean = stats.trim_mean(group3_hc[col[x]], 0.25)
                
                if temp_close_atribute < group1_hc_col_trim_mean:
                    Temp_max = group1_hc_col_trim_mean
                    Temp_min = group1_hc_col_min 
                    dataSetI = df_X_miss_data.values
                    dataSetII = x_dis_list[i].values
                    dataSetI = dataSetI[0:9]
                    dataSetII = dataSetII[0:9]
                    result_simlar =  spatial.distance.cosine(dataSetI, dataSetII) 
                    X1_impu = result_simlar*temp_close_atribute
                    X2_impu = ((group1_hc_col_max-group1_hc_col_min)/temp_close_atribute)*result_simlar
                    X1_impu = X1_impu + X2_impu
                else:
                    Temp_min = group1_hc_col_trim_mean
                    Temp_max = group1_hc_col_max
                    dataSetI = df_X_miss_data.values
                    dataSetII = x_dis_list[i].values
                    dataSetI = dataSetI[0:9]
                    dataSetII = dataSetII[0:9]
                    result_simlar =  spatial.distance.cosine(dataSetI, dataSetII) 
                    X1_impu = result_simlar*temp_close_atribute
                    X2_impu = (temp_close_atribute - group1_hc_col_trim_mean)*result_simlar
                    #X1_impu = X1_impu + X2_impu
                        
                            
            df_X_miss_data[x] = X1_impu        
    
    result_imp.append(df_X_miss_data)                    
                    
                    
df_result_impu = pd.DataFrame(result_imp)                    


common = original_no_missing.index.intersection(miss_data.index)
store_data_compare = original_no_missing.loc[common]
store_data_compare = store_data_compare.drop('key', 1)



################# Accurcy rate##################

from sklearn.metrics import mean_squared_error
rms_total = np.sqrt(mean_squared_error(df_result_impu, store_data_compare))
print("The error is", rms_total)

rms_pregn = np.sqrt(mean_squared_error(df_result_impu['No_pregnant'],store_data_compare['No_pregnant']))

rms_Plasma_glucose = np.sqrt(mean_squared_error(df_result_impu['Plasma_glucose'],store_data_compare['Plasma_glucose']))

rms_Blood_pres = np.sqrt(mean_squared_error(df_result_impu['Blood_pres'],store_data_compare['Blood_pres']))

rms_Skin_thick = np.sqrt(mean_squared_error(df_result_impu['Skin_thick'],store_data_compare['Skin_thick']))

rms_Serum_insu = np.sqrt(mean_squared_error(df_result_impu['Serum_insu'],store_data_compare['Serum_insu']))

rms_Class = np.sqrt(mean_squared_error(df_result_impu['Class'],store_data_compare['Class']))


total = (rms_pregn + rms_Plasma_glucose + rms_Blood_pres + rms_Skin_thick + rms_Serum_insu + rms_Class)/6

    
ax = store_data_compare.plot()
df_result_impu.plot(ax=ax)   
    
    
