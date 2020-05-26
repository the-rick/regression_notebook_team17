from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
%matplotlib inline

############ Test dataset
test_data = pd.read_csv(r'Test.csv')
test_data.head()

########### Rider dataset
riders = pd.read_csv(r'Riders.csv')
riders.head()

########### Train dataset
train_data = pd.read_csv(r'Train.csv')
train_data.head()



######### merging datasets
#                                 TEST
df2 = pd.merge(riders,test_data,on = 'Rider Id',how='left')


#                                 TRAIN
df = pd.merge(riders,train_data,on = 'Rider Id',how='left')
df.head()



"""          
   Dropping vehicle type because it is always a bike
"""

def drop_vehicle_type(input_df):
    input_df = input_df.drop(["Vehicle Type"], axis=1)
    return input_df
    
df = drop_vehicle_type(df)                   # TRAIN DATA
df2 = drop_vehicle_type(df2)                  # TEST DATA



"""      ASSIGNING FEATURES AND PREDICTOR VARIABLES  """
#                               TRAIN DATA
X = df.drop(["Time from Pickup to Arrival"],axis=1)
y = df.iloc[:,-1]

#                                TEST DATA
X_pred = df.iloc[:,:]




"""      DEALING WITH MISSING VALUES               """

"""
   PRECIPITATION
   Missing values will be replaced with 0. The zero will mean that there was no precipitation 
   during that day.
"""
X["Precipitation in millimeters"] = X["Precipitation in millimeters"].fillna(0)  #TRAIN DATA
X_pred["Precipitation in millimeters"] = X_pred["Precipitation in millimeters"].fillna(0)  #TEST DATA

"""
   TEMPERATURE
   Filling NaN values with the mean of the column
"""

def impute_mean(series):
    return series.fillna(series.mean())
X["Temperature"] = round(X.Temperature.transform(impute_mean),1)     # TRAIN DATA
X_pred["Temperature"] = round(X_pred.Temperature.transform(impute_mean),1)      # TEST DATA

"""
   Drop the riders from the Rider dataset who do not have information on the train data and test data
   Number of rows will go from 21237 to 21201 in train dataset
   Number of rows will go from 7206 to 7068 in test dataset
"""

def drop_nan_rows(input_df):
    input_df = input_df.dropna(how='any', subset=['User Id'])
    return input_df
X = drop_nan_rows(X)                    # TRAIN DATA
X_pred = drop_nan_rows(X_pred)                   # TEST DATA




"""         CATEGORISING DATA AND ENCODING IT            """

"""
   RIDER ID
   
   Creating a count variable that counts the number of times each rider ID appers,
   then breaking the counts values into categorical values to reduce the number of dummy variables
"""

#                          TRAIN DATA
X["Counts"] = X.groupby("Rider Id")["Order No"].transform('count')

X["Rider"] = X["Counts"].apply(lambda x: "Busiest Rider" if x >= 150 
                                   else "Busier Rider" if 50 < x < 150 
                                   else  "Busy Rider" )

#                          TEST DATA
X_pred["Counts"] = X_pred.groupby("Rider Id")["Order No"].transform('count')

X_pred["Rider"] = X_pred["Counts"].apply(lambda x: "Busiest Rider" if x >= 150 
                                   else "Busier Rider" if 50 < x < 150 
                                   else  "Busy Rider" )

############ This method did not allow me to use a function, please check it.  ######################
############ Also not sure about my wording "Busy, Busier,Busiest"             ######################
"""
   Dropping the counts and Rider Id columns, after utilizing them
"""

X = X.drop(["Rider Id","Counts"],axis = 1)              # TRAIN DATA
X_pred = X_pred.drop(["Rider Id","Counts"],axis = 1)    # TEST DATA
X.head()



"""
  USER  ID
  
  Doing the same method we did for the rider ID variable to reduce the number of dummy variables
  A user will be a frequent user if they are returning for the second time or more,
  if a user appears once on the list, they are regarded as non-frequent
"""

def user_id_cat(input_df):
    input_df["Counts"] = input_df.groupby("User Id")["Order No"].transform('count')

    input_df["Is_user_frequent"] = input_df["Counts"].apply(lambda x: 1 if x >= 2 else 0 )
    return input_df
X = user_id_cat(X)                   # TRAIN DATA
X_pred = user_id_cat(X_pred)         # TEST DATA
"""
   Dropping the counts and User Id columns, after utilizing them
"""
X = X.drop(["User Id","Counts"],axis = 1)               # TRAIN DATA
X_pred = X_pred.drop(["User Id","Counts"],axis = 1)     # TEST DATA    

X.head(3)



"""
   ORDER NO
   
   Dropping Order No column because  ??????????   (forgot the reason why. Please assist)
"""
X = X.drop(["Order No"],axis=1)                        # TRAIN DATA
X_pred = X_pred.drop(["Order No"],axis=1)                        # TEST DATA







