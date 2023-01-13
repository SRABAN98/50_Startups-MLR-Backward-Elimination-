import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"C:\Users\dell\Documents\DATA SCIENCE,AI & ML\14th,15th\mlr\50_Startups.csv")


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]


x = pd.get_dummies(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LinearRegression
regresssor = LinearRegression()
regresssor.fit(x_train,y_train)
y_pred = regresssor.predict(x_test)


import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1) 


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt = x[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt = x[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt = x[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


x_opt = x[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


# As 0th index is for constant value..so we will 
#consider 1st index i.e R&D for the consideration for the money spend ....
