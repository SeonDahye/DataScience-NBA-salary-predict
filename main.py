import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.decomposition import PCA
import openpyxl
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

# #########################
# #
# # 1. Data Curation
# #
# #########################


df1 = pd.read_csv("Seasons_Stats.csv")
df2 = pd.read_excel("Player - Salaries per Year (1990 - 2017).xlsx")
df3 = pd.read_excel("NBA All Star Games.xlsx")
df1.drop(['Unnamed: 0'], axis=1, inplace=True)  # unnamed columns are not needed
df1.dropna(subset=['Year'], inplace=True)  # get rid of null year value
print("after get rid of the year null value")
print(df1.isna().sum()) # find the nan value
print(df1.info())

# #########################
# #
# # 2. Data Preprocessing
# #
# #########################


df1['Year'] = df1['Year'].astype("int64")  # change the year type to int The original is float
df1 = df1[df1['Year'] > 1990]  # get rid of the old data because salary data starts at 1990
print("After 1990~2017 data")
print(df1)
print(df1.isna().sum())
# fitting season-stat data's team and salary per year's team
df1.replace('BRK', 'NJN', inplace=True)
df1.replace('CHH', 'CHA', inplace=True)
df1.replace('CHO', 'CHA', inplace=True)
df1.replace('NOK', 'NOH', inplace=True)
df1.replace('NOP', 'NOH', inplace=True)
df1.replace('WSB', 'WAS', inplace=True)
# print(df1.columns)
# print(df2.columns)
# print(df2.info())
# print(df2.isna().sum())
# print(df2)
# change the type of Season start and Season end from float to int
df2['Season Start'] = df2['Season Start'].astype("int64")
df2['Season End'] = df2['Season End'].astype("int64")
print(df2.info())  # confirm the df2 info
# for merge, rename the feature
df2.rename(columns={'Player Name': 'Player', 'Season End': 'Year', 'Team': 'Tm'}, inplace=True)
df4 = pd.merge(df1, df2, on=['Player', 'Year', 'Tm'], how="left") # merge
df4.drop(['Season Start'], axis=1, inplace=True)  # drop the trash feature
df4.drop(['blanl', 'blank2'], axis=1, inplace=True)  # drop unecessary feature
print(df4)
print("Before preprocessing")
print(df4.isna().sum())  # need to preprocessing
df4 = df4.dropna(subset=['Salary in $']) # drop the null value of salary in $
print("After preprocessing")
print(df4.isna().sum())
# before second merge
print(df4.info())
print(df3.info())
print(df3.isna().sum())  # trash columns exists drop the trash feature
df3.dropna(how="any", axis=1, inplace=True)  # get rid of the not needed column value
print(df3.info())  # check the state of df3
print(df3)
df3.drop(['NBA Draft Status', 'Nationality', 'Pos', 'HT', 'WT'], axis=1, inplace=True)  # get rid of not needed feature
print(df3)
df3.rename(columns={'Team': 'Full Team Name'}, inplace=True)  # for merge we change the column's name
df_final = pd.merge(df4, df3, on=['Player', 'Year', 'Full Team Name'],
                    how="left")  # merge with all star data and stat data
print(df_final)
# print(df_final.isna().sum())
df_final = df_final[
    df_final['Year'] > 1999]  # get Merged with All-star game list 1990~1999 is we don't have the all star data
df_final = df_final[df_final['Year'] < 2017]
print(df_final)
print(df_final.isna().sum())  # check the null value
df_final['Selection Type'] = df_final['Selection Type'].fillna(
    "Not selected")  # Not all star is not nan. So we fill the new Value not selected
print(df_final.info())  # check the state of data
print(df_final.isna().sum())
# See the index having nan value :4
print(df_final[df_final['PER'].isnull()]['Pos'])
print(df_final[df_final['ORB%'].isnull()]['Pos'])
print(df_final[df_final['DRB%'].isnull()]['Pos'])
print(df_final[df_final['TRB%'].isnull()]['Pos'])
print(df_final[df_final['AST%'].isnull()]['Pos'])
print(df_final[df_final['STL%'].isnull()]['Pos'])
print(df_final[df_final['BLK%'].isnull()]['Pos'])
print(df_final[df_final['USG%'].isnull()]['Pos'])
print(df_final[df_final['WS/48'].isnull()][
          'Pos'])
# they are same players having not null value because their index is same
# different number of nan: 28 , but both are overlapped
print(df_final[df_final['3PAr'].isnull()]['Pos'])
print(df_final[df_final['FTr'].isnull()]['Pos'])
print(df_final[df_final['FG%'].isnull()]['Pos'])
print(df_final[df_final['eFG%'].isnull()]['Pos'])
print(df_final[df_final['DRB%'].isnull()])
# There are outlier g<10 because their statistics are not dependable
df_trade = df_final.loc[df_final['G'] <= 10]
print(df_trade.shape)

print(df_final['G'].mean())  # 52.281978126485974
# game < 10 is outlier
df_final = df_final[df_final['G'] >= 10]
# Features are calculated by another feature So, we judged redundant feature. So, drop
df_final.drop(['2P%', '3P%', 'OWS', 'DWS', 'WS/48', 'FT%', 'Full Team Name', 'Register Value'
                  , 'TRB', 'FG', 'FGA'], axis=1, inplace=True)

df_final.reset_index(inplace=True)
print(df_final.isnull().sum())  # now, NAN cleared
print(df_final.info())



# for test get the year=2016 data
df_final_2016 = df_final[df_final['Year'] == 2016]
df_final_2016.reset_index(inplace=True)
# final dataframe
df_f = df_final.drop(['Player', 'index', 'Salary in $'], axis=1)
df_f_2016 = df_final_2016.drop(['Player', 'index', 'level_0', 'Salary in $'], axis=1)



#############################
# Open Source Contribution Part
#############################

# Linear Regression function
def LinearReg(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, shuffle=True)
    linearReg = LinearRegression()
    linearReg.fit(x_train, y_train)
    score = linearReg.score(x_test, y_test)
    return score

# random forest function
def RandomForestReg(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, shuffle=True)
    rfModel = RandomForestRegressor()
    rfModel.fit(x_train, y_train)
    score = rfModel.score(x_test, y_test)
    return score

# scaling
def scaling(data, scale):
    if scale == 'StandardScaling':
        scaler = StandardScaler()
    elif scale == 'RobustScaling':
        scaler = RobustScaler()
    elif scale == 'MinMaxScaling':
        scaler = MinMaxScaler()
    elif scale == 'MaxAbsScaling':
        scaler = MaxAbsScaler()
    elif scale == 'Normalizer':
        scaler = Normalizer()

    scaled_data = scaler.fit_transform(data)

    df_scaled_data = pd.DataFrame(columns=data.columns, data=scaled_data)
    return df_scaled_data

# encoding
def encoding(data, encode):
    if encode == 'OrdinalEncoding':
        oe = OrdinalEncoder()
        encoded_data = oe.fit_transform(data)
        encoded_data = pd.DataFrame(columns=data.columns, data=encoded_data)
    elif encode == 'OneHotEncoding':
        encoded_data = pd.get_dummies(data)

    return encoded_data


def compute_Score(df, scalingList=None, encodingList=None):
    df_cate = df.select_dtypes(include='object')
    df_nume = df.select_dtypes(exclude='object')

    size = len(scalingList) * len(encodingList)
    dataframeList = [0 for j in range(size)]

    for i in range(len(encodingList)):
        data_cate = encoding(df_cate, encodingList[i])
        for x in range(len(scalingList)):
            data_nume = scaling(df_nume, scalingList[x])

            # OrdinalEncoding
            if i == 0:
                index = x
            # OneHotEncoding
            else:
                index = x + i + len(scalingList) - 1
            dataframeList[index] = pd.concat([data_nume, data_cate], axis=1)

            # ############# Use this if you want to print all the combination dataframe ##############
            # print('########################################')
            # print('Scaling Method:', scalingList[x], '\nEncoding Method:', encodingList[i])
            # print(dataframeList[index])
            # ##############################################################################

    return dataframeList


def Score(df, scalingList, encodingList, algorithmList):
    scaled_df = compute_Score(df, scalingList, encodingList)

    size = len(scalingList) * len(encodingList)
    s = [0 for i in range(size)]
    # Score using these algorithms
    for i in range(len(scaled_df)):
        x = scaled_df[i]
        y = df_final['Salary in $']
        if algorithmList == 'LinearRegression':
            s[i] = LinearReg(x, y)
        elif algorithmList == 'RandomForestRegression':
            s[i] = RandomForestReg(x, y)

    # find the best scores and indices
    max_index = s.index(max(s))  # index of the best score

    scaling_index = max_index % len(scalingList)
    scaled_method = scalingList[scaling_index]  # Save best scaling method
    if max_index % 2 == 0:
        encoding_index = 1
    else:
        encoding_index = 1
    print('encoding_index:', encoding_index)
    encoding_method = encodingList[encoding_index]  # Save best encoding method
    print(algorithmList, s)
    print('The best method:', scaled_method, encoding_method, algorithmList)
    return max(s)

def Score_df(df):
    # if parameter is only dataframe, the other parameter are filled with default value
    scalingList = ['StandardScaling']
    encodingList = ['OrdinalEncoding']
    algorithmList = 'LinearRegression'

    scaled_df = compute_Score(df, scalingList, encodingList)

    size = len(scalingList) * len(encodingList)
    s = [0 for i in range(size)]

    for i in range(len(scaled_df)):
        x = scaled_df[i]
        y = df_final['Salary in $']
        if algorithmList == 'LinearRegression':
            s[i] = LinearReg(x, y)
        elif algorithmList == 'RandomForestRegression':
            s[i] = RandomForestReg(x, y)

    print(algorithmList, s)
    print('Used default values:', scalingList, encodingList, algorithmList)
    return max(s)



########### User  part

print('#########################\nScore ')
score = Score(df_f,
              ['StandardScaling', 'RobustScaling', 'MinMaxScaling', 'MaxAbsScaling','Normalizer'],
              ['OrdinalEncoding', 'OneHotEncoding'], 'LinearRegression')

print('The best score:', score)

# if you want to input only dataframe, use Score_df ( It will use default scaler & encoder & algorithm )
print('#########################\nScore_df')
score = Score_df(df_f)
print('Score using the default values:', score)

#############################
# Actual dataframe we will use in this project

score = Score(df_f, ['StandardScaling', 'RobustScaling', 'MinMaxScaling', 'MaxAbsScaling'],
              ['OrdinalEncoding', 'OneHotEncoding'], 'RandomForestRegression')
print(score)
contributionList = compute_Score(df_f, ['StandardScaling', 'RobustScaling', 'MinMaxScaling', 'MaxAbsScaling'],
                                 ['OrdinalEncoding', 'OneHotEncoding'])
print(contributionList[4])

contributionList_2016 = compute_Score(df_f_2016, ['StandardScaling', 'RobustScaling', 'MinMaxScaling', 'MaxAbsScaling'],
                                 ['OrdinalEncoding', 'OneHotEncoding'])
print(contributionList_2016[4])

#############################


# #########################
# #
# # 3. Data Inspection
# #
# #########################


# hitmap
df_stoe = contributionList[0] # get the data
salary = pd.DataFrame(df_final['Salary in $'])
print(salary)
htm = pd.concat([df_stoe, salary], axis=1)
corrmat = htm.corr()
top_corr_features = corrmat.index # set the covariance matrix related with salary
plt.figure(figsize=(20, 20))
# showing heatmap
g = sn.heatmap(htm[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()  # G, GS,MP, PER, USG%, WS, BPM, VORP, 2P, 2PA, FT, FTA, DRB, TOV, PTS, OBPM이 salary are high-related

# #########################
# #
# # 4. Data Analysis
# #
# #########################


# prepare
y = pd.DataFrame(df_final['Salary in $'])  # set the target
scaler = StandardScaler()
y = scaler.fit_transform(y) # scale the target
y = y.ravel() # make the target to 2-dimension

# use Standard scaling + oneHot encoding
df_stoh = contributionList[4]

# ###################
# # 4.1. Linear Regression (Regression)

x = df_stoh  # set the feature
# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    shuffle=True)  # split test and train 0.75: 0.25
linearReg = LinearRegression() # declare the model
linearReg.fit(x_train, y_train) # fit

predict_salary = linearReg.predict(x_test) # predict the salary
print(predict_salary)
print(y_test)
# evaluate the model score
print(linearReg.score(x_test, y_test))  # 0.5140121317136024
# compare the test data and predict salary
r2 = r2_score(y_test, predict_salary) # compare actual data and predict data
print("R squared:", r2)
plt.scatter(y_test, predict_salary, alpha=0.2) # print the scatter plot by actual salary and predict
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Rent")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show() # show the plot


###########


# ###################
# # 4.2. Decision Tree (Classification)
#
from sklearn import tree

# fit the tree model
# declare the decision tree
reg = tree.DecisionTreeRegressor(max_depth=7, criterion="mse")
dtree = reg.fit(x_train, y_train) # fitting train data
pred = reg.predict(x_test) # predict

print('예측결과:', pred) # print the predict value
print('정답:', y_test) # print the actual data
score_test = reg.score(x_test, y_test) # evaluate model score
print('score: ', score_test)
# print the plot by actual data and predict data
plt.scatter(y_test, pred, alpha=0.2)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Rent")
plt.title("Decision Tree")
plt.show()

# ###################

# ##################
# 4.3. K-means Clustering (Clustering)
#
# use Standard scaling + oneHot encoding
df_stoh = contributionList[4]
y = pd.DataFrame(df_final['Salary in $']) # set the target
y = y.values.ravel()  # set the target

# clustering by other stat feature
y = np.array(df_final['Salary in $'])  # set the target
x = np.array(df_stoh)  # set the feature
pca = PCA(n_components=10) # declare pca
pca.fit(x) # pca fit and transform
plot_columns = pca.fit_transform(x)
print(plot_columns.shape) #print the columns shape
n_cluster = 8  # set the cluster 8 8 is the best k
kmeans = KMeans(n_clusters=n_cluster, random_state=123, algorithm='auto', copy_x=True, init='k-means++',
                max_iter=300, n_init=10,
                tol=0.0001, verbose=0)
# declare the kmeans model
# set the kmeans
kmeans.fit(x)
labels = kmeans.labels_ # get the label
plot_columns = pca.fit_transform(x)
plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels) # showing the cluster of players
plt.show()
labels = kmeans.labels_
# dividing cluster and add feature to original data
df_final['cluster'] = labels
df_stoh['cluster'] = labels
# showing the cluster by plot
ax = plt.subplots(figsize=(10, 5))
ax = sn.countplot(labels) # counts the player counts
title = "histogram of Cluster Counts"
ax.set_title(title, fontsize=12)
plt.show()  # showing the cluster: 8
df_final.drop('index', axis=1, inplace=True)
# divide dataframe by cluster's feature
df_c1 = df_final[df_final['cluster'] == 0]
df_c2 = df_final[df_final['cluster'] == 1]
df_c3 = df_final[df_final['cluster'] == 2]
df_c4 = df_final[df_final['cluster'] == 3]
df_c5 = df_final[df_final['cluster'] == 4]
df_c6 = df_final[df_final['cluster'] == 5]
df_c7 = df_final[df_final['cluster'] == 6]
df_c8 = df_final[df_final['cluster'] == 7]
# cluster exploration
# see the each cluster's  PTS mean
print("PTS mean")
print("cluster0: ", df_c1['PTS'].mean())
print("cluster1: ", df_c2['PTS'].mean())
print("cluster2: ", df_c3['PTS'].mean())
print("cluster3: ", df_c4['PTS'].mean())
print("cluster4: ", df_c5['PTS'].mean())
print("cluster5: ", df_c6['PTS'].mean())
print("cluster6: ", df_c7['PTS'].mean())
print("cluster7: ", df_c8['PTS'].mean())
# see the each cluster's winshare mean
print("Win share Mean")
print("cluster0: ", df_c1['WS'].mean())
print("cluster1: ", df_c2['WS'].mean())
print("cluster2: ", df_c3['WS'].mean())
print("cluster3: ", df_c4['WS'].mean())
print("cluster4: ", df_c5['WS'].mean())
print("cluster5: ", df_c6['WS'].mean())
print("cluster6: ", df_c7['WS'].mean())
print("cluster7: ", df_c8['WS'].mean())
# see the  each cluster's salary mean
print("Salary mean")
print("cluster0: ", df_c1['Salary in $'].mean())
print("cluster1: ", df_c2['Salary in $'].mean())
print("cluster2: ", df_c3['Salary in $'].mean())
print("cluster3: ", df_c4['Salary in $'].mean())
print("cluster4: ", df_c5['Salary in $'].mean())
print("cluster5: ", df_c6['Salary in $'].mean())
print("cluster6: ", df_c7['Salary in $'].mean())
print("cluster7: ", df_c8['Salary in $'].mean())
# see the each cluster's 3p mean
print("3P mean")
print("cluster0: ", df_c1['3P'].mean())
print("cluster1: ", df_c2['3P'].mean())
print("cluster2: ", df_c3['3P'].mean())
print("cluster3: ", df_c4['3P'].mean())
print("cluster4: ", df_c5['3P'].mean())
print("cluster5: ", df_c6['3P'].mean())
print("cluster6: ", df_c7['3P'].mean())
print("cluster7: ", df_c8['3P'].mean())
# see the each cluster's 2p mean
print("2P mean")
print("cluster0: ", df_c1['2P'].mean())
print("cluster1: ", df_c2['2P'].mean())
print("cluster2: ", df_c3['2P'].mean())
print("cluster3: ", df_c4['2P'].mean())
print("cluster4: ", df_c5['2P'].mean())
print("cluster5: ", df_c6['2P'].mean())
print("cluster6: ", df_c7['2P'].mean())
print("cluster7: ", df_c8['2P'].mean())
# for distinguishing their starting player or bench member
print("Frequency of starting member")
print(df_c1['GS'].value_counts())
print(df_c2['GS'].value_counts())
print(df_c3['GS'].value_counts())
print(df_c4['GS'].value_counts())
print(df_c5['GS'].value_counts())
print(df_c6['GS'].value_counts())
print(df_c7['GS'].value_counts())
print(df_c8['GS'].value_counts())

#
# ##################


# ##################
# 4.4. Ensemble (Clustering)
#
#
# prepare
y = pd.DataFrame(df_final['Salary in $'])  # set the target
scaler = StandardScaler()
y = scaler.fit_transform(y)
y = y.ravel()
# use Standard scaling + oneHot encoding
df_stoh = contributionList[4]

x = df_stoh  # set the feature
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)  # split test and train 0.75: 0.25
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=1) # split train and valid

#######################
# 1. Bagging
#######################

print("***** Bagging *****")
# set the param grid
param_grid = {'n_estimators': [50, 100, 150],
              'max_samples': [0.25, 0.5, 1.0],
              'max_features': [0.25, 0.5, 1.0],
              'bootstrap_features': [True, False]
              }
# Grid Search
bagging_grid = GridSearchCV(BaggingRegressor(DecisionTreeRegressor(max_depth=10)), param_grid,
                            cv=5, n_jobs=-1, verbose=1)
bagging_grid.fit(x_train, y_train) # fit
# show the best parameter and best score
print('Best params:', bagging_grid.best_params_)
print('Best score:', bagging_grid.best_score_)
# set the best feature bagging model
bagging_best = BaggingRegressor(DecisionTreeRegressor(max_depth=10),
                                bootstrap_features=False, max_features=1.0,max_samples=1.0, n_estimators=150)
bagging_best.fit(x_train, y_train) # fit
best_pred = bagging_best.predict(x_test) # predict
mse = mean_squared_error(best_pred, y_test) # calculate the mse and RMSE
print("***** Bagging best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse))
print("RMSE : ", np.sqrt(mse))
print("Best score:", bagging_best.score(x_test, y_test))

#######################
# 2. Random Forest
#######################
rfModel = RandomForestRegressor(random_state=10) # declare the randomforest regressor
rfModel.fit(x_train, y_train) # fit
ypred = rfModel.predict(x_test) # predict
print("***** Random Forest*****") # calculate the model score before grid search
print('RF Score: ', rfModel.score(x_test, y_test))  # model score
print('RF RMSE:', np.sqrt(mean_squared_error(ypred, y_test)))

# define grid parameters
param_grid = {'bootstrap': [True, False],
              'n_estimators': [3, 10, 30],
              'max_features': [20, 40, 60],
              'max_samples': [5, 10],
              'max_depth': [2, 4, 6, 8, 10]}

# initialize the GridSearchCV class
rf_grid = GridSearchCV(rfModel, param_grid, cv=5, n_jobs=-1)
# train the class
rf_grid.fit(x_train, y_train) # fit
# find best parameters
print("Best params: ", rf_grid.best_params_)
print("Best score: ", rf_grid.best_score_)
# set the model with best parameter
rf_best = RandomForestRegressor(bootstrap=False, max_depth=10, max_features=60, max_samples=5, n_estimators=30)

# evaluate error
rf_best.fit(x_train, y_train) # fit
best_ypred = rf_grid.predict(x_test) # predict
mse = mean_squared_error(best_ypred, y_test) # evaluate error
print("***** Random Forest best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse)) # mse
print("RMSE : ", np.sqrt(mse)) # RMSE
print("Best score:", rf_best.score(x_test, y_test)) # best score


############################
# 3. GradientBoosting
############################
gbr = GradientBoostingRegressor(random_state=13).fit(x_train, y_train) # declare the gradient boosting
gbr_pred = gbr.predict(x_test) # predict the value
# print the score before parameter grid search
print("***** GradientBoosting *****")
print('Gradient Score: ', gbr.score(x_test, y_test))
print('Gradient RMSE:', np.sqrt(mean_squared_error(gbr_pred, y_test)))

# define grid parameters
param_grid = {'n_estimators': [15, 30, 100, 600],
              'learning_rate': [0.01, 0.05, 0.1],
              'max_features': [0.3, 0.5, 1.0],
              'max_depth': [6, 8, 12]}
gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state=33), param_grid,
                        cv=3, n_jobs=-1, verbose=1)
gbr_grid.fit(x_train, y_train)
# best parameter and score
print('Best params:', gbr_grid.best_params_)
print('Best score:', gbr_grid.best_score_)
# declare the model with best parameters
gbr_best = GradientBoostingRegressor(learning_rate=0.01, max_depth=6,
                                     max_features=0.5, n_estimators=600)
gbr_best.fit(x_train, y_train) # fit
best_pred = gbr_best.predict(x_test) # best predict
mse = mean_squared_error(best_pred, y_test) # evaluate the error
print("***** Gradient Boost best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse)) # mse
print("RMSE : ", np.sqrt(mse)) # RMSE
print("Best score:", gbr_best.score(x_test, y_test)) # best score


############################
# 4. AdaBoosting
############################
abc = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), random_state=13).fit(x_train, y_train) # declare the ada boost
abc_pred = abc.predict(x_test) # predict
# print the score before parameter grid search
print("***** Ada Boosting *****")
print('AdaBoosting Score: ', abc.score(x_test, y_test))
print('AdaBoosting RMSE:', np.sqrt(mean_squared_error(abc_pred, y_test)))

# define grid parameters
param_grid = {'n_estimators': [15, 30, 100, 600],
              'learning_rate': [0.01, 0.05, 0.1],
              'loss': ['linear', 'square', 'exponential']}
abc_grid = GridSearchCV(abc, param_grid,
                        cv=5, n_jobs=-1, verbose=1)
abc_grid.fit(x_train, y_train)
# best parameter and score
print('Best params:', abc_grid.best_params_)
print('Best score:', abc_grid.best_score_)
# declare the model with best parameter
abc_best = AdaBoostRegressor(learning_rate=0.1, loss='linear', n_estimators=600)
abc_best.fit(x_train, y_train) # fit
best_pred = abc_best.predict(x_test)
mse = mean_squared_error(best_pred, y_test) # evaluate error
print("***** Gradient Boost best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse)) # mse
print("RMSE : ", np.sqrt(mse)) # RMSE
print("Best score:", abc_best.score(x_test, y_test)) # score


#########################
# predict 2016 profit
# use Standard scaling + oneHot encoding
df_stoh = contributionList_2016[4] # set the feature with 2016 data
y = pd.DataFrame(df_final_2016['Salary in $']) # set the target
y = y.values.ravel()  # set the target
# df_stoh.drop('Salary in $', axis=1, inplace=True)

x = df_stoh  # set the feature
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    shuffle=True)  # split test and train 0.75: 0.25
linearReg = LinearRegression() # declare linear regression model
linearReg.fit(x_train, y_train) # fit

predict_salary = linearReg.predict(x_test) # predict salary for 2016
print(predict_salary) # print predict salary
print(y_test) # print actual data
print(linearReg.score(x_test, y_test))  # 0.5140121317136024
plt.scatter(y_test, predict_salary, alpha=0.2) # compare the actual and predict with scatter plot
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Rent")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()


# predict with random forest
rfModel = RandomForestRegressor() # declare the model
rfModel.fit(x_train, y_train) # fit
ypred = rfModel.predict(x_test) # predict 2016 salary using random forest

print('RFModel score:', rfModel.score(x_test, y_test)) # print the score
profit = predict_salary - y_test # calculate predict and actual's difference
print(profit.sum())
# predict with adaboost
reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=500)
adaboost = reg.fit(x_train, y_train) # fit
predict_salary = reg.predict(x_test) # predict 2016 salary for using adaboost
print(predict_salary)
print(y_test)
print('AdaBoost Score:', adaboost.score(x_test, y_test))# print the score
profit = predict_salary - y_test # calculate predict and actual's difference
print(profit.sum())

# ##################
