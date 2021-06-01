import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
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
print(df1.isna().sum())
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
df2['Season Start'] = df2['Season Start'].astype("int64")
df2['Season End'] = df2['Season End'].astype("int64")
print(df2.info())  # confirm the df2 info
df2.rename(columns={'Player Name': 'Player', 'Season End': 'Year', 'Team': 'Tm'}, inplace=True)
df4 = pd.merge(df1, df2, on=['Player', 'Year', 'Tm'], how="left")
df4.drop(['Season Start'], axis=1, inplace=True)  # drop the trash feature
df4.drop(['blanl', 'blank2'], axis=1, inplace=True)  # drop the feature
print(df4)
print("Before preprocessing")
print(df4.isna().sum())  # need to preprocessing
df4 = df4.dropna(subset=['Salary in $'])
print("After preprocessing")
print(df4.isna().sum())
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
          'Pos'])  # they are same players having not null value because their index is same
# different number of nan: 28 , but both are overlapped
print(df_final[df_final['3PAr'].isnull()]['Pos'])
print(df_final[df_final['FTr'].isnull()]['Pos'])
print(df_final[df_final['FG%'].isnull()]['Pos'])
print(df_final[df_final['eFG%'].isnull()]['Pos'])
print(df_final[df_final['DRB%'].isnull()])
df_trade = df_final.loc[df_final['G'] <= 10]
print(df_trade.shape)

print(df_final['G'].mean())  # 52.281978126485974
# game < 10 is outlier
df_final = df_final[df_final['G'] >= 10]
df_final.drop(['2P%', '3P%', 'OWS', 'DWS', 'WS/48', 'FT%', 'Full Team Name', 'Register Value'
                  , 'TRB', 'FG', 'FGA'], axis=1, inplace=True)

df_final.reset_index(inplace=True)
print(df_final.isnull().sum())  # now, NAN cleared
print(df_final.info())

df_final_2016 = df_final[df_final['Year'] == 2016]
df_final_2016.reset_index(inplace=True)
# 함수에 들어갈 dataframe
df_f = df_final.drop(['Player', 'index', 'Salary in $'], axis=1)
df_f_2016 = df_final_2016.drop(['Player', 'index', 'level_0', 'Salary in $'], axis=1)


#############################
# open source 기여


def scaling(data, scale):
    if scale == 'StandardScaling':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    elif scale == 'RobustScaling':
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
    elif scale == 'MinMaxScaling':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaler = MaxAbsScaler()
        scaled_data = scaler.fit_transform(data)
    df_scaled_data = pd.DataFrame(columns=data.columns, data=scaled_data)
    return df_scaled_data


def encoding(data, encode):
    if encode == 'OrdinalEncoding':
        oe = OrdinalEncoder()
        encoded_data = oe.fit_transform(data)
        encoded_data = pd.DataFrame(columns=data.columns, data=encoded_data)
    elif encode == 'OneHotEncoding':
        encoded_data = pd.get_dummies(data)
    return encoded_data


def contribution(df):
    df_cate = df.select_dtypes(include='object')
    df_nume = df.select_dtypes(exclude='object')

    scalingList = ['StandardScaling', 'RobustScaling', 'MinMaxScaling', 'MaxAbsScaling']
    encodingList = ['OrdinalEncoding', 'OneHotEncoding']
    dataframeList = ['df_stoe', 'df_rboe', 'df_mmoe', 'df_maoe',
                     'df_stoh', 'df_rboh', 'df_mmoh', 'df_maoh']

    for i in range(len(encodingList)):
        data_cate = encoding(df_cate, encodingList[i])
        for x in range(len(scalingList)):
            data_nume = scaling(df_nume, scalingList[x])
            if i == 0:
                dataframeList[x] = pd.concat([data_nume, data_cate], axis=1)
            else:
                dataframeList[x + i + 3] = pd.concat([data_nume, data_cate], axis=1)
    return dataframeList


contributionList = contribution(df_f)
print(contributionList[4])

contributionList_2016 = contribution(df_f_2016)
print(contributionList_2016[4])

#############################


# #########################
# #
# # 3. Data Inspection
# #
# #########################


# hitmap으로 중요한 feature 알아보기
df_stoe = contributionList[0]
salary = pd.DataFrame(df_final['Salary in $'])
print(salary)
htm = pd.concat([df_stoe, salary], axis=1)
corrmat = htm.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))

g = sn.heatmap(htm[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()  # G, GS,MP, PER, USG%, WS, BPM, VORP, 2P, 2PA, FT, FTA, DRB, TOV, PTS, OBPM이 salary와 높은 연관

# #########################
# #
# # 4. Data Analysis
# #
# #########################


# prepare
y = pd.DataFrame(df_final['Salary in $'])  # set the target
scaler = StandardScaler()
y = scaler.fit_transform(y)
y = y.ravel()

# use Standard scaling + oneHot encoding
df_stoh = contributionList[4]

# ###################
# # 4.1. Linear Regression (Regression)

x = df_stoh  # set the feature
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    shuffle=True)  # split test and train 0.75: 0.25
linearReg = LinearRegression()
linearReg.fit(x_train, y_train)

predict_salary = linearReg.predict(x_test)
print(predict_salary)
print(y_test)
print(linearReg.score(x_test, y_test))  # 0.5140121317136024

r2 = r2_score(y_test, predict_salary)
print("R squared:", r2)
plt.scatter(y_test, predict_salary, alpha=0.2)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Rent")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

# # PCA 적용 전에는 overFitting 되어서 score가 1이 나옴. --> PCA를 써서 차원 축소하여 overFitting 피하기
# pca = PCA(.90)
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
#
# linearReg = LinearRegression()
# linearReg.fit(x_train, y_train)
#
# predict_salary = linearReg.predict(x_test)
# print(predict_salary)
# print(y_test)
# print('PCA Linear Score:', linearReg.score(x_test, y_test))
# plt.scatter(y_test, predict_salary, alpha=0.2)
# plt.xlabel("Actual Salary")
# plt.ylabel("Predicted Rent")
# plt.title("PCA MULTIPLE LINEAR REGRESSION")
# plt.show()

###########


# ###################
# # 4.2. Decision Tree (Classification)
#
from sklearn import tree

# fit the tree model

reg = tree.DecisionTreeRegressor(max_depth=7, criterion="mse")
dtree = reg.fit(x_train, y_train)
pred = reg.predict(x_test)

print('예측결과:', pred)
print('정답:', y_test)
score_test = reg.score(x_test, y_test)
print('score: ', score_test)
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
y = pd.DataFrame(df_final['Salary in $'])
y = y.values.ravel()  # set the target
# df_stoh.drop('Salary in $', axis=1, inplace=True)

# clustering by position
y = np.array(df_final['Salary in $'])  # set the target
x = np.array(df_stoh)  # set the feature
pca = PCA(n_components=10)
pca.fit(x)
plot_columns = pca.fit_transform(x)
print(plot_columns.shape)
n_cluster = 8  # set the cluster 8
kmeans = KMeans(n_clusters=n_cluster, random_state=123, algorithm='auto', copy_x=True, init='k-means++',
                max_iter=300, n_init=10,
                tol=0.0001, verbose=0)
# set the kmeans
kmeans.fit(x)
labels = kmeans.labels_
plot_columns = pca.fit_transform(x)
plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
plt.show()
labels = kmeans.labels_
df_final['cluster'] = labels
df_stoh['cluster'] = labels
ax = plt.subplots(figsize=(10, 5))
ax = sn.countplot(labels)
title = "histogram of Cluster Counts"
ax.set_title(title, fontsize=12)
plt.show()  # showing the cluster: 8
df_final.drop('index', axis=1, inplace=True)
df_c1 = df_final[df_final['cluster'] == 0]
df_c2 = df_final[df_final['cluster'] == 1]
df_c3 = df_final[df_final['cluster'] == 2]
df_c4 = df_final[df_final['cluster'] == 3]
df_c5 = df_final[df_final['cluster'] == 4]
df_c6 = df_final[df_final['cluster'] == 5]
df_c7 = df_final[df_final['cluster'] == 6]
df_c8 = df_final[df_final['cluster'] == 7]
# cluster exploration
print("PTS mean")
print("cluster0: ", df_c1['PTS'].mean())
print("cluster1: ", df_c2['PTS'].mean())
print("cluster2: ", df_c3['PTS'].mean())
print("cluster3: ", df_c4['PTS'].mean())
print("cluster4: ", df_c5['PTS'].mean())
print("cluster5: ", df_c6['PTS'].mean())
print("cluster6: ", df_c7['PTS'].mean())
print("cluster7: ", df_c8['PTS'].mean())
print("Win share Mean")
print("cluster0: ", df_c1['WS'].mean())
print("cluster1: ", df_c2['WS'].mean())
print("cluster2: ", df_c3['WS'].mean())
print("cluster3: ", df_c4['WS'].mean())
print("cluster4: ", df_c5['WS'].mean())
print("cluster5: ", df_c6['WS'].mean())
print("cluster6: ", df_c7['WS'].mean())
print("cluster7: ", df_c8['WS'].mean())
print("Salary mean")
print("cluster0: ", df_c1['Salary in $'].mean())
print("cluster1: ", df_c2['Salary in $'].mean())
print("cluster2: ", df_c3['Salary in $'].mean())
print("cluster3: ", df_c4['Salary in $'].mean())
print("cluster4: ", df_c5['Salary in $'].mean())
print("cluster5: ", df_c6['Salary in $'].mean())
print("cluster6: ", df_c7['Salary in $'].mean())
print("cluster7: ", df_c8['Salary in $'].mean())
print("3P mean")
print("cluster0: ", df_c1['3P'].mean())
print("cluster1: ", df_c2['3P'].mean())
print("cluster2: ", df_c3['3P'].mean())
print("cluster3: ", df_c4['3P'].mean())
print("cluster4: ", df_c5['3P'].mean())
print("cluster5: ", df_c6['3P'].mean())
print("cluster6: ", df_c7['3P'].mean())
print("cluster7: ", df_c8['3P'].mean())
print("2P mean")
print("cluster0: ", df_c1['2P'].mean())
print("cluster1: ", df_c2['2P'].mean())
print("cluster2: ", df_c3['2P'].mean())
print("cluster3: ", df_c4['2P'].mean())
print("cluster4: ", df_c5['2P'].mean())
print("cluster5: ", df_c6['2P'].mean())
print("cluster6: ", df_c7['2P'].mean())
print("cluster7: ", df_c8['2P'].mean())
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
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=1)

#######################
# 1. Bagging
#######################

print("***** Bagging *****")
param_grid = {'n_estimators': [50, 100, 150],
              'max_samples': [0.25, 0.5, 1.0],
              'max_features': [0.25, 0.5, 1.0],
              'bootstrap_features': [True, False]
              }

bagging_grid = GridSearchCV(BaggingRegressor(DecisionTreeRegressor(max_depth=10)), param_grid,
                            cv=5, n_jobs=-1, verbose=1)
bagging_grid.fit(x_train, y_train)

print('Best params:', bagging_grid.best_params_)
print('Best score:', bagging_grid.best_score_)

bagging_best = BaggingRegressor(DecisionTreeRegressor(max_depth=10),
                                bootstrap_features=False, max_features=1.0,max_samples=1.0, n_estimators=150)
bagging_best.fit(x_train, y_train)
best_pred = bagging_best.predict(x_test)
mse = mean_squared_error(best_pred, y_test)
print("***** Bagging best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse))
print("RMSE : ", np.sqrt(mse))
print("Best score:", bagging_best.score(x_test, y_test))

#######################
# 2. Random Forest
#######################
rfModel = RandomForestRegressor(random_state=10)
rfModel.fit(x_train, y_train)
ypred = rfModel.predict(x_test)
print("***** Random Forest*****")
print('RF Score: ', rfModel.score(x_test, y_test))
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
rf_grid.fit(x_train, y_train)

print("Best params: ", rf_grid.best_params_)
print("Best score: ", rf_grid.best_score_)
rf_best = RandomForestRegressor(bootstrap=False, max_depth=10, max_features=60, max_samples=5, n_estimators=30)

# evaluate error
rf_best.fit(x_train, y_train)
best_ypred = rf_grid.predict(x_test)
mse = mean_squared_error(best_ypred, y_test)
print("***** Random Forest best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse))
print("RMSE : ", np.sqrt(mse))
print("Best score:", rf_best.score(x_test, y_test))


############################
# 3. GradientBoosting
############################
gbr = GradientBoostingRegressor(random_state=13).fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
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

print('Best params:', gbr_grid.best_params_)
print('Best score:', gbr_grid.best_score_)

gbr_best = GradientBoostingRegressor(learning_rate=0.01, max_depth=6,
                                     max_features=0.5, n_estimators=600)
gbr_best.fit(x_train, y_train)
best_pred = gbr_best.predict(x_test)
mse = mean_squared_error(best_pred, y_test)
print("***** Gradient Boost best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse))
print("RMSE : ", np.sqrt(mse))
print("Best score:", gbr_best.score(x_test, y_test))


############################
# 4. AdaBoosting
############################
abc = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), random_state=13).fit(x_train, y_train)
abc_pred = abc.predict(x_test)
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

print('Best params:', abc_grid.best_params_)
print('Best score:', abc_grid.best_score_)

abc_best = AdaBoostRegressor(learning_rate=0.1, loss='linear', n_estimators=600)
abc_best.fit(x_train, y_train)
best_pred = abc_best.predict(x_test)
mse = mean_squared_error(best_pred, y_test)
print("***** Gradient Boost best fit *****")
print("The mean squared error(MSE) on test set: {:.4f}".format(mse))
print("RMSE : ", np.sqrt(mse))
print("Best score:", abc_best.score(x_test, y_test))


#########################
# predict 2016 profit
# use Standard scaling + oneHot encoding
df_stoh = contributionList_2016[4]
y = pd.DataFrame(df_final_2016['Salary in $'])
y = y.values.ravel()  # set the target
# df_stoh.drop('Salary in $', axis=1, inplace=True)

x = df_stoh  # set the feature
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,
                                                    shuffle=True)  # split test and train 0.75: 0.25
linearReg = LinearRegression()
linearReg.fit(x_train, y_train)

predict_salary = linearReg.predict(x_test)
print(predict_salary)
print(y_test)
print(linearReg.score(x_test, y_test))  # 0.5140121317136024
plt.scatter(y_test, predict_salary, alpha=0.2)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Rent")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

# # PCA 적용 전에는 overFitting 되어서 score가 1이 나옴. --> PCA를 써서 차원 축소하여 overFitting 피하기
# pca = PCA(.90)
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
#
# linearReg = LinearRegression()
# linearReg.fit(x_train, y_train)
#
# predict_salary = linearReg.predict(x_test)
# print(predict_salary)
# print(y_test)
# print('PCA Linear Score 2016:', linearReg.score(x_test, y_test))
# profit = predict_salary - y_test
# print(profit.sum())
# plt.scatter(y_test, predict_salary, alpha=0.2)
# plt.xlabel("Actual Salary")
# plt.ylabel("Predicted Rent")
# plt.title("PCA MULTIPLE LINEAR REGRESSION")
# plt.show()

rfModel = RandomForestRegressor()
rfModel.fit(x_train, y_train)
ypred = rfModel.predict(x_test)

print('RFModel score:', rfModel.score(x_test, y_test))
profit = predict_salary - y_test
print(profit.sum())

reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=500)
adaboost = reg.fit(x_train, y_train)
predict_salary = reg.predict(x_test)
print(predict_salary)
print(y_test)
print('AdaBoost Score:', adaboost.score(x_test, y_test))
profit = predict_salary - y_test
print(profit.sum())

# ##################
