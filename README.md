# DataScience-NBA-salary-predict
You can get the all combinations of these scaling & encoding & algorithms.

* Available Scaling
  * StandardScaling
  * RobustScaling
  * MinMaxScaling
  * MaxAbsScaling
  * Normalizer
 
* Available Encoding
  * OrdinalEncoding
  * OneHotEncoding


* Available Algorithms to score
  * LinearRegression
  * DecisionTreeRegression

## You can get the result by this way
```python
# Type the scaler / encoder / algorithms that you want to run.
score = Score(df_f,
             ['StandardScaling', 'MaxAbsScaling', 'Normalizer'],
             ['OrdinalEncoding', 'OneHotEncoding'],
             'LinearRegression')

print('The best score:', score)
```
##### If you can't select encoder & scaler, we will use default values (StandardScaling, OrdinalEncoding, LinearRegression)
```python
score = Score_df(df_f)
print('Score using the default values:', score)
```




## Fuction Description

This is fitting & scoreing function with the given data(x, y).
```python
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
 ```
 
 
 This is Scaling function using given scaler.
 
 ```python
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
 
  ```
 
 
 This is Encoding function using given encoder.
 
 ```python
# encoding
def encoding(data, encode):
	if encode == 'OrdinalEncoding':
    	oe = OrdinalEncoder()
    	encoded_data = oe.fit_transform(data)
    	encoded_data = pd.DataFrame(columns=data.columns, data=encoded_data)
	elif encode == 'OneHotEncoding':
    	encoded_data = pd.get_dummies(data)
 
	return encoded_data
 
   ```
 
 
 This function split the given dataset into categorical & numerical dataset.
 Then run the encoder with categorical dataset and scale with the numerical dataset.
 Put the result of it into the list 'dataframeList' by the order.
 This returns that list(scaled & encoded).
 
 ```python
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
 
  ```
 
 
This function fit & test the list of datasets. Gives you the best score and the method.
 
 ```python
 
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
```
