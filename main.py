#Theotziol Neural-FCM classifier

import pandas as pd 
import numpy as np 
import time 
import os 
from dynamic_fcm import *
from preprocessing import * 
from matplotlib import pyplot as plt 
from datetime import datetime
from graphs import *


import warnings

## Suppress all warnings
warnings.filterwarnings("ignore")

# warnings.resetwarnings()

# Data loading

data = 'data'

wdbc_data = os.path.join(data, 'Breast Cancer Wisconsin', 'data.csv') #binary
cad_data = os.path.join(data, 'cad', 'TF SPECT_labelsv2_with_normals - anonymous-final.xlsx') #binary
iris_data = os.path.join(data, 'iris_data', 'Iris.csv') #3 classes
diabetes_data = os.path.join(data, 'pima diabetes', 'diabetes.csv') #binary
thyroid_data = os.path.join(data, 'thyroid', 'new-thyroid.data') #3 classes
wine_data = os.path.join(data, 'wine', 'wine.data') #3 classes
german_data = os.path.join(data, 'german', 'german.data') #2 classses 1=good 2=bad


thyroid_names= ['Class','T3-resin %', 'Total Serum thyroxin','Total serum triiodothyronine','TSH','Maximal absolute difference of TSH value']
wine_names = [	'class',
    'Alcohol',
     'Malic acid',
 	'Ash',
	'Alcalinity of ash',  
 	'Magnesium',
	'Total phenols',
 	'Flavanoids',
 	'Nonflavanoid phenols',
 	'Proanthocyanins',
	'Color intensity',
 	'Hue',
 	'OD280/OD315 of diluted wines',
 	'Proline']
german_names = [
	'account_status',
	'duration',
	'credit_history',
	'purpose',
	'credit_amnt',
	'bonds',
	'employment',
	'installment_rate%',
	'marriage_sex',
	'guarantors',
	'residence',
	'property',
	'age',
	'installment_plans',
	'housing',
	'num_credits',
	'job',
	'num_liable_people',
	'telephone',
	'foreign',
	'class'
]

wdbc = pd.read_csv(wdbc_data, index_col=0)
try:
    wdbc.drop(['Unnamed: 32'], axis = 1, inplace = True)
except:
    pass

cad = pd.read_excel(cad_data)
## Data preprocessing
cad.dropna(inplace = True)
cad.drop(['No'], inplace = True, axis = 1)
cad.reset_index(drop = True, inplace = True)
cad = text_to_numerical(cad, 'SEX')

iris = pd.read_csv(iris_data, index_col= 0)

diabetes = pd.read_csv(diabetes_data)

thyroid = pd.read_csv(thyroid_data, names = thyroid_names)
# thyroid['Class'][thyroid['Class'] == 1] = 0.0
# thyroid['Class'][thyroid['Class'] == 2] = 1.0
# thyroid['Class'][thyroid['Class'] == 0] = 2.0

wine = pd.read_csv(wine_data, names = wine_names)

german = pd.read_csv(german_data, names = german_names, delim_whitespace=True)
text_columns_german = german.select_dtypes(include='object').columns
for column in text_columns_german:
	german = text_to_numerical(german, column)


## categorical dataframes
wdbc_categorical = convert_to_categorical(wdbc, wdbc.columns[0])
cad_categorical = convert_to_categorical(cad, cad.columns[-1])
iris_categorical = convert_to_categorical(iris, iris.columns[-1])
diabetes_categorical = convert_to_categorical(diabetes, diabetes.columns[-1])
thyroid_categorical = convert_to_categorical(thyroid, thyroid.columns[0])
wine_categorical = convert_to_categorical(wine, wine.columns[0])
german_categorical = convert_to_categorical(german, german.columns[-1])


## normalize 
wdbc_categorical = min_max_scaling(wdbc_categorical)
cad_categorical = min_max_scaling(cad_categorical)
iris_categorical = min_max_scaling(iris_categorical)
diabetes_categorical = min_max_scaling(diabetes_categorical)
thyroid_categorical = min_max_scaling(thyroid_categorical)
wine_categorical = min_max_scaling(wine_categorical)
german_categorical = min_max_scaling(german_categorical)

## split input/labels
wdbc_input, wdbc_labels = split_labels(wdbc_categorical, -2)
cad_input, cad_labels = split_labels(cad_categorical, -2)
iris_input, iris_labels = split_labels(iris_categorical, -3)
diabetes_input, diabetes_labels = split_labels(diabetes_categorical, -2)
thyroid_input, thyroid_labels = split_labels(thyroid_categorical, -3)
wine_input, wine_labels = split_labels(wine_categorical, -3)
german_input, german_labels = split_labels(german_categorical, -2)

dfs_inputs = [
	wdbc_input,
	cad_input,
	iris_input,
	diabetes_input,
	thyroid_input,
	wine_input,
	german_input
]
dfs_labels = [
	wdbc_labels,
	cad_labels,
	iris_labels,
	diabetes_labels,
	thyroid_labels,
	wine_labels,
	german_labels,
]

dfs_names = [
	'wdbc',
	'cad',
	'iris',
	'diabetes',
	'thyroid',
	'wine',
	'german',
]

cwd = os.getcwd()
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M")
path = os.path.join('experiments', dt_string)
os.mkdir(path)

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#inside dataset
for i in range(len(dfs_names)):
	kf = KFold(n_splits = 10, shuffle = True)
	fold = 1
	accuracies = []
	f1_macros = []
	times = []
	average_time_per_epoch = []
	epochs = []
	path_dataset = os.path.join(path, dfs_names[i])
	os.mkdir(path_dataset)

	#inside fold
	for train_index, test_index in kf.split(dfs_inputs[i]):
		x_train, y_train = dfs_inputs[i].iloc[train_index].to_numpy(), dfs_labels[i].iloc[train_index].to_numpy()
		x_test, y_test = dfs_inputs[i].iloc[test_index].to_numpy(), dfs_labels[i].iloc[test_index].to_numpy()
		path_fold = os.path.join(path_dataset, f'Fold_{fold}')
		os.mkdir(path_fold)

		print(f'Dataset {dfs_names[i]}, fold {fold}\n')

		train_y = np.concatenate([x_train,y_train], axis = -1)
		nfcm = neural_fcm(x_train.shape[-1],y_train.shape[-1], fcm_iter=2, l_slope=1)
		nfcm.initialize_loss_and_compile('cce')
		time_callback = TimeHistory()
		callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
		history = nfcm.model.fit(x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
		predictions = nfcm.predict_classification(x_test)
		nfcm.metrics_classification(y_test)
		accuracies.append(nfcm.accuracy)
		average_time_per_epoch.append(np.mean(time_callback.times))
		epochs.append(len(time_callback.times))
		f1_macros.append(nfcm.f1_score_macro)
		disp = ConfusionMatrixDisplay(confusion_matrix=nfcm.confusion_matrix)
		disp.plot()
		plt.savefig(os.path.join(path_fold, 'conf_matrix.png'), dpi = 600)
		plt.show()

		plt.plot(history.history['loss'], color = 'b', label = 'train_loss')
		plt.plot(history.history['val_loss'], color = 'c', label = 'val_loss')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Classification Loss')
		plt.title(f'Training ({dfs_names[i]} dataset)')
		plt.savefig(os.path.join(path_fold, 'training_plot.png'), dpi = 600)
		plt.show()

		with open(os.path.join(path_fold, 'stats.txt'), 'w') as f:
			f.write(f'Accuracy = {nfcm.accuracy}\n')
			f.write(f'F1_score (macro) = {nfcm.f1_score_macro}\n')
			f.write(f'Confusion Matrix = {nfcm.confusion_matrix}\n')
			f.write(f'mean time per epoch = {np.mean(time_callback.times)} ms\n')
			f.write(f'Epochs = {len(history.history["loss"])}')
		
		fold += 1
	with open(os.path.join(path_dataset, f'stats_dataset_{fold}fold.txt'), 'w') as f:
		f.write(f'Accuracies = {accuracies}\n')
		f.write(f'F1_scores (macro) = {f1_macros}\n')
		f.write(f'Average accuracy = {np.mean(accuracies)}\n')
		f.write(f'Average F1 (Macros) = {np.mean(f1_macros)}\n')

		f.write(f'Average time per epoch = {np.mean(average_time_per_epoch)} ms\n')
		f.write(f'Average Epochs = {np.mean(epochs)}')
		f.write(f'Total Average Time (average epochs x average time per epochs) = {np.mean(average_time_per_epoch)*np.mean(epochs)} ms')
		f.write(f'Total Average Time (average epochs x average time per epochs) = {np.mean(average_time_per_epoch)*np.mean(epochs)/1000} s')













## split training/testing
# wdbc_x_train, wdbc_x_test = split_df(wdbc_input)
# wdbc_y_train, wdbc_y_test = split_df(wdbc_labels)

# cad_x_train, cad_x_test = split_df(cad_input)
# cad_y_train, cad_y_test = split_df(cad_labels)

# iris_x_train, iris_x_test = split_df(iris_input)
# iris_y_train, iris_y_test = split_df(iris_labels)

# diabetes_x_train, diabetes_x_test = split_df(diabetes_input)
# diabetes_y_train, diabetes_y_test = split_df(diabetes_labels)

# thyroid_x_train, thyroid_x_test = split_df(thyroid_input)
# thyroid_y_train, thyroid_y_test = split_df(thyroid_labels)

# wine_x_train, wine_x_test = split_df(wine_input)
# wine_y_train, wine_y_test = split_df(wine_labels)

# german_x_train, german_x_test = split_df(german_input)
# german_y_train, german_y_test = split_df(german_labels)


# #dataset wdbc
# print('wdbc')
# train_y = np.concatenate([wdbc_x_train,wdbc_y_train], axis = -1)

# nfcm = neural_fcm(wdbc_x_train.shape[-1],wdbc_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(wdbc_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(wdbc_x_test)
# nfcm.metrics_classification(wdbc_y_test)


# #dataset cad
# print('cad')
# train_y = np.concatenate([cad_x_train,cad_y_train], axis = -1)

# nfcm = neural_fcm(cad_x_train.shape[-1],cad_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(cad_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(cad_x_test)
# nfcm.metrics_classification(cad_y_test)


# #dataset iris
# train_y = np.concatenate([iris_x_train,iris_y_train], axis = -1)

# nfcm = neural_fcm(iris_x_train.shape[-1],iris_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(iris_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(iris_x_test)
# print('iris')
# nfcm.metrics_classification(iris_y_test)


# #dataset diabetes
# train_y = np.concatenate([diabetes_x_train,diabetes_y_train], axis = -1)

# nfcm = neural_fcm(diabetes_x_train.shape[-1],diabetes_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(diabetes_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(diabetes_x_test)
# print('diabetes')
# nfcm.metrics_classification(diabetes_y_test)


# #dataset thyroid
# train_y = np.concatenate([thyroid_x_train,thyroid_y_train], axis = -1)

# nfcm = neural_fcm(thyroid_x_train.shape[-1],thyroid_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(thyroid_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(thyroid_x_test)
# print('thyroid')
# nfcm.metrics_classification(thyroid_y_test)


# #dataset wine
# train_y = np.concatenate([wine_x_train,wine_y_train], axis = -1)

# nfcm = neural_fcm(wine_x_train.shape[-1],wine_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(wine_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(wine_x_test)
# print('wine')
# nfcm.metrics_classification(wine_y_test)

# #dataset german
# train_y = np.concatenate([german_x_train,german_y_train], axis = -1)

# nfcm = neural_fcm(german_x_train.shape[-1],german_y_train.shape[-1], fcm_iter=2, l_slope=1)
# nfcm.initialize_loss_and_compile('cce')
# time_callback = TimeHistory()
# callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
# history = nfcm.model.fit(german_x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
# predictions = nfcm.predict_classification(german_x_test)
# print('german')
# nfcm.metrics_classification(german_y_test)





