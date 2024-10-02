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

#new lines
cad2_experts_data = os.path.join(data, 'new_cad_with_experts', 'cad_full_with_artificial_samples.csv') #binary
cad2_data = os.path.join(data, 'new_cad_no_experts', 'cad_full_with_artificial_samples.csv') #binary
labels1_data = os.path.join(data, 'labels1', 'labels1.csv') #binary


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

##new data
cad2_experts = pd.read_csv(cad2_experts_data, delimiter = ';', decimal = ',')
cad2_experts.dropna(inplace = True)

cad2 = pd.read_csv(cad2_data, delimiter = ';', decimal = ',')
cad2.dropna(inplace = True)

labels1 = pd.read_csv(labels1_data, delimiter = ';', decimal = ',')
labels1.drop(['id'], inplace = True, axis = 1)
labels1.fillna("missing", inplace = True)
text_columns_labels1 = labels1.select_dtypes(include='object').columns
for column in text_columns_labels1:
	labels1 = text_to_numerical(labels1, column)


## categorical dataframes
wdbc_categorical = convert_to_categorical(wdbc, wdbc.columns[0])
cad_categorical = convert_to_categorical(cad, cad.columns[-1])
iris_categorical = convert_to_categorical(iris, iris.columns[-1])
diabetes_categorical = convert_to_categorical(diabetes, diabetes.columns[-1])
thyroid_categorical = convert_to_categorical(thyroid, thyroid.columns[0])
wine_categorical = convert_to_categorical(wine, wine.columns[0])
german_categorical = convert_to_categorical(german, german.columns[-1])

##new data
cad2_experts_categorical = convert_to_categorical(cad2_experts, cad2_experts.columns[-1])
cad2_categorical = convert_to_categorical(cad2, cad2.columns[-1])
labels1_categorical = convert_to_categorical(labels1, labels1.columns[-1])

## normalize 
wdbc_categorical = min_max_scaling(wdbc_categorical)
cad_categorical = min_max_scaling(cad_categorical)
iris_categorical = min_max_scaling(iris_categorical)
diabetes_categorical = min_max_scaling(diabetes_categorical)
thyroid_categorical = min_max_scaling(thyroid_categorical)
wine_categorical = min_max_scaling(wine_categorical)
german_categorical = min_max_scaling(german_categorical)


cad2_experts_categorical = min_max_scaling(cad2_experts_categorical)
cad2_categorical = min_max_scaling(cad2_categorical)
labels1_categorical = min_max_scaling(labels1_categorical)

## split input/labels
wdbc_input, wdbc_labels = split_labels(wdbc_categorical, -2)
cad_input, cad_labels = split_labels(cad_categorical, -2)
iris_input, iris_labels = split_labels(iris_categorical, -3)
diabetes_input, diabetes_labels = split_labels(diabetes_categorical, -2)
thyroid_input, thyroid_labels = split_labels(thyroid_categorical, -3)
wine_input, wine_labels = split_labels(wine_categorical, -3)
german_input, german_labels = split_labels(german_categorical, -2)

cad2_experts_input, cad2_experts_labels = split_labels(cad2_experts_categorical, -2)
cad2_input, cad2_labels = split_labels(cad2_categorical, -2)
labels1_input, labels1_labels = split_labels(labels1_categorical, -2)

dfs_inputs = [
	wdbc_input,
	cad_input,
	iris_input,
	diabetes_input,
	thyroid_input,
	wine_input,
	german_input,
	cad2_experts_input,
	cad2_input,
	labels1_input,

]
dfs_labels = [
	wdbc_labels,
	cad_labels,
	iris_labels,
	diabetes_labels,
	thyroid_labels,
	wine_labels,
	german_labels,
	cad2_experts_labels,
	cad2_labels,
	labels1_labels

]

dfs_names = [
	'wdbc',
	'cad',
	'iris',
	'diabetes',
	'thyroid',
	'wine',
	'german',
	'cad2_experts',
	'cad2',
	'labels1'
]

cwd = os.getcwd()
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M")
path = os.path.join('experiments', dt_string)
os.mkdir(path)

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
#inside dataset
for i in range(len(dfs_names)):
	kf = KFold(n_splits = 10, shuffle = True)
	fold = 1
	path_dataset = os.path.join(path, dfs_names[i])
	os.mkdir(path_dataset)
	ls = [1, 2, 3, 4, 5]
	iters = [1, 2, 3, 4, 5, 6]
	stats = {f'l = {l}, iter = {i}' : {'Accuracy' : [], 'average_epoch_time' : [], 'epochs' : [], 'f1_macros' : [], 'conf_matr' : []} for l in ls for i in iters}

	#inside fold
	for train_index, test_index in kf.split(dfs_inputs[i]):
		x_train, y_train = dfs_inputs[i].iloc[train_index].to_numpy(), dfs_labels[i].iloc[train_index].to_numpy()
		x_test, y_test = dfs_inputs[i].iloc[test_index].to_numpy(), dfs_labels[i].iloc[test_index].to_numpy()
		path_fold = os.path.join(path_dataset, f'Fold_{fold}')
		os.mkdir(path_fold)
		print(f'Dataset {dfs_names[i]}, fold {fold}\n')
		for l in ls:
			for it in iters:
				train_y = np.concatenate([x_train,y_train], axis = -1)
				nfcm = neural_fcm(x_train.shape[-1],y_train.shape[-1], fcm_iter=it, l_slope=l)
				nfcm.initialize_loss_and_compile('cce')
				time_callback = TimeHistory()
				callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True), time_callback]
				history = nfcm.model.fit(x_train, train_y, batch_size=64, epochs = 2000, validation_split = 0.2, callbacks = callbacks)
				predictions = nfcm.predict_classification(x_test)
				nfcm.metrics_classification(y_test)

				stats[f'l = {l}, iter = {it}']['Accuracy'].append(nfcm.accuracy)
				stats[f'l = {l}, iter = {it}']['average_epoch_time'].append(np.mean(time_callback.times))
				stats[f'l = {l}, iter = {it}']['epochs'].append(len(time_callback.times))
				stats[f'l = {l}, iter = {it}']['f1_macros'].append(nfcm.f1_score_macro)
				stats[f'l = {l}, iter = {it}']['conf_matr'].append(nfcm.confusion_matrix)
				
				
				disp = ConfusionMatrixDisplay(confusion_matrix=nfcm.confusion_matrix)
				disp.plot()
				plt.savefig(os.path.join(path_fold, f'l{l}_it{it}_conf_matrix.png'), dpi = 600)
				plt.show()

				plt.plot(history.history['loss'], color = 'b', label = 'train_loss')
				plt.plot(history.history['val_loss'], color = 'c', label = 'val_loss')
				plt.legend()
				plt.xlabel('Epochs')
				plt.ylabel('Classification Loss')
				plt.title(f'Training ({dfs_names[i]} dataset)')
				plt.savefig(os.path.join(path_fold, f'l{l}_it{it}_training_plot.png'), dpi = 600)
				plt.show()

				with open(os.path.join(path_fold, f'l{l}_it{it}_stats.txt'), 'w') as f:
					f.write(f'Accuracy = {nfcm.accuracy}\n')
					f.write(f'F1_score (macro) = {nfcm.f1_score_macro}\n')
					f.write(f'Confusion Matrix = {nfcm.confusion_matrix}\n')
					f.write(f'average time per epoch = {np.mean(time_callback.times)} ms\n')
					f.write(f'Epochs = {len(history.history["loss"])}\n')
		
		fold += 1
	
	with open(os.path.join(path_dataset, f'stats_dataset_{fold-1}fold.txt'), 'w') as f:
		
		#arrays to find the best accuracies
		ks = []
		vals_acc = []
		vals_f1 = []
		vals_tt = []
		for key in stats.keys():
			ks.append(key)
			f.write(f'{key} Average Accuracy = {np.mean(stats[key]["Accuracy"])}\n')
			vals_acc.append(np.mean(stats[key]["Accuracy"]))

			f.write(f'{key} Average F1-score(macro) = {np.mean(stats[key]["f1_macros"])}\n')
			vals_f1.append(np.mean(stats[key]["f1_macros"]))

			f.write(f'{key} Average time per epoch = {np.mean(stats[key]["average_epoch_time"])} ms\n')
			f.write(f'{key} Average epochs = {np.mean(stats[key]["epochs"])}\n')

			average_time = np.mean(stats[key]['average_epoch_time']) * np.mean(stats[key]['epochs'])
			f.write(f'{key} Total Average Time (average epochs x average time per epochs)  = {average_time} (s)\n')
			vals_tt.append(average_time)
		ks = np.array(ks)
		f.write(f'\nBest Accuracy = {np.max(vals_acc)} found in {ks[np.where(vals_acc == np.max(vals_acc))[0]]}\n')
		f.write(f'\nBest F1-score (macro) = {np.max(vals_f1)} found in {ks[np.where(vals_f1 == np.max(vals_f1))[0]]}\n')
		f.write(f'\nworst Accuracy = {np.min(vals_acc)} found in {ks[np.where(vals_acc == np.min(vals_acc))[0]]}\n')
		f.write(f'\nworst F1-score (macro) = {np.min(vals_f1)} found in {ks[np.where(vals_f1 == np.min(vals_f1))[0]]}\n')
		

	try:
		with open(os.path.join(path_dataset, f'stats_dataset_{fold-1}fold.json'), 'w') as json_file:
			for ii in stats.keys():
				del stats[ii]['conf_matr']
			json.dump(stats, json_file, indent=10)
	except Exception as e:
		print(e)













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





