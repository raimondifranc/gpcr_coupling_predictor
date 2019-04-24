# Script to perform 5-fold-CV and testing
# By: Gurdeep Singh, Francesco Raimondi & Robert B Russell
# BioQuant, Im Neuenheimer Feld 267, 69120 Heidelberg, Germany
# Contact: gurdeep.singh@bioquant.uni-heidelberg.de
# Execute: python training.py <G-protein>
import os, sys, numpy, math
import pandas as pd
from sklearn.linear_model import LogisticRegression as log_reg
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import matthews_corrcoef as mcc_score
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import f1_score as f1m_score
from sklearn.metrics import precision_score as pre_score
from sklearn.metrics import recall_score as rec_score
from sklearn.metrics import matthews_corrcoef as mcc_score

## Function with metrics' formulas
def metrics(tp, fp, fn, tn, exp, pred):
	try:
		mcc = round(mcc_score(exp, pred), 2)
	except ZeroDivisionError:
		mcc = 'NaN'
	try:	
		acc = round(acc_score(exp, pred), 2)
	except ZeroDivisionError:
		acc = 'NaN'
	try:
		pre = round(pre_score(exp, pred), 2)
	except:
		pre = 'NaN'
	try:					
		rec = round(rec_score(exp, pred), 2)
	except ZeroDivisionError:
		rec = 'NaN'
	try:
		spe = round((float(tn)/(tn+fp)), 2)
	except ZeroDivisionError:
		spe = 'NaN'
	try:
		f1m = round(f1m_score(exp, pred), 2)
	except ZeroDivisionError:
		f1m = 'NaN'
	try:
		if len(exp) != list(exp).count(0) and len(exp) != list(exp).count(0):
			auc = round(auc_score(exp, pred), 2)
		else:
			auc = '-'
	except ZeroDivisionError:
		auc = 'NaN'

	return mcc, acc, pre, rec, spe, f1m, auc

## Function to generate metrics
def testing(df, min_max_scaler_all, model):
	col = list(df.columns.values)
	X = df[col[1:-1]].as_matrix()
	Y = df[col[-1]].as_matrix()
	if Y.tolist().count(1) != 0 and Y.tolist().count(0) != 0:
		X = min_max_scaler_all.transform(X)
		tp, fp, fn, tn, error = check(Y, model.predict(X))
		mcc, acc, pre, rec, spe, f1m, auc = metrics(tp, fp, fn, tn, Y, model.predict(X))
		print str(mcc) +'\t'+ str(acc)+'\t'+ str(pre)+'\t'+ str(rec)+'\t'+ str(spe)+'\t'+ str(f1m)+'\t'+ str(auc) +'\t' + str(Y.tolist().count(1)) + '\t' + str(Y.tolist().count(0))
	else:
		print

# Function to generate stratified
# 5-fold dataframes
def k_fold_sets(df):
	df_new = {}
	i = 1
	l = len(df)
	c = int(float(l)/5)
	while len(df) > 0:
		if len(df) >= c * 2:
			df_new[i] = df.sample(n=c)
			df = df[~df.GPCR.isin(df_new[i].GPCR.tolist())]
		else:
			df_new[i] = df.sample(n=len(df))
			df = df[~df.GPCR.isin(df_new[i].GPCR.tolist())]
		i+=1
	return df_new

# Function to split a dataframe into
# coupling(pos) and un-coupling(neg)
def k_fold(df):
	col = list(df.columns.values)
	df[col[1:-1]] = df[col[1:-1]].astype(float)
	df[col[-1]] = df[col[-1]].astype(int)
	df_pos = k_fold_sets(df[df.O == 1])
	df_neg = k_fold_sets(df[df.O == 0])
	return df_pos, df_neg, col

# Function to convert test files into Pandas' dataframes
def iuphar(file):
	df = pd.read_table(file, lineterminator = '\n', sep = '\t')
	col = list(df.columns.values)
	#print '# Features extracted:', len(col[1:-1])
	df[col[1:-1]] = df[col[1:-1]].astype(float)
	df[col[-1]] = df[col[-1]].astype(int)
	return df

# Function to generate True/False Positives/Negatives
def check(expected, predicted):
	error = []
	tp=0; tn=0; fp=0; fn=0
	for i in range(0, len(predicted)):
		if expected[i]==predicted[i]:
			error.append(1)
			if expected[i] == 1:
				tp+=1
			else:
				tn+=1
		else:
			error.append(0)
			if expected[i] == 1:
				fn+=1
			else:
				fp+=1			
	return tp, fp, fn, tn, error

# Function to find best parameters using
# Grid Search with stratified 5-fold-CV
def grid_search(file):
	df = pd.read_table(file, lineterminator = '\n', sep = '\t')
	col = list(df.columns.values)
	
	df[col[1:-1]] = df[col[1:-1]].astype(float)
	df[col[-1]] = df[col[-1]].astype(int)
	X = df[col[1:-1]].as_matrix()
	X = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X)
	X = X.astype(float, order = 'C')
	Y = df[col[-1]].as_matrix()
	Y = Y.astype(float, order = 'C')
	
	parameters = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0], 'solver' : ['lbfgs'], 'max_iter' : [100, 250, 500, 750, 1000, 1500, 2500]}
	lr = log_reg(penalty='l2', class_weight='balanced')
	model = GridSearchCV(lr, parameters, cv = 5, scoring = 'roc_auc', n_jobs=5)
	model.fit(X, Y)
	return model.best_params_, df

## The main function
def main():
	# Given G-protein name in the command line
	gprotein = sys.argv[1]
	
	# Folder with train and test files
	feat_file = 'data/'
	
	# Find the best parameters from the train file
	best_param, df_orig = grid_search(feat_file+str(gprotein)+'_train.txt')
	
	# Convert test files into pandas' dataframes
	df_iuphar = iuphar(feat_file+str(gprotein)+'_class_A_test_both.txt')
	
	##
	## The code henceforth uses the the best parameters and repeats
	## the entire experiment 10 times to ensure minimal variance
	##
	count = 1; test = []; train = []
	while count <= 10:
		# Split data fram into stratified 5-fold
		# couplind(pos) and un-coupling(neg) sets
		df_pos, df_neg, col = k_fold(df_orig)
		i = 1; M = 0; R = 0; row = []
		data_train = []
		data_test = []
		
		# Perform 5-fold CV
		while i <= 5:
			j = 1
			error = []
			df_test = pd.DataFrame()
			df_train = pd.DataFrame()
			df_all = pd.DataFrame()
			
			# Choose one of the sets as validation set
			# and the other ones as training set
			while j <= 5:
				if i == j:
					df_test = pd.concat([df_test, df_pos[j], df_neg[j]])
					df_all = pd.concat([df_all, df_pos[j], df_neg[j]])
				else:
					df_train = pd.concat([df_train, df_pos[j], df_neg[j]])
					df_all = pd.concat([df_all, df_pos[j], df_neg[j]])
				j+=1
			
			col = list(df_train.columns.values)
			X_train = df_train[col[1:-1]].as_matrix()
			Y_train = df_train[col[-1]].as_matrix()
			X_all = df_all[col[1:-1]].as_matrix()
			Y_all = df_all[col[-1]].as_matrix()
			col = list(df_test.columns.values)
			X_test = df_test[col[1:-1]].as_matrix()
			Y_test = df_test[col[-1]].as_matrix()
			min_max_scaler_train = MinMaxScaler()
			X_train = min_max_scaler_train.fit_transform(X_train)
			X_train = X_train.astype(float, order = 'C')
			Y_train = Y_train.astype(float, order = 'C')
			X_test = min_max_scaler_train.transform(X_test)
			X_test = X_test.astype(float, order = 'C')
			Y_test = Y_test.astype(float, order = 'C')
			
			# Use the best parameters obtained from the Grid Search CV
			model = log_reg(penalty='l2', C=best_param['C'], solver=best_param['solver'], max_iter=best_param['max_iter'], class_weight='balanced')
			
			# Unhash the next line to implement Randomization using Salzberg test
			# Y_train = numpy.random.permutation(Y_train)
			
			# Fit the model over the train set
			model.fit(X_train, Y_train)
			
			# Predictions on the chosen test set
			Y_pred = model.predict(X_test)
			tp, fp, fn, tn, error = check(Y_test, Y_pred)
			
			# Find performance over the validation sets
			mcc, acc, pre, rec, spe, f1m, auc = metrics(tp, fp, fn, tn, Y_test, Y_pred)
			row = []
			row.append(mcc)
			row.append(acc)
			row.append(pre)
			row.append(rec)
			row.append(spe)
			row.append(f1m)
			row.append(auc)
			data_test.append(row)
			
			# Find performace over the train set
			tp, fp, fn, tn, error = check(Y_train, model.predict(X_train))
			mcc, acc, pre, rec, spe, f1m, auc = metrics(tp, fp, fn, tn, Y_train, model.predict(X_train))
			row = []
			row.append(mcc)
			row.append(acc)
			row.append(pre)
			row.append(rec)
			row.append(spe)
			row.append(f1m)
			row.append(auc)
			data_train.append(row)
			
			# Use the best parameters to fit on the
			# entire set and predict over the test set
			if i == 1 and count == 10:
				min_max_scaler_all = MinMaxScaler()
				X_all = min_max_scaler_all.fit_transform(X_all)
				X_all = X_all.astype(float, order = 'C')
				Y_all = Y_all.astype(float, order = 'C')
				
				model = log_reg(penalty='l2', C=best_param['C'], solver=best_param['solver'], max_iter=best_param['max_iter'], class_weight='balanced')

				model.fit(X_all, Y_all)
				tp, fp, fn, tn, error = check(Y_all, model.predict(X_all))
				mcc, acc, pre, rec, spe, f1m, auc = metrics(tp, fp, fn, tn, Y_all, model.predict(X_all))
				
				print 'SET\t\tMCC\tACC\tPRE\tREC\tSPE\tF1M\tAUC\t+ve\t-ve'
				print  'Training set', '\t', str(mcc) +'\t'+ str(acc)+'\t'+ str(pre)+'\t'+ str(rec)+'\t'+ str(spe)+'\t'+ str(f1m)+'\t'+ str(auc) + '\t' + str(Y_all.tolist().count(1)) + '\t' + str(Y_all.tolist().count(0))
				
				#### Prediction over the test setIuphar. Same model - all
				print 'Test set','\t',
				testing(df_iuphar, min_max_scaler_all, model)
			i+=1
		test.append(numpy.mean(numpy.array(data_test), axis = 0))
		train.append(numpy.mean(numpy.array(data_train), axis = 0))
		count += 1
	print '5-fold-CV', '\t',
	for num in numpy.around(numpy.mean(numpy.array(test), axis = 0), decimals = 2):
		print str(num) + '\t',
	print '-\t-'
	print '5-fold-CV_STD', '\t',
	for num in numpy.around(numpy.std(numpy.array(test), axis = 0), decimals = 2):
		print str(num) + '\t',
	print '-\t-'
	sys.exit()

main()
sys.exit()
## End of script
