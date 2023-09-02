import numpy as np
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( Z_trn ):
################################
#  Non Editable Region Ending  #
################################
	X_trn = Z_trn[:, 0:64]
	Y_trn = Z_trn[:, 72]
	P_trn = Z_trn[:, 64:68]
	Q_trn = Z_trn[:, 68:72]
	X_new = X_trn
	X_new = np.append(X_new, np.array([Y_trn]).T, axis=1)

	n_models = 255
	i = n_models
	data = []
	while(i):
		data.append([])
		i-=1

	v = X_new.shape[0]
	for i in range(v):
		p = P_trn[i][3]*8 + P_trn[i][2]*4 + P_trn[i][1]*2 + P_trn[i][0]
		q = Q_trn[i][3]*8 + Q_trn[i][2]*4 + Q_trn[i][1]*2 + Q_trn[i][0]
		if p>q:
			p, q = q, p
			# swap
			X_new[i][-1] = 1 - X_new[i][-1]
		index = int(16*p + q)
		data[index].append(X_new[i])

	training_data = []
	for t in data:
		training_data.append(np.array(t))

	models = []
	for j in range(len(training_data)):
		models.append([])
		if(len(training_data[j]) != 0):
			models[j] = LinearSVC(tol = 0.1, C=1, max_iter = 100)
			x = training_data[j][:, 0:-1]
			y = np.array([training_data[j][:, -1]]).T

			models[j].fit(x, y)

	return models

################################
# Non Editable Region Starting #
################################
def my_predict( Z_tst, model ):
################################
#  Non Editable Region Ending  #
################################
	X_tst = Z_tst[:, 0:64]
	P_tst = Z_tst[:, 64:68]
	Q_tst = Z_tst[:, 68:72]

	v = X_tst.shape[0]
	pred = np.ones(v)

	for i in range(v):
		p = P_tst[i][3]*8 + P_tst[i][2]*4 + P_tst[i][1]*2 + P_tst[i][0]
		q = Q_tst[i][3]*8 + Q_tst[i][2]*4 + Q_tst[i][1]*2 + Q_tst[i][0]
		swp=0
		if p>q:
			p, q = q, p
			swp = 1
		indx = int(16*p + q)
		pred[i] = model[indx].predict(np.array([X_tst[i]]))
		if swp == 1:
			pred[i] = 1 - pred[i]
	return pred