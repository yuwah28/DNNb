# Code References:
# https://www.datacamp.com/community/tutorials/deep-learning-python#gs.5fOWGnE
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

import h5py
import numpy as np
import matplotlib.pyplot as plt

#
# Read Data from hdf5
#
filepath = '/Users/yuwahlim/Arxiv/Programming/MLtutorial/DNNbtag/hdf5Data/'
f = h5py.File(filepath+'gjj_Variables.hdf5', 'r')

# Define variables
# Structure of high_input[6391,1,16]: 6391 samples, 16 variables 
# Structure of y_intput[6391,3]: 6391 samples, 3 discrete values 
array_size = 10000
high = f['high_input'][0:array_size]  # expert level variable
y = f['y_input'][0:array_size]        # particle type
mask = ~np.isnan(high).any(axis=2)    # Remove NAN
high_input = high[mask[:,0],...]
y_input = y[mask[:,0],...]
'''
print("high_input", high_input[4,0,:])
print("lenght", len(high_input))
print("high shape", high_input.shape[2])
print("y shape", y_input.shape[1])
print("y_input", y_input[6,:])
'''
# Shaffling training and test set and standardized
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
TestSize=0.33
RandomState=23
X = high_input[:,0,0:16]
y = y_input[:,0:3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TestSize, random_state=RandomState)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
print("X_test",X_test)
print("X_train",X_train)
print("y_test",y_test)
print("y_train",y_train)
print("X_test shape",X_test.shape)
print("X_train shape",X_train.shape)
print("y_test shape",y_test.shape)
print("y_train shape",y_train.shape)
print("y_train class",y_train.shape[1])
'''
#
# Build NNlayer using Keras
#
print("Building Keras Model (Feedforward)...")
nepochs = 100
BatchSize = 100
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam         # Add optimizer
model = Sequential()
model.add(Dense(45, activation='relu', input_dim=16))
model.add(Dropout(0.3))
model.add(Dense(45, activation='relu'))   # Add first hidden layer                         
model.add(Dropout(0.3))
model.add(Dense(45, activation='relu'))   # Add second hidden layer                        
#model.add(Dropout(0.2))                                                                   
model.add(Dense(45, activation='relu'))   # Add third hidden layer                         
#model.add(Dropout(0.2))                                                                   
model.add(Dense(45, activation='relu'))   # Add fourth hidden layer                        
#model.add(Dropout(0.2))                                                                   
model.add(Dense(45, activation='relu'))   # Add fifth hidden layer                         
#model.add(Dropout(0.2))                                                                   
model.add(Dense(45, activation='relu'))   # Add sixth hidden layer                         
#model.add(Dropout(0.2))                                                                   
model.add(Dense(45, activation='relu'))   # Add seventh hidden layer                       
#model.add(Dropout(0.2))                                                                   
model.add(Dense(45, activation='relu'))   # Add eighth hidden layer                        
#model.add(Dropout(0.2))                                                                   
model.add(Dense(40, activation='relu'))   # Add ninth hidden layer                         
#model.add(Dropout(0.2))                                                                   
model.add(Dense(3, activation='softmax')) # Add an output layer                            
model.summary()

# Configure learning process  
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, y_train,epochs=nepochs, batch_size=BatchSize, verbose=1)
y_pred = model.predict(X_test)
print("y_pred shape",y_pred.shape)

#
# Add ROC curve and ROC area for each class
#
print("Start ROC calculation...")
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
Nclasses = y.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(Nclasses):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Nclasses)])) # Aggregate all fpr
mean_tpr = np.zeros_like(all_fpr) 

#interpolate all ROC curves
for i in range(Nclasses):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# averaging and calculate AUC
mean_tpr /= Nclasses 
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
print("Plotting ROC...")
particle_pid = ['b','c','light']
plt.figure()
lw=2
colors = cycle(['aqua', 'cornflowerblue', 'orange'])
for i, color in zip(range(Nclasses), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(particle_pid[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
