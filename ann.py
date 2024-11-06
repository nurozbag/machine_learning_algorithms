import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pyp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier


data_set= pd.read_csv("C:\\Users\\Pc\\Desktop\\ML_project\\Epileptic Seizure Recognition.csv")
data_set = data_set.replace(0, pd.NA).dropna()
X = data_set.iloc[:,1:-1].values
y = data_set.iloc[:,-1:].values
X, y = make_classification(n_samples=11500, n_features=180, n_informative=100, n_classes=6, random_state=42)
kfold = KFold(n_splits=40, shuffle=True, random_state=42)
accuracy_scores = []
for train_index, test_index in kfold.split(X, y):
    X_kfold, X_test = X[train_index], X[test_index]
    y_kfold, y_test = y[train_index], y[test_index]
    X_train, _, y_train, _ = train_test_split(X_kfold, y_kfold, test_size=0.3, shuffle=True, random_state=42)
    y_train_categorical = to_categorical(y_train, num_classes=6)
    y_test_categorical = to_categorical(y_test, num_classes=6)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 

ann_model = Sequential()
ann_model.add(Dense(units=512,activation='relu'))
ann_model.add(Dense(units=512,activation='relu'))
ann_model.add(Dense(units=128,activation='relu'))
ann_model.add(Dense(units=128,activation='relu'))
ann_model.add(Dense(units=128,activation='relu'))
ann_model.add(Dense(units=64,activation='relu'))
ann_model.add(Dense(units=6,activation='softmax'))
ann_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
chckpt = ModelCheckpoint(filepath='C:\\Users\\Pc\\Desktop\\ML_project"/check_point' , verbose=1, save_best_only=True)

ann_model.fit(X_train,y_train_categorical,batch_size=37,epochs = 150,validation_split=(0.2),callbacks=[early],verbose=1)
#estimator = KerasClassifier(build_fn=ann_model, epochs=200, batch_size=5, verbose=0)
accuracy_score = ann_model.evaluate(X_test, y_test_categorical)
accuracy_scores.append(accuracy_score[1])
print(accuracy_score,kfold)
print("Mean Accuracy:", np.mean(accuracy_scores))
classifier = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(512, 512, 128, 128, 128, 64),
                                               max_iter=1000,
                                               random_state=42))

# Binarize the output
y_train_binary = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5])
y_test_binary = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])

# Train the classifier
classifier.fit(X_train, y_train_binary)

# Get predicted probabilities for each class
y_score = classifier.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(6):  # Assuming you have 6 classes
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(6):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
plt.legend(loc="lower right")
plt.show()
#results = cross_val_score(estimator, X, y, cv=kfold)
ann_model.save("C:\\Users\\Pc\\Desktop\\ML_project")