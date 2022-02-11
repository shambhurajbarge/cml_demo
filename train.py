from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# model
depth = 5
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
disp = plot_confusion_matrix(clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("plot.png")

# classification_report
prediction = clf.predict(X_test)
class_score = classification_report(y_test, prediction)

with open("classification_report.txt", "w") as outfile:
    outfile.write("classification_report: " + str(class_score) + "\n")
