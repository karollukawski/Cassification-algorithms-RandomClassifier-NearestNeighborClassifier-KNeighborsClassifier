from random import sample
import numpy as np
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, accuracy_score
from sklearn import datasets
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

class RandomClassifier():

    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes, licznoscKlas = np.unique(y, return_counts=True)
        major_class = np.argmax(licznoscKlas, axis = 0)
        self.X_, self.y_ = X, y
        self.classes = major_class
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        closest = np.argmin(euclidean_distances(X,self.X_), axis = 1)
        return self.y_[closest]
        
class NearestNeighborClassifier:
    def fit(self, features, labels):
        self.features_train = features
        self.labels_train = labels

    def predict(self, features_test):
        predictions = []

        for row in features_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = distance.euclidean(row, self.features_train[0])
        best_index = 0

        for i in range(0, len(self.features_train)):
            dist = distance.euclidean(row, self.features_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.labels_train[best_index]

X, y = datasets.make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_repeated=0,
    n_redundant=0,
    random_state=1024,
    )


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=1024)

clf = RandomClassifier()
clf.fit(X_train, y_train)

predict = clf.predict(X_test)
print ("Percent %.3f " % accuracy_score(y_test, predict))

clf = NearestNeighborClassifier()
clf.fit(X_train, y_train)

predict = clf.predict(X_test)
print ("Percent neighbour %.3f " % accuracy_score(y_test, predict))

sasiad = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='brute')
clf.fit(X_train, y_train)

predict = clf.predict(X_test)
print ("Percent neighbour SKL %.3f " % accuracy_score(y_test, predict))


