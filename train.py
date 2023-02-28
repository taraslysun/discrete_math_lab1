from sklearn import datasets
from sklearn.model_selection import train_test_split
from tree import MyDecisionTreeClassifier

data = datasets.load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = MyDecisionTreeClassifier(max_depth = 10)
clf.fit(X_train, y_train)
prediction = clf.predict([X_test[1]])

print(f'Class: {prediction[0]} for params {list(X_test[1])}')