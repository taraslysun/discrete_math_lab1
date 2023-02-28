import numpy as np

class Node:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class MyDecisionTreeClassifier:
    def __init__(self, max_depth = 10):
        self.max_depth = max_depth


    def split_data(self, X, y):
        '''
        Splits data by Gini criteria
        '''
        m = y.size

        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes)]

        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_feats):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]

                num_left[c] += 1
                num_right[c] -= 1

                # Gini itself
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr


    def build_tree(self, X, y, depth = 0):
        '''
        Builds a tree from scratch recursively
        '''
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(
            X = y.size,
            y = predicted_class
        )

        if depth < self.max_depth:
            idx, thr = self.split_data(X, y)

            if idx:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                node.feature_index = idx
                node.threshold = thr

                node.left = self.build_tree(X_left, y_left, depth + 1)
                node.right = self.build_tree(X_right, y_right, depth + 1)

        return node


    def fit(self, X, y):
        '''
        Builds a tree
        '''
        # Dimension of tree
        self.n_classes = len(set(y))
        self.n_feats = X.shape[1]

        self.tree = self.build_tree(X, y)


    def predict(self, X):
        '''
        Predicts for one elem
        '''
        return [self.predict_value(params) for params in X]


    def predict_value(self, params):
        '''
        Predicts for whole dataset
        '''
        node = self.tree

        while node.left:
            if params[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.y
