import math


class Node:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._fit(X, y, depth=0)

    def _fit(self, X, y, depth):
        if not y:
            return None
        
        if len(set(y)) == 1:
            return Node(y[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(max(set(y), key=y.count))

        N, M = len(X), len(X[0])
        best_col = None
        best_val = None
        best_gain = -float("inf")

        for col in range(M):
            unique_values = set(X[i][col] for i in range(N))
            for val in unique_values:
                left_val = [y[i] for i in range(N) if X[i][col] < val]
                right_val = [y[i] for i in range(N) if X[i][col] >= val]

                if not left_val or not right_val:
                    continue

                left_entropy = self._entropy(left_val)
                right_entropy = self._entropy(right_val)
                total_entropy = self._entropy(y)
                gain = total_entropy - (left_entropy * len(left_val) + right_entropy * len(right_val)) / N

                if gain > best_gain:
                    best_gain = gain
                    best_col = col
                    best_val = val
                
        left_indices = [i for i in range(N) if X[i][best_col] < best_val]
        right_indices = [i for i in range(N) if X[i][best_col] >= best_val]

        X_left = [X[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_left = [y[i] for i in left_indices]
        y_right = [y[i] for i in right_indices]

        node = Node((best_col, best_val))
        node.add_child(self._fit(X_left, y_left, depth + 1))
        node.add_child(self._fit(X_right, y_right, depth + 1))

        return node

    def _entropy(self, y):
        N = len(y)
        entropy = 0
        for label in set(y):
            p = y.count(label) / N
            entropy -= p * math.log(p)
        return entropy

    def predict(self, X):
        return [self._predict(x, self.root) for x in X]

    def _predict(self, x, node):
        if not node.children:
            return node.data

        col, val = node.data
        if x[col] < val:
            return self._predict(x, node.children[0])
        else:
            return self._predict(x, node.children[1])
        
if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    dt = DecisionTree(max_depth=10)
    dt.fit(X, y)
    print(dt.predict(X)) 