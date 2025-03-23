import math  # Provides mathematical functions such as exp and log
import random  # Used for generating random initial weights

# A small constant to avoid division by zero or taking log(0)
EPS = 1e-9


class LogiticRegression:
    """
    A simple implementation of logistic regression using gradient descent.

    Mathematical Equations:
    -------------------------
    1. Linear Model & Prediction:
       For a given input vector X = [x1, x2, ..., xn]:
         z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
       The predicted probability (ŷ) is computed by applying the sigmoid function:
         ŷ = σ(z) = 1 / (1 + exp(-z))

    2. Loss Function (Cross-Entropy Loss):
       For a dataset with N samples, the loss is computed as:
         L(y, ŷ) = -∑ [ y * log(ŷ + EPS) + (1 - y) * log(1 - ŷ + EPS) ]
       where y is the true label and ŷ is the predicted probability.

    3. Gradients:
       To update the weights and bias, we use the gradients computed as:
         For weight wⱼ:
           ∂L/∂wⱼ = (1/N) * ∑ [ xᵢⱼ * (ŷᵢ - yᵢ) ]
         For bias b:
           ∂L/∂b = (1/N) * ∑ [ (ŷᵢ - yᵢ) ]
    """
    
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.01):
        """
        Train the logistic regression model using mini-batch gradient descent.

        Parameters:
        - X: List of feature vectors (each is a list of features).
        - y: List of binary target values (0 or 1).
        - epochs: Number of times the entire dataset is processed.
        - batch_size: Number of samples used in each gradient update.
        - lr: Learning rate for updating weights.
        """
        # Determine the number of samples (N) and features (n)
        n_samples, n_features = len(X), len(X[0])
        
        # Randomly initialize the weights (w) and bias (b)
        self.W = [random.random() for _ in range(n_features)]
        self.b = random.random()

        # Loop over epochs
        for e in range(epochs):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                self._update_weights(X_batch, y_batch, lr)
            print(f"EPOCH: {e:>3d}, LOSS: {self.loss(y, self.predict(X)):.8f}")

    def _update_weights(self, X, y, lr):
        """
        Update weights and bias for a given batch using gradient descent.

        Parameters:
        - X: Batch of feature vectors.
        - y: Batch of true target values.
        - lr: Learning rate.
        """
        # Predict probabilities for the current batch
        y_pred = self.predict(X)
        
        # Compute gradients for weights and bias
        dW = self.gradient_W(X, y, y_pred)
        db = self.gradient_b(y, y_pred)
        
        # Update each weight: w = w - lr * (∂L/∂w)
        self.W = [self.W[i] - lr * dW[i] for i in range(len(self.W))]
        # Update bias: b = b - lr * (∂L/∂b)
        self.b = self.b - lr * db

    def gradient_W(self, X, y, _y):
        """
        Compute the gradient of the loss with respect to each weight.

        Uses the equation:
          ∂L/∂wⱼ = (1/N) * ∑ [ xᵢⱼ * (ŷᵢ - yᵢ) ]
        
        Parameters:
        - X: Batch of feature vectors.
        - y: True labels for the batch.
        - _y: Predicted probabilities for the batch.
        
        Returns:
        - A list of gradients for each weight.
        """
        return [
            sum([X[i][j] * (_y[i] - y[i]) for i in range(len(y))]) / len(y)
            for j in range(len(X[0]))
        ]

    def gradient_b(self, y, _y):
        """
        Compute the gradient of the loss with respect to the bias.

        Uses the equation:
          ∂L/∂b = (1/N) * ∑ [ (ŷᵢ - yᵢ) ]
        
        Parameters:
        - y: True labels for the batch.
        - _y: Predicted probabilities for the batch.
        
        Returns:
        - The gradient for the bias.
        """
        return sum([_y[i] - y[i] for i in range(len(y))]) / len(y)

    def loss(self, y, _y):
        """
        Compute the cross-entropy loss over all samples.

        Uses the equation:
          L(y, ŷ) = -∑ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]
        
        Parameters:
        - y: List of true labels.
        - _y: List of predicted probabilities.
        
        Returns:
        - The total loss value.
        """
        return -sum(
            [
                y[i] * math.log(_y[i] + EPS) + (1 - y[i]) * math.log(1 - _y[i] + EPS)
                for i in range(len(y))
            ]
        )

    def predict(self, X):
        """
        Predict probabilities for given feature vectors.

        Steps:
          1. Compute the linear model: z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
          2. Apply the sigmoid function: ŷ = 1 / (1 + exp(-z))
        
        Parameters:
        - X: List of feature vectors.
        
        Returns:
        - A list of predicted probabilities (values between 0 and 1).
        """
        
        linear_model = [self.linear_model(x) for x in X]
        y_pred = self._sigmoid(linear_model)
        return y_pred

    def linear_model(self, X):
        """
        Compute the linear combination (z) of the inputs and weights.

        Equation:
          z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
        
        Parameters:
        - X: A single feature vector.
        
        Returns:
        - The computed linear value.
        """
        return sum([self.W[i] * X[i] for i in range(len(X))]) + self.b

    def _sigmoid(self, x):
        """
        Apply the sigmoid function to a list of values.

        Equation:
          σ(z) = 1 / (1 + exp(-z))
        
        Parameters:
        - x: List of values (usually from the linear model).
        
        Returns:
        - A list of values after applying the sigmoid function.
        """
        return [1 / (1 + math.exp(-x_i)) for x_i in x]


if __name__ == "__main__":
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    y = [0, 0, 0, 1, 1, 1]

    model = LogiticRegression()
    model.fit(X, y, epochs=1000, batch_size=2, lr=0.5)
    
    preds = model.predict([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    print("\nPREDICTIONS")
    for i, p in enumerate(preds):
        print(f"X: {X[i]}, PRED: {p:.4f}")
