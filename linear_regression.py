import math
import random

class LinearRegression:
    """
    A simple implementation of linear regression using gradient descent.
    
    Mathematical Equations:
    -------------------------
    1. Linear Model:
       For an input vector X = [x₁, x₂, ..., xₙ]:
         y_pred = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b

    2. Loss Function (Mean Squared Error - MSE):
       For a dataset with N samples, the loss is computed as:
         L(y, y_pred) = (1/(2N)) * ∑ (yᵢ - y_predᵢ)²
       The factor 1/2 is used for convenience when computing gradients.

    3. Gradients:
       The gradients for the weights and bias are derived as follows:
         ∂L/∂wⱼ = (1/N) * ∑ (y_predᵢ - yᵢ) * xᵢⱼ
         ∂L/∂b  = (1/N) * ∑ (y_predᵢ - yᵢ)
    """
    
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.01):
        """
        Train the linear regression model using mini-batch gradient descent.
        
        Parameters:
        - X: List of feature vectors (each is a list of features).
        - y: List of target values.
        - epochs: Number of passes through the entire dataset.
        - batch_size: Number of samples used in each gradient update.
        - lr: Learning rate for updating weights.
        """
        n_samples, n_features = len(X), len(X[0])
        self.W = [random.random() for _ in range(n_features)]
        self.b = random.random()
        
        for e in range(epochs):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                self._update_weights(X_batch, y_batch, lr)
            print(f"EPOCH: {e:02d}, LOSS: {self.loss(y, self.predict(X)):.8f}")

    def _update_weights(self, X, y, lr):
        """
        Update the weights and bias for a given mini-batch using gradient descent.
        
        Parameters:
        - X: Batch of feature vectors.
        - y: Batch of target values.
        - lr: Learning rate.
        """
        # Compute predictions for the current batch
        y_pred = self.predict(X)
        
        # Calculate gradients for weights and bias
        dW = self.gradient_W(X, y, y_pred)
        db = self.gradient_b(y, y_pred)
        
        # Update weights: w = w - lr * (∂L/∂w)
        self.W = [self.W[i] - lr * dW[i] for i in range(len(self.W))]
        # Update bias: b = b - lr * (∂L/∂b)
        self.b = self.b - lr * db

    def gradient_W(self, X, y, y_pred):
        """
        Compute the gradient of the loss with respect to each weight.
        
        Uses the equation:
          ∂L/∂wⱼ = (1/N) * ∑ (y_predᵢ - yᵢ) * xᵢⱼ
        
        Parameters:
        - X: Batch of feature vectors.
        - y: True target values for the batch.
        - y_pred: Predicted values for the batch.
        
        Returns:
        - A list of gradients for each weight.
        """
        N = len(y)
        return [
            sum((y_pred[i] - y[i]) * X[i][j] for i in range(N)) / N
            for j in range(len(X[0]))
        ]

    def gradient_b(self, y, y_pred):
        """
        Compute the gradient of the loss with respect to the bias.
        
        Uses the equation:
          ∂L/∂b = (1/N) * ∑ (y_predᵢ - yᵢ)
        
        Parameters:
        - y: True target values for the batch.
        - y_pred: Predicted values for the batch.
        
        Returns:
        - The gradient for the bias.
        """
        N = len(y)
        return sum(y_pred[i] - y[i] for i in range(N)) / N

    def loss(self, y, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss.
        
        Uses the equation:
          L(y, y_pred) = (1/(2N)) * ∑ (yᵢ - y_predᵢ)²
        
        Parameters:
        - y: List of true target values.
        - y_pred: List of predicted values.
        
        Returns:
        - The total loss value.
        """
        N = len(y)
        return sum((y[i] - y_pred[i]) ** 2 for i in range(N)) / (2 * N)

    def predict(self, X):
        """
        Predict target values for given feature vectors.
        
        For each sample, computes:
          y_pred = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
        
        Parameters:
        - X: List of feature vectors.
        
        Returns:
        - A list of predicted values.
        """
        # Compute the linear model for each sample
        return [self.linear_model(x) for x in X]

    def linear_model(self, X):
        """
        Compute the linear combination (z) of the inputs and weights.
        
        Equation:
          z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b
        
        Parameters:
        - X: A single feature vector.
        
        Returns:
        - The computed linear value (prediction).
        """
        return sum(self.W[i] * X[i] for i in range(len(X))) + self.b


if __name__ == "__main__":
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    y = [3, 5, 7, 9, 11, 13]  

    model = LinearRegression()
    model.fit(X, y, epochs=1000, batch_size=2, lr=0.007)
    
    # Generate predictions for the dataset
    preds = model.predict(X)

    print("\nPREDICTIONS")
    for i, p in enumerate(preds):
        print(f"X: {X[i]}, PRED: {p:.4f}")
