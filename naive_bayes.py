import math
import string
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = set()
        self.vocab = set()
        self.word_freqs = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.class_probs = {}

    def preprocess(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.lower().translate(translator).split()

    def fit(self, X, y):
        for doc, label in zip(X, y):
            self.classes.add(label)
            self.class_counts[label] += 1
            words = self.preprocess(doc)
            for word in words:
                self.vocab.add(word)
                self.word_freqs[label][word] += 1

        total_docs = sum(self.class_counts.values())
        for c in self.classes:
            self.class_probs[c] = math.log(self.class_counts[c] / total_docs)

    def predict(self, X):
        predictions = []
        for doc in X:
            words = self.preprocess(doc)
            class_scores = {}

            for c in self.classes:
                # Start with class log probability
                score = self.class_probs[c]
                total_count = sum(self.word_freqs[c].values())

                # Add log probabilities for each word
                for word in words:
                    word_count = self.word_freqs[c][word] + 1  # Laplace smoothing
                    word_prob = math.log(word_count / total_count)
                    score += word_prob
                class_scores[c] = score

            # Pick the class with the highest probability
            predictions.append(max(class_scores, key=class_scores.get))

        return predictions

# Example usage:
if __name__ == "__main__":
    # Example dataset
    X_train = [
        "I love this movie",
        "This is an amazing film",
        "I hated this movie",
        "This was a terrible film"
    ]
    y_train = ["positive", "positive", "negative", "negative"]

    X_test = [
        "This film is great",
        "This movie was terrible"
    ]

    # Train classifier
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    # Predict
    predictions = nb.predict(X_test)
    print(predictions)  # Output: ['positive', 'negative']
