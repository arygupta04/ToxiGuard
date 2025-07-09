import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd


class ToxicityClassifier:
    def __init__(self, model_path, max_features=200000, sequence_length=1800):
        """
        Initializes the classifier with a single multi-label model and a text vectorizer.

        Args:
            model_path (str): Path to the multi-label classification model.
            max_features (int): Max number of words in the vocabulary.
            sequence_length (int): Output length of the vectorized sequence.
        """
        # Load the multi-label model
        self.model = tf.keras.models.load_model(model_path)

        # Define label names corresponding to output neurons
        self.labels = ['toxic', 'obscene', 'threat', 'insult', 'identity hate']

        # Load and prepare data for adapting the vectorizer
        df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train.csv')
        X = df['comment_text']

        # Initialize and adapt the text vectorizer
        self.vectorizer = TextVectorization(max_tokens=max_features,
                                            output_sequence_length=sequence_length,
                                            output_mode='int')
        self.vectorizer.adapt(X.values)

    def predict(self, comment):
        """
        Predicts the probabilities for each label from the comment.

        Args:
            comment (str): The comment to classify.

        Returns:
            dict: A dictionary of labels with their predicted probabilities.
        """
        # Vectorize the input comment
        vectorized = self.vectorizer([comment])
        probs = self.model.predict(vectorized, verbose=0)[0]

        # Map predictions to labels
        return dict(zip(self.labels, probs))

    def classify(self, comment, threshold=0.5):
        """
        Returns the list of labels whose probability exceeds the threshold.

        Args:
            comment (str): The comment to classify.
            threshold (float): Minimum confidence to flag a label.

        Returns:
            list: Labels with probability above threshold.
        """
        predictions = self.predict(comment)
        return [label for label, score in predictions.items() if score > threshold]
