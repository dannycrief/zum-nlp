import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class SentimentAnalysisPreprocessor:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    @staticmethod
    def average_vector(words, word_vectors):
        word_vectors = [word_vectors[word] for word in words if word in word_vectors]
        if len(word_vectors) == 0:
            return None
        return normalize(sum(word_vectors).reshape(1, -1))

    def preprocess(self):
        # Load the collected data
        df = pd.read_csv(self.input_file)

        # Perform data cleaning and preprocessing
        df['cleaned_title'] = df['title'].apply(lambda x: self.clean_text(x))

        # Tokenize text
        df['tokenized_title'] = df['cleaned_title'].apply(lambda x: x.split())

        # Create word embeddings using a pre-trained model (e.g., Word2Vec)
        model = Word2Vec(df['tokenized_title'].tolist(), vector_size=100, window=5, min_count=1, workers=4)

        # Get the vector representation of each word in the titles
        word_vectors = model.wv

        # Get the average vector for each title
        df['avg_vector'] = df['tokenized_title'].apply(
            lambda x: self.average_vector(x, word_vectors)
            .tolist() if self.average_vector(x, word_vectors) is not None else None)
        df = df.dropna(subset=['avg_vector'])

        # Manually label a portion of the data for training the classifier
        # Add a new column 'manual_label' to the DataFrame and set it to None
        df['manual_label'] = None

        # Manually label some examples (replace the index and label with appropriate values)
        # 0 - negative, 1 - neutral, 2 - positive
        df.at[0, 'manual_label'] = 2
        df.at[5, 'manual_label'] = 0
        # ...

        # Split the manually labeled data into training and testing sets
        labeled_data = df.dropna(subset=['manual_label'])
        X_train, X_test, y_train, y_test = train_test_split(
            np.vstack(labeled_data['avg_vector'].values),
            labeled_data['manual_label'].values.astype(int),
            test_size=0.3,
            random_state=42)

        # Train a classifier (e.g., logistic regression) on
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate the classifier on the test set
        y_pred = clf.predict(X_test)
        print(f"Classifier accuracy: {accuracy_score(y_test, y_pred)}")

        # Assign sentiment labels to the remaining data using the trained classifier
        unlabeled_data = df[df['manual_label'].isnull()]
        unlabeled_data['predicted_label'] = clf.predict(np.vstack(unlabeled_data['avg_vector'].values))

        # Combine the manually labeled and predicted data
        labeled_data['predicted_label'] = labeled_data['manual_label']
        result_df = pd.concat([labeled_data, unlabeled_data], ignore_index=True)

        # Save the preprocessed data
        result_df.to_csv(self.output_file, index=False, lineterminator='\n', float_format='%.8f', header=True)
