import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class SentimentAnalysisPreprocessor:

    def __init__(self, input_file, output_file, num_clusters=3):
        self.input_file = input_file
        self.output_file = output_file
        self.num_clusters = num_clusters

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
        df = pd.read_csv(self.input_file, sep='\t')

        # Perform data cleaning and preprocessing
        df['cleaned_title'] = df['title'].apply(lambda x: self.clean_text(x))

        # Tokenize text
        df['tokenized_title'] = df['cleaned_title'].apply(lambda x: x.split())

        # Create word embeddings using a pre-trained model (e.g., Word2Vec)
        model = Word2Vec(df['tokenized_title'].tolist(), vector_size=100, window=5, min_count=1, workers=4)

        # Get the vector representation of each word in the titles
        word_vectors = model.wv

        # Get the average vector for each title
        # df['avg_vector'] = df['tokenized_title'].apply(lambda x: self.average_vector(x, word_vectors))
        df['avg_vector'] = df['tokenized_title'].apply(
            lambda x: self.average_vector(x, word_vectors)
            .tolist() if self.average_vector(x, word_vectors) is not None else None)
        df = df.dropna(subset=['avg_vector'])

        # Apply K-Means clustering to the embeddings
        X = np.vstack(df['avg_vector'].values)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(X)

        # Assign sentiment labels to the clusters
        df['cluster'] = kmeans.labels_

        # Save the preprocessed data
        df.to_csv(self.output_file, index=False, lineterminator='\n', float_format='%.8f', header=True, sep='\t')
