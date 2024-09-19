from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql import Row
from pyspark.mllib.clustering import KMeans
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

nltk.download('vader_lexicon')

# Step1: Initialize PySpark session
spark = SparkSession.builder.appName("TweetClustering").getOrCreate()

# step 2: Load the CSV file with tweets
# Assuming you have a CSV file with a "tweets" column
tweets_df = spark.read.csv("Data/tweets_cnn_clean_short.csv", header=True)
tweets_rdd = tweets_df.rdd.map(lambda row: row['tweet_clean'])

# Step 3: Preprocess the Tweets for TF-IDF
def preprocess_tweets(text):
    if text is None:
        return ""  # Handle None values
    return text.lower()

preprocessed_rdd = tweets_rdd.map(preprocess_tweets)

# Convert RDD of strings to RDD of Rows with a named column for the DataFrame
preprocessed_rows_rdd = preprocessed_rdd.map(lambda tweet: Row(tweet_clean=tweet))

# Convert the RDD of Rows to a Dataframe
tweets_df = spark.createDataFrame(preprocessed_rows_rdd)

# Step 4: tokenize the tweets
tokenizer = Tokenizer(inputCol="tweet_clean", outputCol="words")
tokenized_df = tokenizer.transform(tweets_df)

# Step 5: Convert text to vector (TF-IDF)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)
featurized_df = hashingTF.transform(tokenized_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# Convert the TF-IDF DataFrame to RDD for clustering
tfidf_rdd = tfidf_df.select("features").rdd.map(lambda row: row.features.toArray())

# step 6:method 1 - KMeans Clustering
kmeans_model = KMeans.train(tfidf_rdd, k=3, maxIterations=10, initializationMode="random")
kmeans_labels = kmeans_model.predict(tfidf_rdd).collect()  # Collect labels for plotting

#step 7: SentimentAnalysis using VADER
sid = SentimentIntensityAnalyzer()

# Compute sentiment scores for each tweet
def compute_sentiment(tweet):
    return sid.polarity_scores(tweet)['compound']

sentiment_rdd = preprocessed_rdd.map(lambda tweet: compute_sentiment(tweet))

#Collect sentiment scores as a list for combining with TF-IDF
sentiment_scores = np.array(sentiment_rdd.collect()).reshape(-1, 1)

# Step 8: Combine TF IDF features with sentiment scores
tfidf_array = np.array(tfidf_rdd.collect())
combined_features = np.hstack((tfidf_array, sentiment_scores))  # combine TF-IDF features and sentiment scores

# Step 9: Clustering on Combined Features (TF-IDF + Sentiment Scores)
kmeans_with_sentiment_model = KMeans.train(spark.sparkContext.parallelize(combined_features), k=3, maxIterations=10, initializationMode="random")
kmeans_sentiment_labels = kmeans_with_sentiment_model.predict(spark.sparkContext.parallelize(combined_features)).collect()

# Dimensionality Reduction (PCA) for Visualization
pca = PCA(n_components=2)
tfidf_2d = pca.fit_transform(combined_features)

# Step 10: Plot Clustering Results forKMeans with Sentiment Scores
plt.figure(figsize=(10, 7))
sns.scatterplot(x=tfidf_2d[:, 0], y=tfidf_2d[:, 1], hue=kmeans_sentiment_labels, palette="muted", s=100, marker="o")
plt.title("KMeans Clustering with Sentiment (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("KMeans Clustering with Sentiment")


