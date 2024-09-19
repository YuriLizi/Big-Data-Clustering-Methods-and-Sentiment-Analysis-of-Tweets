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

# Step 1: Initialize PySpark session
spark = SparkSession.builder.appName("TweetClustering").getOrCreate()

# Step 2: Load the CSV file with tweets
# Assuming you have a CSV file with a "tweets" column
tweets_df = spark.read.csv("/home/cortica/2nd_degree/big_Data/Finalproject/pythonProject/Data/tweets_cnn_clean_short.csv", header=True)
tweets_rdd = tweets_df.rdd.map(lambda row: row['tweet_clean'])

# Step 3: Preprocess the Tweets for TF-IDF
def preprocess_tweets(text):
    if text is None:
        return ""  # or handle None values differently as needed
    return text.lower()

preprocessed_rdd = tweets_rdd.map(preprocess_tweets)

# Convert RDD of strings to RDD of Rows with a named column for the DataFrame
preprocessed_rows_rdd = preprocessed_rdd.map(lambda tweet: Row(tweet_clean=tweet))

# Convert the RDD of Rows to a DataFrame
tweets_df = spark.createDataFrame(preprocessed_rows_rdd)

# Step 4: Tokenize the tweets
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

# Step 6: Method 1 - KMeans Clustering
kmeans_model = KMeans.train(tfidf_rdd, k=3, maxIterations=10, initializationMode="random")
kmeans_labels = kmeans_model.predict(tfidf_rdd).collect()  # Collect labels for plotting

# Step 7: Method 2 - DBSCAN (Density-based clustering)
from sklearn.cluster import DBSCAN

# Collect the TF-IDF features into a numpy array for DBSCAN
tfidf_array = np.array(tfidf_rdd.collect())
dbscan_model = DBSCAN(eps=0.5, min_samples=2)
dbscan_labels = dbscan_model.fit_predict(tfidf_array)

# Step 8: Plotting KMeans Clusters
# Reduce dimensions using PCA to 2D for visualization
pca = PCA(n_components=2)
tfidf_2d = pca.fit_transform(tfidf_array)

# Convert labels to numpy arrays for easier handling
kmeans_labels = np.array(kmeans_labels)

# Plot KMeans Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=tfidf_2d[:, 0], y=tfidf_2d[:, 1], hue=kmeans_labels, palette="deep", s=100, marker="o")
plt.title("KMeans Clustering of Tweets (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("KMeans Clustering of Tweets.png")

# Step 9: Plotting DBSCAN Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=tfidf_2d[:, 0], y=tfidf_2d[:, 1], hue=dbscan_labels, palette="Set1", s=100, marker="o")
plt.title("DBSCAN Clustering of Tweets (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("DBSCAN Clustering of Tweets.png")

# Step 10: Method 3 - Agglomerative Clustering (Hierarchical)
from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_model.fit_predict(tfidf_array)

# Plot Agglomerative Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=tfidf_2d[:, 0], y=tfidf_2d[:, 1], hue=agg_labels, palette="Set2", s=100, marker="o")
plt.title("Agglomerative Clustering of Tweets (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("Agglomerative Clustering of Tweets.png")
