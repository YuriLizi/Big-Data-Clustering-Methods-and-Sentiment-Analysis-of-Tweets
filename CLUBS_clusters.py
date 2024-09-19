from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Initialize PySpark session
spark = SparkSession.builder.appName("TweetClusteringWithAffinityPropagationAndGMM").getOrCreate()

# Step 2: Load the CSV file with tweets
tweets_df = spark.read.csv("Data/tweets_cnn_clean_short.csv", header=True)

# Step 3: Preprocess Tweets for TF-IDF
def preprocess_tweets(text):
    if text:
        return text.lower()
    return ""

tweets_rdd = tweets_df.select("tweet_clean").rdd.map(lambda row: preprocess_tweets(row["tweet_clean"])).filter(lambda x: x != "")

# Step 4: Create DataFrame with proper schema
tweets_rows = tweets_rdd.map(lambda tweet: Row(tweet=tweet))
tweets_preprocessed_df = spark.createDataFrame(tweets_rows)

# Step 5: Convert Tweets to TF-IDF Features
tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
tokenized_df = tokenizer.transform(tweets_preprocessed_df)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)
featurized_df = hashingTF.transform(tokenized_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# Convert the TF-IDF DataFrame to RDD for clustering
tfidf_rdd = tfidf_df.select("features").rdd.map(lambda row: row.features.toArray())

# Step 6: Approximate CLUBS+ Clustering with Affinity Propagation + GMM

# Collect the data from RDD into a NumPy array for clustering
tfidf_array = np.array(tfidf_rdd.collect())

# Step 6.1: First phase - Affinity Propagation for initial clustering
affinity_model = AffinityPropagation(random_state=42)
affinity_labels = affinity_model.fit_predict(tfidf_array)

# Step 6.2: Second phase - Gaussian Mixture Model (GMM) for fine-grained clustering
gmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_labels = gmm_model.fit_predict(tfidf_array)

# Combine Affinity Propagation and GMM results
# For simplicity, combine by assigning GMM labels to affinity clusters for a refined structure
combined_labels = np.where(gmm_labels != -1, gmm_labels, affinity_labels)

# Step 7: Plot the clusters (using PCA for dimensionality reduction)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_array)  # Reduce to 2 dimensions

# Step 8: Plot using Matplotlib
plt.figure(figsize=(10, 6))
for i in np.unique(combined_labels):
    points = reduced_data[combined_labels == i]
    plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}")

plt.title("Approximated CLUBS+ Clustering of Tweets (Affinity Propagation + GMM)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend()
plt.savefig("Approximated CLUBS+ Clustering of Tweets.png")
