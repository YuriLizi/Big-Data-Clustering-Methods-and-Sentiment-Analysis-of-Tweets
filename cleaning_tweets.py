from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
import matplotlib.pyplot as plt
import re
import unicodedata
import contractions
from pyspark.sql.types import StringType

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("TweetStatistics").getOrCreate()

# Step 2: Read CSV file with tweets
csv_path = "Data/tweets_cnn.csv"
tweets_df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Step 3: Define the tweet cleaning functions
def remove_URL(text):
    return re.sub(r"http\S+", "", text)

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_non_ascii(text):
    return ''.join([c for c in text if c in set(unicodedata.normalize('NFKD', c))])

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

def remove_punct(text):
    return re.sub(r'[^\w\s]', '', text)

# Step 4: UDF to clean tweets
def clean_tweet(text):
    text = text.lower()
    text = contractions.fix(text)
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = remove_punct(text)
    return text

# Register the function as a UDF in PySpark
clean_tweet_udf = udf(lambda tweet: clean_tweet(tweet), StringType())

# Step 5: Apply cleaning function to the tweet column and create a new column 'tweet_clean'
tweets_clean_df = tweets_df.withColumn("tweet_clean", clean_tweet_udf(col("tweet")))

# Step 6: Extract cleaned tweets column as an RDD
tweets_rdd = tweets_clean_df.select("tweet_clean").rdd.flatMap(lambda x: x)

# Step 7: Calculate tweet lengths
tweet_lengths_rdd = tweets_rdd.map(lambda tweet: len(tweet))

# Step 8: Calculate statistics: mean, standard deviation, average
total_tweets = tweet_lengths_rdd.count()
sum_lengths = tweet_lengths_rdd.sum()
mean_length = sum_lengths / total_tweets
std_dev = tweet_lengths_rdd.stdev()
max_length = tweet_lengths_rdd.max()
min_length = tweet_lengths_rdd.min()

# Step 9: Print out the statistics
print(f"Total Tweets: {total_tweets}")
print(f"Mean Tweet Length: {mean_length:.2f}")
print(f"Standard Deviation of Tweet Lengths: {std_dev:.2f}")
print(f"Max Tweet Length: {max_length}")
print(f"Min Tweet Length: {min_length}")

# Step 10: Plot the results using matplotlib
tweet_lengths = tweet_lengths_rdd.collect()

# Plotting
plt.figure(figsize=(10, 6))

# Histogram of tweet lengths
plt.hist(tweet_lengths, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.title("Distribution of Tweet Lengths (Cleaned Tweets)")
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")
plt.grid(True)

# Save the plot
plt.savefig("tweet_lengths_distribution_cleaned.png")
plt.show()

# Step 11: Stop the Spark session
spark.stop()
