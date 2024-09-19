from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("TweetStatistics").getOrCreate()

# Step 2: Read CSV file with tweets
# Assuming the CSV has a column named 'tweet_text'
csv_path = "Data/tweets_cnn.csv"
tweets_df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Step 3: Extract the tweets column as a list and create an RDD
tweets = tweets_df.select("tweet").rdd.flatMap(lambda x: x)

# Alternatively, use spark.sparkContext.parallelize if you already have the list in memory
# tweets_rdd = spark.sparkContext.parallelize(tweets_list)

# Step 4: Calculate tweet lengths
tweet_lengths_rdd = tweets.map(lambda tweet: len(tweet))

# Step 5: Calculate statistics: mean, standard deviation, average
total_tweets = tweet_lengths_rdd.count()
sum_lengths = tweet_lengths_rdd.sum()
mean_length = sum_lengths / total_tweets
std_dev = tweet_lengths_rdd.stdev()
max_length = tweet_lengths_rdd.max()
min_length = tweet_lengths_rdd.min()

# Step 6: Print out the statistics
print(f"Total Tweets: {total_tweets}")
print(f"Mean Tweet Length: {mean_length:.2f}")
print(f"Standard Deviation of Tweet Lengths: {std_dev:.2f}")
print(f"Max Tweet Length: {max_length}")
print(f"Min Tweet Length: {min_length}")

# Step 7: Plot the results using matplotlib
tweet_lengths = tweet_lengths_rdd.collect()

# Plotting
plt.figure(figsize=(10, 6))

# Histogram of tweet lengths
plt.hist(tweet_lengths, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.title("Distribution of Tweet Lengths")
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")
plt.grid(True)

# Save the plot
plt.savefig("tweet_lengths_distribution.png")
plt.show()

# Step 8: Stop the Spark session
spark.stop()
