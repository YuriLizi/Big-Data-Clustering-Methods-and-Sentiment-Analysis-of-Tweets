import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, avg
from pyspark.sql.types import DoubleType
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize Spark
spark = SparkSession.builder.appName("SentimentAnalysisRDD").getOrCreate()

# Load the CSV into a Spark DataFrame
file_path = "Data/tweets_cnn_clean.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get the sentiment score, with null handling
def get_sentiment_score(tweet):
    if tweet is None:  # Check for None (missing tweets)
        return 0.0     # Return neutral sentiment score for missing tweets
    sentiment = sid.polarity_scores(tweet)
    return sentiment['compound']

# Register the function as a UDF (User Defined Function)
get_sentiment_score_udf = udf(get_sentiment_score, DoubleType())

# Apply sentiment analysis UDF to the 'tweet' column
df = df.withColumn("sentiment_score", get_sentiment_score_udf(col("tweet_clean")))

# Convert the 'date' column to date type
df = df.withColumn("date", col("date").cast("date"))

# Group by date and calculate the average sentiment score per day
average_sentiment_df = df.groupBy("date").agg(avg("sentiment_score").alias("average_sentiment"))

# Convert to Pandas DataFrame for plotting
average_sentiment_pandas_df = average_sentiment_df.orderBy("date").toPandas()

# Apply a rolling mean to smooth the graph (adjust window size as needed)
rolling_window_size = 50
average_sentiment_pandas_df['smoothed_sentiment'] = average_sentiment_pandas_df['average_sentiment'].rolling(window=rolling_window_size).mean()

# Plot the smoothed average sentiment scores per day
plt.figure(figsize=(10, 6))
plt.plot(average_sentiment_pandas_df['date'], average_sentiment_pandas_df['smoothed_sentiment'])
plt.title("CNN Smoothed Average Sentiment Score of Tweets Per Day")
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=90)
plt.savefig("CNN_Smoothed_Average_Sentiment_Score_of_Tweets_Per_Day.png")
