import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf

import scala.language.postfixOps
import org.apache.spark.sql.SparkSession


object TestStream {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    print("To stop working with stream press Ctrl+C")

    val sprconf = new SparkConf().set("spark.driver.allowMultipleContexts", "true").setMaster("local[*]").setAppName("TwitterStreamSentiment")
    val spark = SparkSession.builder().config(sprconf).getOrCreate()

    TweetStream.start(sprconf, spark)
  }
}