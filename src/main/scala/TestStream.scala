import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf

import scala.language.postfixOps
import org.apache.spark.sql.SparkSession


object TestStream {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("owlqn").setLevel(Level.OFF)

    print("To stop working with stream press Ctrl+C\n")
    print("Run on yarn or local? (1-yarn, 2-local)\n")
    val flag = scala.io.StdIn.readLine()
    var sprconf:SparkConf = null
    if (flag == "1") {
      sprconf = new SparkConf().set("spark.driver.allowMultipleContexts", "true").setMaster("yarn").setAppName("TwitterStreamSentiment")
    }
    else {
      sprconf = new SparkConf().set("spark.driver.allowMultipleContexts", "true").setMaster("local[*]").setAppName("TwitterStreamSentiment")
    }
    val spark = SparkSession.builder().config(sprconf).getOrCreate()

    TweetStream.start(sprconf, spark, flag)
  }
}