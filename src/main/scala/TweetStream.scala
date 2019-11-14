import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.{DataStreamWriter, StreamingQuery}

object TweetStream{
  var fileWriter: StreamingQuery = _
  def start(spark: SparkConf, sparkSession: SparkSession): Unit ={
    val classifier = new Classifier()
    if (!scala.reflect.io.File("./models/model").exists) {
      classifier.fit(sparkSession)
    }

    val stream = sparkSession.readStream.format("socket")
      .option("host","10.90.138.32").option("port", 8989)
      .load().toDF("features_raw")

    val writer = classifier.predict(stream).select("features_raw", "prediction").writeStream
    writer.format("console").option("truncate", "false").start()
    this.fileWriter = writer.format("csv").option("checkpointLocation", "./chkpnt")
      .option("path", "./output").start()
    sparkSession.streams.awaitAnyTermination()
  }
}