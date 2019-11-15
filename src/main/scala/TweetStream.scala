import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.{DataStreamWriter, StreamingQuery}

object TweetStream{
  var fileWriter: StreamingQuery = _
  def start(spark: SparkConf, sparkSession: SparkSession): Unit ={
    val classifier = new Classifier()
    if (!scala.reflect.io.File("./models/model").exists) {
      classifier.fit_and_evaluate(sparkSession)
    }

    val stream = sparkSession.readStream.format("socket")
      .option("host","10.90.138.32").option("port", 8989)
      .load().toDF("features_raw")

    val writer1 = classifier.predict(stream, "1").select("features_raw", "prediction").writeStream
    val writer2 = classifier.predict(stream, "2").select("features_raw", "prediction").writeStream
    val writer3 = classifier.predict(stream, "3").select("features_raw", "prediction").writeStream
    val writer4 = classifier.predict(stream, "4").select("features_raw", "prediction").writeStream

    writer1.format("console").option("truncate", "false").start()

    this.fileWriter = writer1.format("csv").option("checkpointLocation", "./chkpnt/1")
      .option("path", "./output/1").start()
    this.fileWriter = writer2.format("csv").option("checkpointLocation", "./chkpnt/2")
      .option("path", "./output/2").start()
    this.fileWriter = writer3.format("csv").option("checkpointLocation", "./chkpnt/3")
      .option("path", "./output/3").start()
    this.fileWriter = writer4.format("csv").option("checkpointLocation", "./chkpnt/4")
      .option("path", "./output/4").start()

    sparkSession.streams.awaitAnyTermination()
  }
}