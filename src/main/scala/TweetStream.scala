import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import org.apache.spark.sql.streaming.{DataStreamWriter, StreamingQuery}
import org.apache.spark.sql.functions.{current_timestamp, lit}

object TweetStream{
  var fileWriter: StreamingQuery = _
  def start(spark: SparkConf, sparkSession: SparkSession): Unit ={
    val classifier = new Classifier()
    if (!scala.reflect.io.File("./models/model1").exists) {
      classifier.fit_and_evaluate(sparkSession)
    }

    val stream = sparkSession.readStream.format("socket")
    .option("host","10.90.138.32").option("port", 8989)
    .load().toDF("features_raw")

    for(i <- 1 to 4){
      var writer : DataStreamWriter[Row] = classifier.predict(stream, i.toString)
        .select("features_raw", "prediction")
        .withColumn("time_stamp", lit(current_timestamp()))
        .select("time_stamp", "features_raw", "prediction")
        .writeStream

      writer.format("console").option("truncate", "false").start()

      this.fileWriter = writer.format("csv")
      .option("checkpointLocation", "./chkpnt/".concat(i.toString))
      .option("path", "./output/".concat(i.toString)).start()
    }



    sparkSession.streams.awaitAnyTermination()
  }
}