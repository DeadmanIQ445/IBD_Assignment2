import scala.reflect.io.File
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext, SparkSession}
import org.apache.spark.sql.streaming.{DataStreamWriter, StreamingQuery}
import org.apache.spark.sql.functions.{current_timestamp, lit}


object TweetStream{
  var fileWriter: StreamingQuery = _
  def start(spark: SparkConf, sparkSession: SparkSession, flag: String): Unit ={
    val classifier = new Classifier()
    var modelsPath: String = null
    var chkpntPath: String = null
    var outputPath: String = null
    var trainPath: String = null

    if (flag == "1"){
      modelsPath = "hdfs://namenode:9000/user/jakarta/models"
      chkpntPath = "hdfs://namenode:9000/user/jakarta/chkpnt"
      outputPath = "hdfs://namenode:9000/user/jakarta/output"
      trainPath =  "hdfs://namenode:9000/twitter/twitter_sentiment_data.csv"
    }
    else {
      modelsPath  = "./models"
      chkpntPath  = "./chkpnt"
      outputPath = "./output"
      trainPath = "./csv/train.csv"
    }

    val outputDir = File(outputPath)
    if (outputDir.exists)
      outputDir.deleteRecursively()
    val ckpntDir = File(chkpntPath)
    if (ckpntDir.exists)
      ckpntDir.deleteRecursively()
    chkpntPath = chkpntPath.concat("/")
    outputPath = outputPath.concat("/")

    if (!scala.reflect.io.File(modelsPath.concat("/model1")).exists) {
      classifier.fit_and_evaluate(sparkSession, trainPath, modelsPath)
    }

    val stream = sparkSession.readStream.format("socket")
    .option("host","10.90.138.32").option("port", 8989)
    .load().toDF("features_raw")

    for(i <- 1 to 4){
      var writer : DataStreamWriter[Row] = classifier.predict(stream, i.toString, modelsPath)
        .withColumn("time_stamp", lit(current_timestamp()))
        .select("time_stamp", "features_raw", "prediction")
        .writeStream

//      writer.format("console").option("truncate", "false").start()

      this.fileWriter = writer.format("csv")
      .option("checkpointLocation", chkpntPath.concat(i.toString))
      .option("path", outputPath.concat(i.toString)).start()
    }

    val wc = WordCount.count(stream).writeStream
      .format("memory")
      .queryName("WordCount")
      .outputMode("complete")
      .start()
    while(wc.isActive){
      Thread.sleep(30000)
      sparkSession.sql("select * from WordCount").coalesce(1).write.mode("overwrite").csv(outputPath.concat("WS"))
    }
    sparkSession.streams.awaitAnyTermination()
  }
}