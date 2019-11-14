import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.lit
import org.apache.spark.streaming.dstream.ReceiverInputDStream
import twitter4j.Status

/** This class uses TF/IDF to prepare data. You can change lsvc to another model.*/
class Classifier{

  /**
   * These are the words that will be thrown away from our vectorization
   */
  val stop_words: Seq[String] = Seq("quot", "a", "and", "for", "in", "of", "on", "the", "with", "s", "t", " ", ".", ";")

  val regexTokenizer: RegexTokenizer = new RegexTokenizer()
    .setInputCol("features_raw")
    .setOutputCol("features_split")
    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

  val remover: StopWordsRemover = new StopWordsRemover()
    .setStopWords(stop_words.toArray)
    .setInputCol("features_split")
    .setOutputCol("features_split_stop")

  val cv: CountVectorizer = new CountVectorizer()
    .setInputCol("features_split_stop")
    .setOutputCol("cv")


  val idf: IDF = new IDF()
    .setInputCol("cv")
    .setOutputCol("features")

  /**
   * This is the machine learning model. You can change it to another one,
   * just don't forget to change it in the pipeline
   */
  val lsvc: LinearSVC = new LinearSVC()
    .setMaxIter(10)
    .setRegParam(0.1)

  val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  val steps =  Array( regexTokenizer, remover, cv, idf, lsvc)
  val pipeline: Pipeline = new Pipeline().setStages(steps)
  var model: PipelineModel = _

  /**
   * method for training our model
   * @param spark - SparkSession, look at the testml.main for example of it
   */
  def fit(spark: SparkSession): Unit ={
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load("./csv/train.csv").toDF("id","label","features_raw")
    this.model = pipeline.fit(df)
    this.model.write.overwrite().save("./models/model")
  }

  /**
   * method for training our model and evaluationg it (by training on 80% of data and testing on 20&)
   * @param spark - SparkSession, look at the testml.main for example of it
   */
  def fit_and_evaluate(spark: SparkSession): Double ={
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load("./csv/train.csv").toDF("id","label","features_raw")
    val Array(training, test) = df.select("label","features_raw").randomSplit(Array(0.8, 0.2), seed = 12345)
    this.model = pipeline.fit(training)
    val predictions = model.transform(test)
    evaluator.evaluate(predictions)
  }

  /**
   * Accepts a dataset of tweets and classifies it
   * @param dataset - dataset with tweets
   * @return dataset with tweets and predictions
   */
  def predict(dataset: Dataset[Row]): Dataset[Row] ={
    import org.apache.spark.ml._
    val dfWithFoobar = dataset.withColumn("label", lit(null: String))
    if (this.model == null)
      this.model = PipelineModel.load("./models/model")
    this.model.transform(dfWithFoobar)
  }

  def predict(dataset: ReceiverInputDStream[Status], sparkSession: SparkSession): Unit ={
    import org.apache.spark.ml._
    if (this.model == null)
      this.model = PipelineModel.load("./models/model")
    val df = dataset.map(status => status.getText)
    df.foreachRDD { rdd =>

      // Get the singleton instance of SparkSession
      import sparkSession.implicits._

      // Convert RDD[String] to DataFrame
      val wordsDataFrame = rdd.toDF("features_raw")
      wordsDataFrame.show(false)
      wordsDataFrame.withColumn("label", lit(null: String))
      val predict = this.model.transform(wordsDataFrame)
      predict.select("features_raw", "prediction").show()
    }
  }
}