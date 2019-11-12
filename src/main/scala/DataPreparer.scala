import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, Word2Vec}
import org.apache.spark.sql.functions.lit

import scala.language.postfixOps
import org.apache.spark.sql.{Dataset, Row, SparkSession}


/** This class uses Word2Vec to prepare data. You can change lsvc (linear SVC) to another model.*/
class ClassifierW2V{
  /**
   * These are the words that will be thrown away from our vectorization
   */
  val stop_words: Seq[String] = Seq("a", "and", "for", "in", "of", "on", "the", "with", "s", "t", " ", ".", ";")

  val regexTokenizer: RegexTokenizer = new RegexTokenizer()
    .setInputCol("features_raw")
    .setOutputCol("features_split")
    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)


  val remover: StopWordsRemover = new StopWordsRemover()
    .setStopWords(stop_words.toArray)
    .setInputCol("features_split")
    .setOutputCol("features_split_stop")


  val model_for_w2v: Word2Vec = new Word2Vec().setInputCol("features_split_stop")
    .setOutputCol("features")

  /**
   * This is the machine learning model. You can change it to another one,
   * just don't forget to change it in the pipeline
   */
  val lsvc: LinearSVC = new LinearSVC()
    .setMaxIter(10)
    .setRegParam(0.1)

  val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  val steps =  Array( regexTokenizer, remover, model_for_w2v, lsvc)
  val pipeline: Pipeline = new Pipeline().setStages(steps)
  var model: PipelineModel = null

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
    this.model.write.overwrite().save("./models/modelW2V")
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
    val model = pipeline.fit(training)
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
    if (this.model.equals(null))
      this.model = PipelineModel.load("./models/modelW2V")
    this.model.transform(dataset)
  }

}

/** This class uses TF/IDF to prepare data. You can change lsvc to another model.*/
class Classifier{

  /**
   * These are the words that will be thrown away from our vectorization
   */
  val stop_words: Seq[String] = Seq("a", "and", "for", "in", "of", "on", "the", "with", "s", "t", " ", ".", ";")

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
    .setLabelCol("indexedLabel")
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  val steps =  Array( regexTokenizer, remover, cv, idf, lsvc)
  val pipeline: Pipeline = new Pipeline().setStages(steps)
  var model: PipelineModel = null

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
  def fit_and_evaluate(spark: SparkSession) ={
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load("./csv/train.csv").toDF("id","label","features_raw")
    val Array(training, test) = df.select("label","features_raw").randomSplit(Array(0.8, 0.2), seed = 12345)
    val model = pipeline.fit(training)
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
    if (this.model.equals(null))
      this.model = PipelineModel.load("./models/model")
    this.model.transform(dfWithFoobar)
  }
}

//Examples of work with DataPreparer
object testml {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("Classifier").getOrCreate()

    val testPath = "./csv/test.csv"

    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load(testPath).toDF("id","features_raw")

    val classifier = new Classifier()
    print(classifier.fit_and_evaluate(spark))
    classifier.fit(spark)
    val df_classed = classifier.predict(df)
    df_classed.show(20, false)

  }
}