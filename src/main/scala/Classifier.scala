import org.apache.hadoop.mapred.TaskCompletionEvent.Status
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, Word2Vec}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.lit
import org.apache.spark.streaming.dstream.ReceiverInputDStream

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

  val model_for_w2v: Word2Vec = new Word2Vec().setInputCol("features_split_stop")
    .setOutputCol("features")

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

  val lr : LogisticRegression = new LogisticRegression()
    .setMaxIter(100)
    .setRegParam(0.02)
    .setElasticNetParam(0.3)

  val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  val steps1 =  Array( regexTokenizer, remover, cv, idf, lsvc)
  val steps2 =  Array( regexTokenizer, remover, cv, idf, lr)
  val steps3 =  Array( regexTokenizer, remover, model_for_w2v, lr)
  val steps4 =  Array( regexTokenizer, remover, model_for_w2v, lsvc)
  val pipeline1: Pipeline = new Pipeline().setStages(steps1)
  val pipeline2: Pipeline = new Pipeline().setStages(steps2)
  val pipeline3: Pipeline = new Pipeline().setStages(steps3)
  val pipeline4: Pipeline = new Pipeline().setStages(steps4)
  var model1: PipelineModel = _
  var model2: PipelineModel = _
  var model3: PipelineModel = _
  var model4: PipelineModel = _

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

    this.model1 = pipeline1.fit(df)
    this.model1.write.overwrite().save("./models/model1")

    this.model2 = pipeline2.fit(df)
    this.model2.write.overwrite().save("./models/model2")

    this.model3 = pipeline3.fit(df)
    this.model3.write.overwrite().save("./models/model3")

    this.model4 = pipeline4.fit(df)
    this.model4.write.overwrite().save("./models/model4")
  }

  /**
   * method for training our model and evaluationg it (by training on 90% of data and testing on 10%)
   * @param spark - SparkSession, look at the testml.main for example of it
   */
  def fit_and_evaluate(spark: SparkSession): Unit ={
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load("./csv/train.csv").toDF("id","label","features_raw")
    val Array(training, test) = df.select("label","features_raw").randomSplit(Array(0.9, 0.1), seed = 12345)

    this.model1 = pipeline1.fit(training)
    val predictions1 = model1.transform(test)
    println("TF-IDF + LinearSVC")
    evaluation(predictions1)
    this.model1.write.overwrite().save("./models/model1")

    this.model2 = pipeline2.fit(training)
    val predictions2 = model2.transform(test)
    println("TF-IDF + LogisticRegression")
    evaluation(predictions2)
    this.model2.write.overwrite().save("./models/model2")

    this.model3 = pipeline3.fit(training)
    val predictions3 = model3.transform(test)
    println("Word2Vec + LinearSVC")
    evaluation(predictions3)
    this.model3.write.overwrite().save("./models/model3")

    this.model4 = pipeline4.fit(training)
    val predictions4 = model4.transform(test)
    println("Word2Vec + LogisticRegression")
    evaluation(predictions4)
    this.model4.write.overwrite().save("./models/model4")
  }

  def evaluation(predictions : DataFrame): Unit ={
    val areaUnderROC = evaluator.evaluate(predictions)
    val lp : DataFrame = predictions.select("prediction", "label")
    val counttotal = predictions.count().toDouble
    val correct = lp.filter("label == prediction").count().toDouble
    val wrong = lp.filter("label != prediction").count().toDouble
    val ratioWrong = wrong / counttotal
    val accuracy = correct / counttotal
    val truen = lp.filter("label == 0.0").filter("label == prediction").count() / counttotal
    val truep = lp.filter("label == 1.0").filter("label == prediction").count() / counttotal
    val falsen = lp.filter("label == 0.0").filter("label != prediction").count() / counttotal
    val falsep = lp.filter("label == 1.0").filter("label != prediction").count() / counttotal
    val precision = truep / (truep + falsep)
    val recall = truep / (truep + falsen)
    val f1_score = (2 * precision * recall) / (precision + recall)

    println("AreaUnderROC: " + areaUnderROC)
    println("Accuracy:   ", accuracy)
    println("Presicion:  ", precision)
    println("Recall:     ", recall)
    println("F1-score:   ", f1_score)
  }

  /**
   * Accepts a dataset of tweets and classifies it
   * @param dataset - dataset with tweets
   * @return dataset with tweets and predictions
   */
  def predict(dataset: Dataset[Row], number: String): Dataset[Row] ={
    import org.apache.spark.ml._
    val dfWithFoobar = dataset.withColumn("label", lit(null: String))
    number match {
      case "1" =>
        if (this.model1 == null)
          this.model1 = PipelineModel.load("./models/model1")
        this.model1.transform(dfWithFoobar)
      case "2" =>
        if (this.model2 == null)
          this.model2 = PipelineModel.load("./models/model2")
        this.model2.transform(dfWithFoobar)
      case "3" =>
        if (this.model3 == null)
          this.model3 = PipelineModel.load("./models/model3")
        this.model3.transform(dfWithFoobar)
      case "4" =>
        if (this.model4 == null)
          this.model4 = PipelineModel.load("./models/model4")
        this.model4.transform(dfWithFoobar)
    }
  }

}