import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, Word2Vec}

import scala.language.postfixOps
import org.apache.spark.sql.{Dataset, Row, SparkSession}

class DataPreparer{
  def getData(spark: SparkSession, inputPath: String):List[Dataset[Row]] = {
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load(inputPath).toDF("id","label","features_raw")

    val stop_words = Seq("a", "and", "for", "in", "of", "on", "the", "with", "s", "t", " ", ".", ";")


    val regexTokenizer = new RegexTokenizer()
      .setInputCol("features_raw")
      .setOutputCol("features_split")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)


    val regexTokenized = regexTokenizer.transform(df)
    regexTokenized.select("features_raw", "features_split").show(false)



    val remover = new StopWordsRemover()
      .setStopWords(stop_words.toArray)
      .setInputCol("features_split")
      .setOutputCol("features_split_stop")


    val df5 = remover.transform(regexTokenized)
    val model_for_w2v = new Word2Vec().setInputCol("features_split_stop")
      .setOutputCol("features")


    val modelW2V = model_for_w2v.fit(df5)
    val parsedData = modelW2V.transform(df5)

    parsedData.show(20, false)
    val Array(training, test) = parsedData.select("label","features").randomSplit(Array(0.8, 0.2), seed = 12345)
    val arr = List(training, test)
    arr
  }

  def useTFIDF(spark: SparkSession, inputPath: String):List[Dataset[Row]] = {
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load(inputPath).toDF("id","label","features_raw")

    val stop_words = Seq("a", "and", "for", "in", "of", "on", "the", "with", "s", "t", " ", ".", ";")

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("features_raw")
      .setOutputCol("features_split")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val regexTokenized = regexTokenizer.transform(df)

    val remover = new StopWordsRemover()
      .setStopWords(stop_words.toArray)
      .setInputCol("features_split")
      .setOutputCol("features_split_stop")

    val df5 = remover.transform(regexTokenized)

    val cv = new CountVectorizer()
      .setInputCol("features_split_stop")
      .setOutputCol("cv")

    val idf = new IDF()
      .setInputCol("cv")
      .setOutputCol("features")

    val df6 = cv.fit(df5).transform(df5)
    val parsedData = idf.fit(df6).transform(df6)

    val Array(training, test) = parsedData.select("label","features").randomSplit(Array(0.8, 0.2), seed = 12345)
    val arr = List(training, test)
    arr
  }

  def forPipleine(spark: SparkSession, inputPath: String) = {

    val stop_words = Seq("a", "and", "for", "in", "of", "on", "the", "with", "s", "t", " ", ".", ";")

    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load(inputPath).toDF("id","label","features_raw")

    val Array(training, test) = df.select("label","features_raw").randomSplit(Array(0.8, 0.2), seed = 12345)

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("features_raw")
      .setOutputCol("features_split")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)


    val remover = new StopWordsRemover()
      .setStopWords(stop_words.toArray)
      .setInputCol("features_split")
      .setOutputCol("features_split_stop")

    val cv = new CountVectorizer()
      .setInputCol("features_split_stop")
      .setOutputCol("cv")


    val idf = new IDF()
      .setInputCol("cv")
      .setOutputCol("features")

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

    val steps =  Array( regexTokenizer, remover, cv, idf,lsvc)
    val pipeline = new Pipeline().setStages(steps)
    val model = pipeline.fit(training)
    val predictions = model.transform(test)
    val a = evaluator.evaluate(predictions)
    print(a)
  }
}

//Examples of work with DataPreparer
object testml {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("ValidationFrameWork").getOrCreate()
    val inputPath = "file:///home/deadmaniq445/IdeaProjects/project/train.csv"
    val outputPath = "file:///home/deadmaniq445/IdeaProjects/project/out/"

    // If you want to use pipeline
    new DataPreparer().forPipleine(spark, inputPath)




    // If you want to use transform
    val List(training, test) = new DataPreparer().useTFIDF(spark,inputPath)

    training.show(false)



    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    val lsvcModel = lsvc.fit(training)

    val predictions = lsvcModel.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

    val a = evaluator.evaluate(predictions)
    print(a)
    print("Privet kak dela!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  }
}