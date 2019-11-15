import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object WordCount {
  def count(df: DataFrame):DataFrame ={

    val wordsDF = df.explode("features_raw","word")((line: String) => line
      .toLowerCase().split(" "))
    wordsDF.groupBy("word").count().sort(desc("count"))
  }
}
