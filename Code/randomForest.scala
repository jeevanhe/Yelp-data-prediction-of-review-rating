import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils


val data = MLUtils.loadLibSVMFile(sc,"bigdata/random_input.txt")

val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

val numClasses = 6
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 60
val featureSubsetStrategy = "auto"
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)


val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val accuracy = (1.0* labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count())*100
exit
