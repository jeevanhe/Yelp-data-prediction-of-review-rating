pig -x local dataparse.pig
python review.py 
python convert_to_svm.py random_reviewfeature_out.txt random_input.txt
hdfs dfs -mkdir bigdata
hdfs dfs -copyFromLocal random_reviewfeature_out.txt bigdata
hdfs dfs -copyFromLocal random_input.txt bigdata
spark-shell -i naiveBase.scala
spark-shell -i randomForest.scala
