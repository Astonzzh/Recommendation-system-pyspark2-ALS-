from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline


def setting():
    mlflow.set_tracking_uri('http://ip_address:port')
    mlflow.set_experiment('new')


def readin(name):
    # read file in hdfs (hdfs dfs -ls)
    df = spark.read.csv(sep='\t', path=name, inferSchema='true')
    return df


def preprocess(df):
    # preprocess the training dataset
    result = df.drop('_c3')
    result = result.withColumnRenamed('_c0', 'user')
    result = result.withColumnRenamed('_c1', 'item')
    result = result.withColumnRenamed('_c2', 'rating')
    return result


def train_model(df):
    als = ALS(rank=10, maxIter=10, regParam=0.1, userCol="user", itemCol="item", ratingCol="rating")
    pipeline = Pipeline(stages=[als])
    model = pipeline.fit(df)
    return model


def submit(model):
    mlflow.spark.log_model(model, 'ALS_model_1')
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


def get_score(model):
    test_result = readin('u1.test')
    test_result = test_result.select('_c2').collect()
    test_result = [i[0] for i in test_result]
    test = readin('u1.test')
    test = test.drop('_c3', '_c2')
    test = test.withColumnRenamed('_c0', 'user')
    test = test.withColumnRenamed('_c1', 'item')
    predictions = sorted(model.transform(test).collect(), key=lambda r: r[0])
    predictions = [i[2] for i in predictions]

    # prediction values may contains nan, do not take these values into account when calculating score
    i = 0
    while i < len(predictions):
        if np.isnan(predictions[i]):
            predictions.pop(i)
            test_result.pop(i)
        else:
            i = i + 1
    
    def square_error(a,b):
        result = 0
        for i in range(len(a)):
            result = result + (a[i]-b[i]) * (a[i]-b[i])
        return result
    
    return square_error(test_result, predictions)


def main_log():
    setting()
    train = readin('u1.base')
    train_preprocessed = preprocess(df)
    model = train_model(train_preprocessed)
    submit(model)


def main_score():
    setting()
    train = readin('u1.base')
    train_preprocessed = preprocess(df)
    model = train_model(train_preprocessed)
    print(getscore(model))
    
    
def main():
    setting()
    train = readin('u1.base')
    train_preprocessed = preprocess(train)
    model = train_model(train_preprocessed)
    score = get_score(model)
    mlflow.log_metric("score", score)
    mlflow.spark.log_model(model, "ALS_model_1")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
    print(score)


main()

