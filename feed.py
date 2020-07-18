import requests as rq
import pandas as pd


# spark dataframe
test_spark = spark.read.csv(sep='\t', path='u1.test', inferSchema='true')
test_spark = test_spark.drop('_c3', '_c2')
test_spark = test_spark.withColumnRenamed('_c0', 'user')
test_spark = test_spark.withColumnRenamed('_c1', 'item')


# pandas dataframe
test = pd.read_csv('ALS/u1.test', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'], index_col=None)
test = test.drop(['rating', 'timestamp'], axis=1)
test = test.to_json(orient='split', index=False)


url = "http://localhost:<a specific port for this model>/invocations"
headers={'Content-Type': 'application/json; format=pandas-split'}
rp = rq.post(url=url, data=test, headers=headers)


print(rp.text)

