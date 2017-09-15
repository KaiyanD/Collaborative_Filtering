## Read data from customer invoice table
import numpy as np
from pyspark.sql import HiveContext
from pyspark import SparkContext
sc =SparkContext.getOrCreate()
hc = HiveContext(sc)
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.sql import functions as F

CF_Tab1 = hc.sql("select cust_nbr, material_id, sales_qty_base from ds_db.customer_invoice_Retest")
# filter out if sum of sales is less than 0, change the number to 100 if it is larger than 100 for testing.
CF_Tab2 = CF_Tab1.groupby("cust_nbr","material_id").agg(F.sum("sales_qty_base").alias("sales_qty_base"))
CF_Tab3 = CF_Tab2.filter(CF_Tab2.sales_qty_base>=0)
CF_Tab = CF_Tab3.withColumn("sales_qty_base",F.when(CF_Tab3.sales_qty_base>100,100.0).otherwise(CF_Tab3.sales_qty_base))

## We are replacing cust_nbr with user_index and material_id with material_id because cust_nbr is too long as an integer and material_id can not contain string for feeding the ALS algorithm.
user_distinct = CF_Tab.select("cust_nbr").distinct()
from pyspark.sql.functions import monotonically_increasing_id
# This will return a new DF with all the columns + index
res = user_distinct.coalesce(1).withColumn("user_index", monotonically_increasing_id()).withColumn("cust_nbr_distinct",user_distinct.cust_nbr)

material_distinct = CF_Tab.select("material_id").distinct()
from pyspark.sql.functions import monotonically_increasing_id
# This will return a new DF with all the columns + id
res_mat = material_distinct.coalesce(1).withColumn("material_index", monotonically_increasing_id()).withColumn("material_id_distinct",material_distinct.material_id)

table_all_1= CF_Tab.join(res, CF_Tab.cust_nbr == res.cust_nbr_distinct)
table_all_2 = table_all_1.join(res_mat,table_all_1.material_id==res_mat.material_id)
table_all_3 = table_all_2.select(["user_index","material_index","sales_qty_base"])

## Ramdomly selected 60% of the data as Train and 20% of the data as Test set.
cf_tabRDD = table_all_3.rdd.map(lambda l: Rating(int(l[0]),int(l[1]), float(l[2])))
Train_RDD, Val_RDD, Test_RDD = cf_tabRDD.randomSplit([6, 2, 2],seed=0)

## Fit the ALS model on Train set.
rank = 10
numIterations = 10
model = ALS.trainImplicit(Train_RDD, rank, numIterations, alpha=0.01,nonnegative=True)
## Use the model to predict on Test set and calculate the MSE. Save the prediction as results_table.
testdata = Train_RDD.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = Train_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))
results_table = ratesAndPreds.map(lambda p:(p[0][0],p[0][1],p[1][0],p[1][1])).toDF(["cust_nbr","material_id","sales_qty_base","prediction"])

# Save the result as hive table.
res.createOrReplaceTempView("tmp")
sqlContext.sql("drop table IF EXISTS ds_db.tmp")
sqlContext.sql("create table ds_db.tmp as select * from tmp")

## Use the model to predict for specific user_index
mat_ls1 = results_table.select("material_id").rdd.distinct().collect()
mat_ls2 = 9*mat_ls1
user_ls = []
l = [1615,1704,1556,812,1575,1853,69,1731,1157]
for x in l:
    user_ls += [x]*2513
    
test_table = sc.parallelize(zip(user_ls,mat_ls2)).map(lambda x:(x[0],x[1][0]))
prediction_table = model.predictAll(test_table).toDF()

## Get the recommendation for each user based on the prediction_table.
import pandas as pd
rec_table = prediction_table.filter(prediction_table.user==l[0]).sort(prediction_table.rating.desc()).select("product").limit(15).toDF("1").toPandas()
for x in l[1:]:
    temp_pd = prediction_table.filter(prediction_table.user==x).sort(prediction_table.rating.desc()).select("product").limit(15).toDF(str(x)).toPandas()
    rec_table = pd.concat([rec_table, temp_pd], axis=1, ignore_index=True)
# Save the recommendation as hive table
spark.createDataFrame(rec_table).createOrReplaceTempView("tmp")
sqlContext.sql("drop table IF EXISTS ds_db.ALS_resultsfor20")
sqlContext.sql("create table ds_db.ALS_resultsfor20 as select * from tmp")
