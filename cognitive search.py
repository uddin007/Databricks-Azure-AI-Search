# Databricks notebook source
# MAGIC %pip install yake

# COMMAND ----------

import json
from datetime import datetime
import logging
import traceback
import datetime
import pyodbc
import scipy.stats
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import re
import yake
import base64
import requests
import uuid
import pandas as pd

# COMMAND ----------

text = "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning "\
"competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud "\
"Next conference in San Francisco this week, the official announcement could come as early as tomorrow. "\
"Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. "\
"Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, "\
"was founded by Goldbloom  and Ben Hamner in 2010. "\
"The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, "\
"it has managed to stay well ahead of them by focusing on its specific niche. "\
"The service is basically the de facto home for running data science and machine learning competitions. "\
"With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, "\
"it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow "\
"and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, "\
"Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. "\
"That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google "\
"will keep the service running - likely under its current name. While the acquisition is probably more about "\
"Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition "\
"and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can "\
"share this code on the platform (the company previously called them 'scripts'). "\
"Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with "\
"that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) "\
"since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, "\
"Google chief economist Hal Varian, Khosla Ventures and Yuri Milner "

# COMMAND ----------

pd.set_option('display.max_colwidth', None)

# COMMAND ----------

name = [
    "Energy-Efficient Tuning of Spintronic Neurons to Imitate the Non-linear Oscillatory Neural Networks of the Human Brain",
    "Astronomers Detect Electromagnetic Signal Caused by Unequal Neutron-Star Collision"
]

value = [
    "('Non-linear Oscillatory Neural', 0.0006487982591412822), ('Oscillatory Neural Networks', 0.0006487982591412822), ('Human Brain', 0.004925103769790319), ('Tuning of Spintronic', 0.007444681613352736)",
    "('Detect Electromagnetic Signal', 0.0032173869679631944), ('Electromagnetic Signal Caused', 0.0032173869679631944), ('Astronomers Detect Electromagnetic', 0.0035308295728302113), ('Unequal Neutron-Star Collision', 0.003995142956050651)"
]

dict1 = {'news_title': name, 'keywords': value} 
 
# Calling DataFrame constructor on list
pdf1 = pd.DataFrame(dict1)
pdf1.head()

# COMMAND ----------

language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

text = "Astronomers Detect Electromagnetic Signal Caused by Unequal Neutron-Star Collision"

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)

for kw in keywords:
    print(kw)

# COMMAND ----------

language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)

for kw in keywords:
    print(kw)

# COMMAND ----------

df = spark.table('accenture.example.heartrate_device')
df.show()

# COMMAND ----------

def each_partition_func(partition, col_name,key_word_to_search):
    key_word_to_search = "my_table"
    Regex = '^\d'


    content_list = []
    yake_str = ""
    for row in partition:
        if row[col_name] != "":
                #content_list += [row[col_name]]
                content_list.append(row[col_name])
                if not re.search(Regex,row[col_name]):
                    if not (row[col_name].isdigit() or row[col_name][0] == '-' and row[col_name][1:].isdigit()):
                        yake_str += row[col_name]

    content = content_list
    #content = col_list
    #print(content[0:1000])
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(yake_str)
    key_phrases = []
    for kw in keywords:
        key_phrases.append(kw[0])

# COMMAND ----------

key_word_to_search = "my_table"
Regex = '^\d'
for column in df.columns:
    selected_df = df.filter(col(column).isNotNull() & (col(column) != '*')).select(column).distinct()
    selected_df.show(3)
    total_count = selected_df.count()
    print(column,total_count)
    new_no_of_partitions = int(total_count/10000)
    print(column,new_no_of_partitions)
    if new_no_of_partitions < 5:
        selected_df = selected_df.repartition(5)
    else:
        selected_df = selected_df.repartition(new_no_of_partitions)
    selected_df.foreachPartition(lambda partition: each_partition_func(partition, column,key_word_to_search))

# COMMAND ----------

import re
from pyspark.sql import Row

df = spark.createDataFrame(
    [(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])

def func(itr):
    for person in itr:
        print(person.name)

df.foreachPartition(func)

content_list = []
col_name = 'name'
yake_str = ""
Regex = r"some_regex_pattern"  # Define your regex pattern here

rows = df.collect()
for row in rows:
    if row[col_name] != "":
        content_list.append(row[col_name])
        if not re.search(Regex, row[col_name]):
            if not (row[col_name].isdigit() or 
                    (row[col_name][0] == '-' and row[col_name][1:].isdigit())):
                yake_str += row[col_name]

# Convert content_list to DataFrame
content_df = spark.createDataFrame([Row(name=name) for name in content_list])

# Convert yake_str to DataFrame
yake_df = spark.createDataFrame([Row(yake_str=yake_str)])

display(content_df)
display(yake_df)

# COMMAND ----------

for row in df:
    print(row[col_name])

# COMMAND ----------

content_list.append(row[col_name])
content_list

# COMMAND ----------

# MAGIC %fs ls /mnt/delta/orders/

# COMMAND ----------


