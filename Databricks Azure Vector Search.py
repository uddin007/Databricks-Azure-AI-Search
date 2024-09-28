#!/usr/bin/env python
# coding: utf-8

# ### Install All Packages here in this cell

# In[ ]:


%pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.0.344 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 databricks-sdk==0.12.0 mlflow[databricks] protobuf==3.19.0 
dbutils.library.restartPython()

# ### Enter Credentials

# In[ ]:


# dbutils.widgets.text("azure_search_endpoint","")
# dbutils.widgets.text("primary_admin_key","")
# dbutils.widgets.text("azure_openai_endpoint","")
# dbutils.widgets.text("azure_openai_key","")

# ### Load Sample Data 

# In[ ]:


import json

file_path = '/dbfs/FileStore/tables/text_sample.json'
with open(file_path, 'r', encoding='utf-8') as file:
    input_data = json.load(file)

# In[ ]:


from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")

titles = [item['title'] for item in input_data]
content = [item['content'] for item in input_data]

# In[ ]:


title_response = deploy_client.predict(endpoint="textEmbeddingOpenAImodel", inputs={"input": titles})
print(title_response.data)

# In[ ]:


title_embeddings = [item['embedding'] for item in title_response.data]
print(title_embeddings)

# In[ ]:


embedding_dimensions = len(title_embeddings[0])
print(embedding_dimensions)

# In[ ]:


content_response = deploy_client.predict(endpoint="textEmbeddingOpenAImodel", inputs={"input": content[0]})
content_response_list = content_response.data[0]['embedding']
print(content_response_list)

# In[ ]:


import time

content_embeddings = []

for i in range(len(content)):
    content_response = deploy_client.predict(endpoint="textEmbeddingOpenAImodel", inputs={"input": content[i]})
    content_response_list = content_response.data[0]['embedding']
    content_embeddings.append(content_response_list)
    print(len(content_response_list))
    print(len(content_embeddings))
    print('-------')
    time.sleep(1)

print(content_embeddings)

# In[ ]:


# Generate embeddings for title and content fields
for i, item in enumerate(input_data):
    title = item['title']
    content = item['content']
    item['titleVector'] = title_embeddings[i]
    item['contentVector'] = content_embeddings[i]

print(input_data)

# In[ ]:


# Output embeddings to docVectors.json file
output_path = '/dbfs/FileStore/tables/docVectors.json'
with open(output_path, 'w') as f:
    json.dump(input_data, f)

# In[ ]:


import json

output_path = '/dbfs/FileStore/tables/docVectors.json'
with open(output_path, 'r', encoding='utf-8') as file:
    input_data = json.load(file)

print(input_data)

# In[ ]:


azure_search_endpoint = dbutils.widgets.get("azure_search_endpoint")
primary_admin_key = dbutils.widgets.get("primary_admin_key")
azure_openai_endpoint = dbutils.widgets.get("azure_openai_endpoint")
azure_openai_key = dbutils.widgets.get("azure_openai_key")

# In[ ]:


from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters
)

# Create a search index
azure_search_endpoint = azure_search_endpoint
primary_admin_key = primary_admin_key
azure_openai_endpoint = azure_openai_endpoint
azure_openai_embedding_deployment = "textEmbeddingOpenAImodel"
embedding_model_name = "text-embedding-3-small"
azure_openai_key = azure_openai_key
credential = AzureKeyCredential(primary_admin_key)

index_name = "vector-search-index-01"

index_client = SearchIndexClient(endpoint=azure_search_endpoint, credential=credential)

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchableField(name="category", type=SearchFieldDataType.String, filterable=True),
    SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=embedding_dimensions, vector_search_profile_name="myHnswProfile"),
    SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=embedding_dimensions, vector_search_profile_name="myHnswProfile"),
]

# Configure the vector search configuration  
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(name="myHnsw")
    ],
    profiles=[
        VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw", vectorizer="myVectorizer")
    ],
    vectorizers=[
        AzureOpenAIVectorizer(
            name="myVectorizer",
            azure_open_ai_parameters=AzureOpenAIParameters(
                resource_uri=azure_openai_endpoint,
                deployment_id=azure_openai_embedding_deployment,
                model_name=embedding_model_name,
                api_key=azure_openai_key
            )
        )
    ]
)

semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[SemanticField(field_name="category")],
        content_fields=[SemanticField(field_name="content")]
    )
)

# Create the semantic settings with the configuration
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
result = index_client.create_or_update_index(index)
print(f' {result.name} created')

# In[ ]:


from azure.search.documents import SearchClient

search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=credential)
result = search_client.upload_documents(input_data)
print(f"Uploaded {len(input_data)} documents") 

# In[ ]:


from azure.search.documents.models import VectorizedQuery
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")

query = "Kubernetes"  

response = deploy_client.predict(endpoint="textEmbeddingOpenAImodel", inputs={"input": query}).data[0]
embedding = response['embedding']

vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="contentVector")
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["title", "content", "category"],
)  
  
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"Content: {result['content']}")  
    print(f"Category: {result['category']}\n")  
