# Databricks-Azure-AI-Search
Azure AI Vector + Semantic Search is a powerful feature of Microsoft Azure's AI services that enhances search capabilities by combining vector-based and semantic search techniques. Here's a summary:

### **1. Vector Search**
   - **Purpose**: Designed to handle unstructured data like text, images, and audio by converting them into high-dimensional vectors using AI models.
   - **Functionality**: It allows searching based on the similarity of content rather than just keywords, enabling more relevant results even when exact terms are not used. This is especially useful for complex queries where the exact wording might not match the stored data.

### **2. Semantic Search**
   - **Purpose**: Enhances traditional keyword search by understanding the context and meaning behind search queries.
   - **Functionality**: It leverages natural language processing (NLP) to provide results that are contextually relevant, focusing on the intent of the query rather than just matching keywords. This improves the accuracy and relevance of search results, making it easier to find precise information.

### **3. Combined Capabilities**
   - **Integration**: The combination of vector search with semantic search allows for a more robust and intelligent search experience. Vector search handles complex, unstructured data while semantic search ensures that the results are contextually relevant.
   - **Use Cases**: This combination is particularly beneficial in applications such as enterprise search, customer support, and content discovery, where understanding user intent and content similarity is crucial.

As of the latest updates, Azure Cognitive Search supports several vector search algorithms, with **Hierarchical Navigable Small World (HNSW)** being the primary algorithm used for vector similarity search. In this project we used HNSW.

**HNSW (Hierarchical Navigable Small World)**
   - **Description**: HNSW is a graph-based algorithm widely used for approximate nearest neighbor (ANN) search in high-dimensional spaces. It builds a hierarchical graph of nodes where each node represents a vector, and edges represent proximity or similarity between vectors.
   - **Advantages**:
     - **Efficiency**: HNSW is highly efficient, providing fast search times even with large datasets.
     - **Scalability**: It scales well with both the number of vectors and the dimensionality of the vectors.
     - **Accuracy**: Offers a good balance between search accuracy and performance, making it suitable for real-time applications.
   - **Use Cases**: Ideal for applications requiring high-speed, large-scale vector similarity searches, such as recommendation systems, image retrieval, and natural language processing (NLP) tasks.

Other algorithms are, KNN, Cosine Similarity, Euclidean Distance and Doc Product. Finally, we create the index passing Vector Search and Semantic Search configurations.

### **How HNSW Works:**
HNSW builds a multi-layered graph where:
- **Each layer** contains a subset of the nodes (vectors), with higher layers having fewer nodes.
- **Navigating from top to bottom layers**, the search algorithm progressively narrows down the nearest neighbors by exploring connections between nodes.
- **Edges** in the graph connect nodes that are close to each other, facilitating quick traversal during search queries.

#### **Configure Index:**

```python
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
```
