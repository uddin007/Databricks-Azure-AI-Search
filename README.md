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

Other algorithms are, KNN, Cosine Similarity, Euclidean Distance and Doc Product. 

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

Finally, we create the index passing Vector Search and Semantic Search configurations.

```python
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
result = index_client.create_or_update_index(index)
```

When performing a vector query in Azure Cognitive Search, several key parameters control how the search is conducted, how results are retrieved, and how performance and accuracy are balanced. These parameters allow you to tailor the vector search to specific use cases. Below are the key parameters typically involved in a vector query:

### 1. **Vector**
   - **`vector`**: The query vector itself, usually represented as an array of floating-point numbers (a vector embedding). This is the core input for the vector search, representing the feature space of the item or query you want to find similar results for.
   
### 2. **Fields**
   - **`fields`**: The name of the vector field in the search index where the vector embeddings are stored. This tells the search engine which field to compare the query vector against.

   **Example:**
   ```python
  fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchableField(name="category", type=SearchFieldDataType.String, filterable=True),
    SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True,     vector_search_dimensions=embedding_dimensions, vector_search_profile_name="myHnswProfile"),
    SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=embedding_dimensions, vector_search_profile_name="myHnswProfile"),
]
   ```

### 3. **k**
   - **`k`**: The number of nearest neighbors (similar vectors) to return in the search results. This determines how many results the search should return, typically based on proximity in the vector space.

   **Example:**
   ```python
   k=5
   ```

### 4. **Vector Search Algorithm Parameters**
   - **`efSearch`** (for HNSW):
     - A parameter specific to the HNSW (Hierarchical Navigable Small World) algorithm that controls the trade-off between search speed and accuracy. `efSearch` determines the size of the dynamic list of nearest neighbors explored during the search. Higher values typically improve recall but increase query time.

   **Example:**
   ```python
   "efSearch": 100
   ```

### 5. **Filter**
   - **`filter`**: An optional OData filter expression that can be applied to narrow down the search results. This allows you to combine vector search with traditional filtering based on other document fields (e.g., date, category).

   **Example:**
   ```python
   filter="category eq 'news'"
   ```

### 6. **Search Text**
   - **`search_text`**: A keyword or phrase that can be used in combination with the vector search to refine the results. This allows you to conduct a hybrid search that considers both vector similarity and keyword matching.

   **Example:**
   ```python
   search_text="example document"
   ```

### 7. **OrderBy**
   - **`orderBy`**: Specifies how to sort the results. Typically, vector search results are ordered by similarity score (distance) by default, but you can further customize sorting based on other fields.

   **Example:**
   ```python
   orderBy="date desc"
   ```

### 8. **Select**
   - **`select`**: Specifies which fields to include in the returned results. This is useful for reducing the size of the response by only including relevant information.

   **Example:**
   ```python
   select=["id", "title", "content"]
   ```

### 9. **Top**
   - **`top`**: Specifies the maximum number of documents to retrieve. Similar to `k`, but more general for non-vector queries; in vector search, `k` is more commonly used to specify the number of nearest neighbors.

   **Example:**
   ```python
   top=10
   ```

These parameters give you fine-grained control over how vector searches are conducted in Azure Cognitive Search, allowing you to optimize for speed, accuracy, and relevance depending on your specific application needs.
