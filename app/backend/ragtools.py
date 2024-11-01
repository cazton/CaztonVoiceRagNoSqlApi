from langchain.schema import Document
import re
import os
import uuid
from openai import AzureOpenAI
from dotenv import load_dotenv
from rtmt import Tool, ToolResult, ToolResultDirection
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv(override=True)

# Search tool schema
_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base. The knowledge base is in English, translate to and from English if " + \
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " + \
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
}

# Grounding tool schema
_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " + \
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " + \
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names that were used."
            }
        },
        "required": ["sources"]
    }
}

def chunk_text(text, chunk_size=1000):
    # Split text into chunks of the given size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def add_pdf_documents(pdf_dir, chunk_size=1000):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            print("Processing File:", filename)
            filepath = os.path.join(pdf_dir, filename)
            loader = PDFPlumberLoader(filepath)
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=250)
            loaded_documents = loader.load_and_split(text_splitter)
            
            # Add metadata to each chunk
            for i, doc in enumerate(loaded_documents):
                doc.metadata = {"title": f"{filename}_chunk_{i}"}
                documents.append(doc)
                
    return documents

# Vector search using CosmosDB Vector Store
def vector_search(query, vector_store):
    # Perform similarity search on the query
    docs = vector_store.similarity_search(query)
    return docs

def _search_tool(vector_store, args):
    query = args['query']
    print(f"Searching for '{query}' in the knowledge base.")

    # Perform vector search using CosmosDB vector store
    results = vector_search(query, vector_store)
    
    # Format results to be sent as a system message to the LLM
    result_str = ""
    for i, doc in enumerate(results):
        truncated_content = doc.page_content[:2000] if len(doc.page_content) > 2000 else doc.page_content
        print(truncated_content)
        result_str += f"[doc_{i}]: Content: {truncated_content}\n-----\n"
    if not result_str or result_str.isspace():
        result_str = "1"
    
    return ToolResult(result_str, ToolResultDirection.TO_SERVER)

def _report_grounding_tool(vector_store, args):
    sources = args["sources"]
    valid_sources = [s for s in sources if re.match(r'^[a-zA-Z0-9_=\-]+$', s)]
    list_of_sources = " OR ".join(valid_sources)
    print(f"Grounding source: {list_of_sources}")

    # Fetch documents from the vector store using the search functionality
    search_results = []

    for source in valid_sources:
        # Perform a vector search using the title as the query
        docs = vector_store.similarity_search(source)
        for doc in docs:
            search_results.append({
                "content": doc.page_content,
                "metadata": doc
            })

    # Format the results
    result_str = ""
    for result in search_results:
        result_str += f"[{result['content'][:200]}...\n-----\n"
    
    if not result_str or result_str.isspace():
        result_str = "1"

    return ToolResult(result_str.strip(), ToolResultDirection.TO_SERVER)

def check_index_exists(collection, index_name):
    indexes = collection.index_information()
    return index_name in indexes

def check_vector_store_empty(vector_store):
    return vector_store._collection.count_documents({}) == 0

# Initialize CosmosDB client
def init_cosmosdb_client(cosmsodb_uri, cosmosdb_key):
    return CosmosClient(url=cosmsodb_uri, credential=cosmosdb_key)

def get_vector_indexing_policy(embedding_path, embedding_type):
    for i in range(0, len(embedding_type)):
        vectorIndexes = []
        vectorIndex = {"path": embedding_path[0], "type": f"{embedding_type[0]}"}
        vectorIndexes.append(vectorIndex)
        
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        'excludedPaths': [{'path': '/"_etag"/?'}],
        "vectorIndexes": vectorIndexes
    }

def get_vector_embedding_policy(embedding_path, distance_function, data_type, dimensions):
    for i in range(0, len(distance_function)):
        vectorEmbeddings = []
        vectorEmbedding = {
                    "path": embedding_path[0],
                    "dataType": f"{data_type[0]}",
                    "dimensions": dimensions[0],
                    "distanceFunction": f"{distance_function[0]}"
                }
        vectorEmbeddings.append(vectorEmbedding)
        
    return {
        "vectorEmbeddings": vectorEmbeddings
    }

# Check and create Cosmos DB Database and Container
def check_and_create_cosmosdb_database_container(cosmos_client, database_name, container_name, indexing_policy, vector_embedding_policy):
    database = cosmos_client.create_database_if_not_exists(database_name)
    print('Database with id \'{0}\' created'.format(database_name))
    
    container = database.create_container_if_not_exists(
        id=container_name,
        partition_key=PartitionKey(path="/id"),
        offer_throughput=30000,
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy
    )

    return container

def attach_rag_tools_cosmosdb(rtmt, cosmosdb_uri, cosmosdb_key, pdf_dir):
    
    cosmos_client = init_cosmosdb_client(cosmosdb_uri, cosmosdb_key)
    
    azure_openai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment= os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
    )

    cosmosdb_databse = os.getenv("COSMOSDB_DATABASE")
    cosmosdb_container = os.getenv("COSMOSDB_CONTAINER")
   
    cosmos_db_vector_embedding_type = os.getenv("COSMOSDB_VECTOR_EMBEDDINGS_TYPE")

    # cosmosdb_container= "diskann3"
    # cosmos_db_vector_embedding_type="diskANN"
    # cosmos_db_vector_embedding_type="quantizedFlat"
    
    partition_key = PartitionKey(path="/id")
    cosmos_container_properties = {"partition_key": partition_key}

    indexing_policy=get_vector_indexing_policy(
        embedding_path=["/embedding"],
        embedding_type=[cosmos_db_vector_embedding_type]
        )
    vector_embedding_policy=get_vector_embedding_policy(
        embedding_path=["/embedding"],
        distance_function=["cosine"],
        data_type=["float32"],
        dimensions=[1536]
        )
    
    check_and_create_cosmosdb_database_container(cosmos_client, cosmosdb_databse, cosmosdb_container, indexing_policy, vector_embedding_policy)

    documents = add_pdf_documents(pdf_dir)
    print("Documents", len(documents))

    vector_store = AzureCosmosDBNoSqlVectorSearch.from_documents(
        documents=documents,
        embedding=azure_openai_embeddings,
        cosmos_client=cosmos_client,
        database_name=cosmosdb_databse,
        container_name=cosmosdb_container,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_database_properties={
            "id": cosmosdb_databse,
            },
        cosmos_container_properties=cosmos_container_properties,
    )

    # Attach search and grounding tools
    rtmt.tools["search"] = Tool(
        schema=_search_tool_schema,
        target=lambda args: _search_tool(vector_store, args)
    )
    rtmt.tools["report_grounding"] = Tool(
        schema=_grounding_tool_schema,
        target=lambda args: _report_grounding_tool(vector_store, args)
    )

# generate openai embeddings
def generate_embeddings_azure(text, openai_embeddings):    
    response = openai_embeddings.embed_query(text)
    return response