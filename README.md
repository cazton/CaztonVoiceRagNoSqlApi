# VoiceRAG: Implementing RAG with Voice Integration Using Azure Cosmos DB NoSQL and GPT-4o Realtime API

This repository provides a comprehensive guide to implementing **retrieval-augmented generation (RAG)** in applications with voice interfaces, powered by the **GPT-4o Realtime API** for audio and **Azure Cosmos DB NoSQL** as a storage solution for vector embeddings.

The application leverages the **text-embedding-ada-002** model from Azure OpenAI to generate document embeddings that facilitate the retrieval of relevant documents during the RAG process.

## Getting Started

Follow these four key steps to run this example in your own environment: preparing prerequisites,
setting up the vector store, configuring the environment, and running the application.

### 1. Prerequisites

Before proceeding, ensure that you have access to the following Azure services:

1. Azure OpenAI:

   - Two model deployments: one for gpt-4o-realtime-preview and another for embeddings (e.g., text-embedding-ada-002).

2. Azure Cosmos DB NoSQL API:
   - This will store your document embeddings and metadata for vector search.

### 2. Configuring the Vector Store with Azure Cosmos DB NoSQL

In this application, document embeddings are stored in Azure Cosmos DB NoSQL. The process involves configuring
Cosmos DB to host the knowledge base (e.g., documents or other content) and the corresponding embeddings for vector search.

#### Steps to Store Documents:

1. **Create a Collection**: Set up a new database and collection in Azure Cosmos DB NoSQL. This collection will hold both the documents and their vector embeddings.

2. **Upload Documents**: Add your documents to the ./data/ folder in the project.

3. **Generate Embeddings**: When documents are processed, embeddings will be created using the text-embedding-ada-002 model. These embeddings consist of 1536-dimensional vectors, enabling semantic search via cosine similarity.

### 3. Setting Up the Environment

To enable the application to communicate with the necessary Azure services, set the following variables as environment variables or add them to a .env file in the root directory:

# Azure OpenAI credentials

```bash
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_DEPLOYMENT=""
AZURE_OPENAI_API_VERSION=""
AZURE_OPENAI_API_KEY=""

# Azure Cosmos DB for NoSQL configuration

COSMOSDB_ACCOUNT_URI=""
COSMOSDB_ACCOUNT_KEY=""
COSMOSDB_DATABASE=""
COSMOSDB_CONTAINER=""
COSMOSDB_VECTOR_EMBEDDINGS_TYPE=""

# Azure OpenAI Embeddings Model configuration

AZURE_OPENAI_EMBEDDINGS_MODEL_NAME=""
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=""
```

### 4. Running the Application

Once your development environment is set up, follow these steps to run the application:

#### Running in VS Code Dev Containers

You can run the project locally using the Dev Containers extension:

1. Start Docker Desktop (if not already installed).

2. Open the project using the link below:

   [![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/azure-samples/aisearch-openai-rag-audio)

3. Once the project files load (this may take several minutes), open a new terminal and follow the next steps.

#### Running Locally

1. Install the required tools:

   - Node.js (https://nodejs.org/en)
   - Python 3.11 or later (https://www.python.org/downloads/)
   - Powershell (https://learn.microsoft.com/powershell/scripting/install/installing-powershell)

2. Set up a Python virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Start the application:
   Windows:

   ```
   cd app
   pwsh .\start.ps1
   ```

   Linux/Mac:

   ```
   cd app
   ./app/start.sh
   ```

4. To incorporate basic knowledge documents into the RAG process, place the relevant PDF files in the ./data/ directory. The app will automatically process and store these documents in the vector store in a collection you specify in the .env. This enables efficient retrieval of relevant information during searches based on specified query parameters. These documents will serve as the foundational data source for generating responses.

5. Access the app at http://localhost:8765.

   Once the application is running, navigating to the URL will display the app's start screen:

### Frontend: Direct Communication with AOAI Realtime API

If needed, you can configure the frontend to communicate directly with the AOAI Realtime API. However, this bypasses the RAG process and exposes your API key, making it unsuitable for production environments.

# Add the following parameters to the useRealtime hook in your frontend:

```typescript
const { startSession, addUserAudio, inputAudioBufferClear } = useRealTime({
    useDirectAoaiApi: true,
    aoaiEndpointOverride: "wss://<NAME>.openai.azure.com",
    aoaiApiKeyOverride: "<YOUR API KEY, INSECURE!!!>",
    aoaiModelOverride: "gpt-4o-realtime-preview",
    ...
});

### Important Notes

The sample PDF documents included in this demo are generated by a language model (Azure OpenAI Service) and are for demonstration purposes only. They do not reflect the opinions or beliefs of Cazton. Cazton makes no guarantees regarding the accuracy, completeness, or suitability of the information provided in this demo.

## Want us to help with your AI projects? Contact: info@cazton.com

Credit: Thanks to Microsoft for the original repo. It uses Azure AI Search. We have modified it to work with Azure Cosmos DB code.
https://github.com/Azure-Samples/aisearch-openai-rag-audio
```
