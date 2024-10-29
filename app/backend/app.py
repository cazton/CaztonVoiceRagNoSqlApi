
import os
from dotenv import load_dotenv
from aiohttp import web
from ragtools import attach_rag_tools_cosmosdb
from rtmt import RTMiddleTier
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

if __name__ == "__main__":
    load_dotenv()
    llm_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    llm_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    mongo_connection_string = os.environ.get("MONGO_CONNECTION_STRING")
    cosmosdb_uri=os.environ.get("COSMOSDB_ACCOUNT_URI")
    cosmosdb_key=os.environ.get("COSMOSDB_ACCOUNT_KEY")
    database_name = os.environ.get("MONGO_DB_NAME")
    collection_name = os.environ.get("MONGO_COLLECTION_NAME")

    credentials = DefaultAzureCredential() if not llm_key else None

    app = web.Application()

    # Initialize real-time middleware with OpenAI for conversation history tracking
    rtmt = RTMiddleTier(llm_endpoint, llm_deployment, AzureKeyCredential(llm_key) if llm_key else credentials)
    rtmt.system_message = (
        "You are a helpful assistant. Only answer questions based on information you searched in the knowledge base, "
        "accessible with the 'search' tool. "
        "The user is listening to answers with audio, so it's *super* important that answers are as short as possible, "
        "a single sentence if at all possible. "
        "Never read file names or source names or keys out loud. "
        "Always use the following step-by-step instructions to respond: \n"
        "1. Always use the 'search' tool to check the knowledge base before answering a question. \n"
        "2. Always use the 'report_grounding' tool to report the source of information from the knowledge base. \n"
        "3. Produce an answer that's as short as possible. If the answer isn't in the knowledge base, say you don't know."
    )
    
    pdf_dir="../../data"

    # Attach CosmosDB NoSQL database to the real-time middleware.
    attach_rag_tools_cosmosdb(rtmt, cosmosdb_uri, cosmosdb_key, pdf_dir)

    rtmt.attach_to_app(app, "/realtime")

    app.add_routes([web.get('/', lambda _: web.FileResponse('./static/index.html'))])
    app.router.add_static('/', path='./static', name='static')
    web.run_app(app, host='localhost', port=8765)
