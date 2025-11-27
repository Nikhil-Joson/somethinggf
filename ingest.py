import os
import uuid
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
from chromadb import Client, Settings
import httpx  

load_dotenv()

import tiktoken
client = httpx.Client(verify=False)
tiktoken_cache_dir = r"backend\token"
os.environ["TIKTOKEN_CACHE_DIR"]=tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))


# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Azure OpenAI API key and endpoint must be set in environment variables.")

# Initialize Azure OpenAI Embeddings
embeddings_model =OpenAIEmbeddings(
base_url=AZURE_OPENAI_ENDPOINT ,
model = AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, 
api_key=AZURE_OPENAI_API_KEY,
http_client = client 
)

# Initialize Azure OpenAI Chat Model for synthetic data generation
# llm = AzureChatOpenAI(
#     azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
#     openai_api_version=AZURE_OPENAI_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     openai_api_key=AZURE_OPENAI_API_KEY,
#     temperature=0.7
#     
# )

client = httpx.Client(verify=False) 
llm = ChatOpenAI( 
base_url=AZURE_OPENAI_ENDPOINT ,
model = AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, 
api_key=AZURE_OPENAI_API_KEY,
http_client = client 
)


# ChromaDB Configuration
CHROMA_DB_PATH = r"C:\Users\GenAITVMSEZUSR\Desktop\EIRA\chroma_db"
COLLECTION_NAME = "energy_incidents"

# def generate_synthetic_incident(llm_instance):
#     """Generates a single synthetic energy incident report using an LLM."""
#     prompt = """
#     Generate a realistic energy sector incident report. Include:
#     - A concise title.
#     - A detailed description of the incident, including symptoms and context.
#     - Specific resolution steps.
#     - A category (e.g., Equipment Failure, Grid Instability, Cyber Attack, Human Error, Weather Related).
#     - Some plausible numerical data related to the incident (e.g., voltage drop, temperature spike, power output reduction, duration in hours).
    
#     Format the output as a JSON object with the following keys:
#     "title": "...",
#     "description": "...",
#     "resolution_steps": ["step 1", "step 2", ...],
#     "category": "...",
#     "numerical_data": {"metric1": value1, "metric2": value2, ...}
#     """
    
#     try:
#         response = llm_instance.invoke(prompt)
#         content = response.content
#         # Attempt to parse content as JSON
#         import json
#         incident_data = json.loads(content)
#         return incident_data
#     except Exception as e:
#         print(f"Error generating synthetic incident: {e}")
#         print(f"LLM response content: {content}")
#         return None

def generate_synthetic_incident(llm_instance):
    """Generates a single synthetic energy incident report using an LLM."""
    prompt = """
    Generate a realistic energy sector incident report. Include:
    - A concise title.
    - A detailed description of the incident, including symptoms and context.
    - Specific resolution steps.
    - A category (e.g., Equipment Failure, Grid Instability, Cyber Attack, Human Error, Weather Related).
    - Some plausible numerical data related to the incident (e.g., voltage drop, temperature spike, power output reduction, duration in hours).
    
    Format the output as a JSON object with the following keys:
    "title": "...",
    "description": "...",
    "resolution_steps": ["step 1", "step 2", ...],
    "category": "...",
    "numerical_data": {"metric1": value1, "metric2": value2, ...}
    
    Return ONLY the JSON object without any markdown formatting or code fences.
    """
    
    try:
        response = llm_instance.invoke(prompt)
        content = response.content.strip()
        
        # Remove markdown code fences if present
        if content.startswith("```"):
            # Remove opening fence (```json or ```)
            content = content.split('\n', 1)[1] if '\n' in content else content[3:]
            # Remove closing fence
            if content.endswith("```"):
                content = content.rsplit('\n```', 1)[0]
        
        import json
        incident_data = json.loads(content)
        return incident_data
    except Exception as e:
        print(f"Error generating synthetic incident: {e}")
        print(f"LLM response content: {response.content if 'response' in locals() else 'N/A'}")
        return None

import json

def ingest_data_to_chromadb(num_incidents=5):
    """Generates synthetic incidents and ingests them into ChromaDB."""
    print(f"Initializing ChromaDB client at {CHROMA_DB_PATH}...")
    from chromadb import PersistentClient
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    print(f"Generating and ingesting {num_incidents} synthetic incidents...")
    
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for i in range(num_incidents):
        print(f"Generating incident {i+1}/{num_incidents}...")
        incident = generate_synthetic_incident(llm)
        if incident:
            doc_id = str(uuid.uuid4())
            
            # Prepare document for embedding
            full_description = f"Title: {incident['title']}\nDescription: {incident['description']}"
            
            # Prepare metadata - convert lists and dicts to JSON strings
            metadata = {
                "incident_id": doc_id,
                "title": incident["title"],
                "category": incident["category"],
                "resolution_steps": json.dumps(incident["resolution_steps"]),  # Convert to JSON string
                "numerical_data": json.dumps(incident["numerical_data"])  # Convert to JSON string
            }
            
            documents_to_add.append(full_description)
            metadatas_to_add.append(metadata)
            ids_to_add.append(doc_id)
        else:
            print(f"Skipping incident {i+1} due to generation error.")

    # if documents_to_add:
    #     print("Creating embeddings and adding documents to ChromaDB...")
    #     embeddings = embeddings_model.embed_documents(documents_to_add)
        
        
    #     collection.add(
    #         embeddings=embeddings,
    #         documents=documents_to_add,
    #         metadatas=metadatas_to_add,
    #         ids=ids_to_add
    #     )
    #     print(f"Successfully ingested {len(documents_to_add)} incidents into ChromaDB.")
    # else:
    #     print("No incidents were successfully generated or ingested.")
    if documents_to_add:
        print("Creating embeddings and adding documents to ChromaDB...")
        embeddings = embeddings_model.embed_documents(documents_to_add)
        
        collection.add(
            embeddings=embeddings,
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f"Successfully ingested {len(documents_to_add)} incidents into ChromaDB.")
        
        # ADD THESE LINES TO FIND THE DATABASE
        import os
        abs_path = os.path.abspath(CHROMA_DB_PATH)
        print(f"\n=== DATABASE LOCATION ===")
        print(f"Absolute path: {abs_path}")
        print(f"Exists: {os.path.exists(abs_path)}")
        if os.path.exists(abs_path):
            print(f"Contents: {os.listdir(abs_path)}")
        print(f"Current working directory: {os.getcwd()}")
if __name__ == "__main__":
    # Create a dummy .env file for testing if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("AZURE_OPENAI_API_KEY=YOUR_API_KEY\n")
            f.write("AZURE_OPENAI_ENDPOINT=YOUR_API_ENDPOINT\n")
            f.write("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002\n")
            f.write("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o\n")
        print("Created a dummy .env file. Please fill in your Azure OpenAI API key and endpoint.")
        print("Exiting. Please update .env and run again.")
    else:
        ingest_data_to_chromadb(num_incidents=2)
