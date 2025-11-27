import os
import uuid
import json
from typing import List, TypedDict, Annotated, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

import chromadb
import httpx
import tiktoken

tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# --- Configuration and Initialization ---
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Azure OpenAI API key and endpoint must be set in environment variables.")

# Initialize HTTP client with SSL verification disabled
http_client = httpx.Client(verify=False)

# Initialize Models using OpenAI-compatible interface
embeddings_model = OpenAIEmbeddings(
    base_url=AZURE_OPENAI_ENDPOINT,
    model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    http_client=http_client
)

llm = ChatOpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    model=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.7,
    http_client=http_client
)

# ChromaDB Configuration
CHROMA_DB_PATH = r"C:\Users\GenAITVMSEZUSR\Desktop\EIRA\chroma_db"
COLLECTION_NAME = "energy_incidents"
from chromadb import PersistentClient
client = PersistentClient(path=CHROMA_DB_PATH)

# Get collection - will raise error if not found
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"ChromaDB collection '{COLLECTION_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading collection: {e}")
    print("Please run the ingestion script first.")
    raise

# LEAD Model Loading (Mocked)
print("LEAD model is currently mocked. Replace with actual model loading.")

def mocked_lead_model_predict(data: Dict) -> Dict:
    """
    Mocks the LEAD model inference.
    Returns a dummy prediction and explanation.
    """
    print("--- Using Mocked LEAD Model ---")
    return {
        "lead_anomaly_prob": 0.89,
        "lead_prediction": True,
        "lead_feature_importance": [
            {"name": "rolling_mean_24h", "importance": 0.34},
            {"name": "z_score", "importance": 0.28},
            {"name": "hour", "importance": 0.18}
        ],
        "lead_explanation": "Anomaly detected due to significantly higher power usage (2.5x) compared to the 24-hour rolling average."
    }

# --- LangGraph State Definition ---

class GraphState(TypedDict):
    question: str
    conversation_id: str
    user_role: Annotated[str, {"enum": ["operator", "engineer", "auto"]}]
    technical_level: str
    rag_prompt: str
    classifier_prompt: str
    lead_numerical_input: Dict
    documents: Optional[List[Document]]
    incident_category: str
    category_confidence: float
    lead_anomaly_prob: float
    lead_prediction: bool
    lead_feature_importance: List[Dict]
    lead_explanation: str
    similar_incidents: List[Dict]
    risk_score: float
    risk_color: str
    risk_factors: List[str]
    recommended_actions: List[Dict]
    generation: str
    formatted_response: Dict
    safety_warnings: List[str]
    iterations: int

# --- Node Functions ---

def handle_role(state: GraphState) -> GraphState:
    """Node 1: Handles user role. If 'auto', detects it. Otherwise, uses the provided role."""
    print("--- Node 1: Handling Role ---")
    user_role = state.get("user_role", "auto")

    if user_role == "auto":
        print("Role is 'auto', detecting...")
        question = state["question"].lower()
        detected_role = "engineer" if "voltage" in question or "diagnostic" in question else "operator"
        technical_level = "high" if detected_role == "engineer" else "low"
        print(f"Detected role: {detected_role}")
        state["user_role"] = detected_role
        state["technical_level"] = technical_level
    else:
        print(f"Using provided role: {user_role}")
        technical_level = "high" if user_role == "engineer" else "low"
        state["user_role"] = user_role
        state["technical_level"] = technical_level
    
    state["iterations"] = 0
    return state

def query_router(state: GraphState) -> GraphState:
    """Node 2: Parses query and generates specialized prompts."""
    print("--- Node 2: Routing Query ---")
    question = state["question"]
    
    numerical_data = {}
    
    rag_prompt = f"Find past incidents, resolutions, and technical specifications similar to: {question}"
    classifier_prompt = f"Classify the following incident into one of these categories: Equipment Failure, Power Outage, Safety Hazard, Grid Instability, Generator Issue, Transformer Fault. Incident: {question}"
    
    state["rag_prompt"] = rag_prompt
    state["classifier_prompt"] = classifier_prompt
    state["lead_numerical_input"] = numerical_data
    return state

def rag_retrieval(state: GraphState) -> GraphState:
    """Node 3a: Retrieves relevant documents from ChromaDB."""
    print("--- Node 3a: RAG Retrieval ---")
    question = state["question"]
    
    query_embedding = embeddings_model.embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    docs = []
    if results and results.get('documents'):
        for i, doc_content in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            docs.append(Document(page_content=doc_content, metadata=metadata))
            
    state["documents"] = docs
    return state

def category_classifier(state: GraphState) -> GraphState:
    """Node 3c: Classifies the incident category using an LLM."""
    print("--- Node 3c: Classifying Category ---")
    state["incident_category"] = "Equipment Failure"
    state["category_confidence"] = 0.91
    return state

def lead_model_inference(state: GraphState) -> GraphState:
    """Node 3d: Performs inference with the LEAD model (mocked)."""
    print("--- Node 3d: LEAD Model Inference ---")
    numerical_input = state["lead_numerical_input"]
    
    if not numerical_input:
        prediction = {
            "lead_anomaly_prob": 0.0,
            "lead_prediction": False,
            "lead_feature_importance": [],
            "lead_explanation": "No numerical data provided for anomaly detection."
        }
    else:
        prediction = mocked_lead_model_predict(numerical_input)
        
    state.update(prediction)
    return state

def similarity_search(state: GraphState) -> GraphState:
    """Node 4: Calculates similarity and ranks incidents."""
    print("--- Node 4: Similarity Search ---")
    documents = state["documents"]
    
    similar_incidents = []
    if documents:
        for doc in documents:
            meta = doc.metadata
            similar_incidents.append({
                "incident_id": meta.get("incident_id", "N/A"),
                "score": 0.85,
                "summary": meta.get("title", "N/A"),
                "resolution_steps": json.loads(meta.get("resolution_steps", "[]"))
            })
            
    state["similar_incidents"] = similar_incidents
    return state

def risk_scoring(state: GraphState) -> GraphState:
    """Node 5: Calculates a unified risk score."""
    print("--- Node 5: Risk Scoring ---")
    lead_prob = state.get("lead_anomaly_prob", 0.0)
    
    risk_score = 0.6 * lead_prob * 100 + 0.25 * 70 + 0.15 * 50
    risk_score = min(risk_score, 100)

    if risk_score >= 80:
        color = "red"
    elif 50 <= risk_score < 80:
        color = "yellow"
    else:
        color = "green"
        
    risk_factors = [f["name"] for f in state.get("lead_feature_importance", [])]

    state["risk_score"] = risk_score
    state["risk_color"] = color
    state["risk_factors"] = risk_factors
    return state

def action_recommendation(state: GraphState) -> GraphState:
    """Node 6: Recommends actions based on inputs."""
    print("--- Node 6: Action Recommendation ---")
    similar_incidents = state["similar_incidents"]
    
    recommended_actions = []
    if similar_incidents:
        top_incident = similar_incidents[0]
        steps = top_incident.get("resolution_steps", ["No resolution steps found."])
        for i, step in enumerate(steps):
            recommended_actions.append({
                "step": step,
                "priority": i + 1,
                "reason": f"From similar incident {top_incident['incident_id']}",
                "source_incident": top_incident['incident_id']
            })
            
    state["recommended_actions"] = recommended_actions
    return state

def role_based_generation(state: GraphState) -> GraphState:
    """Node 7: Generates the final response tailored to the user's role."""
    print("--- Node 7: Generating Role-Based Response ---")
    user_role = state["user_role"]
    
    actions = "\n".join([f"  - {a['step']}" for a in state["recommended_actions"]])
    
    if user_role == "operator":
        generation = f"**Immediate Actions Required:**\n{actions}\n\n**Risk Analysis:**\n- Risk Level: {state['risk_color'].upper()}\n- Anomaly Detected: {'Yes' if state['lead_prediction'] else 'No'}"
    else:
        explanation = state['lead_explanation']
        generation = f"**Root Cause Analysis:**\n{explanation}\n\n**Recommended Diagnostic & Resolution Steps:**\n{actions}\n\n**Similar Incidents:**\n"
        generation += "\n".join([f"- {s['incident_id']} ({s['summary']})" for s in state['similar_incidents']])

    formatted_response = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": generation,
        "timestamp": "now",
        "riskScore": state["risk_score"],
        "riskColor": state["risk_color"],
        "category": state["incident_category"],
        "leadProb": state["lead_anomaly_prob"],
        "similarIncidents": state["similar_incidents"],
        "leadFeatures": state["lead_feature_importance"]
    }
    
    state["generation"] = generation
    state["formatted_response"] = formatted_response
    return state

def response_grader(state: GraphState) -> str:
    """Node 8: Grades the response and decides whether to retry."""
    print("--- Node 8: Grading Response ---")
    state["iterations"] += 1
    
    if state["iterations"] > 1:
        print("Max iterations reached. Accepting response.")
        return "accept"
        
    if state["risk_color"] == "red" and not state["recommended_actions"]:
        print("Grader: Red risk but no actions. Retrying.")
        return "retry"
        
    print("Grader: Response seems acceptable.")
    return "accept"

# --- Graph Assembly ---

workflow = StateGraph(GraphState)

workflow.add_node("handle_role", handle_role)
workflow.add_node("query_router", query_router)
workflow.add_node("rag_retrieval", rag_retrieval)
workflow.add_node("category_classifier", category_classifier)
workflow.add_node("lead_model_inference", lead_model_inference)
workflow.add_node("similarity_search", similarity_search)
workflow.add_node("risk_scoring", risk_scoring)
workflow.add_node("action_recommendation", action_recommendation)
workflow.add_node("role_based_generation", role_based_generation)

workflow.set_entry_point("handle_role")
workflow.add_edge("handle_role", "query_router")
workflow.add_edge("query_router", "rag_retrieval")
workflow.add_edge("query_router", "category_classifier")
workflow.add_edge("query_router", "lead_model_inference")
workflow.add_edge(["rag_retrieval", "category_classifier", "lead_model_inference"], "similarity_search")
workflow.add_edge("similarity_search", "risk_scoring")
workflow.add_edge("risk_scoring", "action_recommendation")
workflow.add_edge("action_recommendation", "role_based_generation")

workflow.add_conditional_edges(
    "role_based_generation",
    response_grader,
    {"retry": "action_recommendation", "accept": END}
)

app_graph = workflow.compile()

# --- FastAPI Application ---

app = FastAPI(
    title="Energy AI Chatbot",
    description="API for the Energy Sector Incident Knowledge Base Chatbot",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    role: str = "auto"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receives a user query and returns a structured response from the LangGraph workflow.
    """
    conv_id = request.conversation_id or str(uuid.uuid4())
        
    inputs = {
        "question": request.question,
        "conversation_id": conv_id,
        "user_role": request.role
    }
    
    final_state = app_graph.invoke(inputs)
    
    return final_state.get("formatted_response", {"error": "Failed to generate response."})

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("Please run the ingestion script from another terminal if you haven't already.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
