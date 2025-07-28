# langchain_fastapi_rag.py
import os
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
import json
from logging_config import logger

# === Environment and Logging Setup ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
os.environ["OPENAI_API_KEY"] = api_key

## Loguru logger is now set up via logging_config.py

# === FastAPI App Initialization ===
app = FastAPI(title="Fashion RAG API with LangGraph")
logger.info("FastAPI app initialized.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic Models ===
class QueryInput(BaseModel):
    query: str
    user_preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "shirt for brunch date outfit for men less than 1000 rs",
                "user_preferences": {
                    "gender": "Men",
                    "usage": "Casual", 
                    "master_category": "Apparel",
                    "price_range": [None, 1000]
                }
            }
        }

# === AI Model and VectorDB Setup ===
logger.info("Initializing AI models and vector database...")
try:
    embed_model_name = "dejasi5459/clip-fashion-embeddings-final-10k-ft"
    embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"trust_remote_code": True}
    )
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    persist_directory = "fashion_vector_db"
    collection_name = "fashion_products"
    if not os.path.exists(persist_directory):
        error_msg = f"Vector DB directory '{persist_directory}' not found."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )
    logger.info(f"Connected to ChromaDB collection '{collection_name}'.")
except Exception as e:
    logger.exception("Failed to initialize models or VectorDB.")
    raise e

# === Fashion RAG Retriever ===
class FashionRAGRetriever:
    def __init__(self, vectorstore: Chroma):
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        logger.info("FashionRAGRetriever initialized.")

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        logger.info(f"Performing search for query: '{query}' with filters: {filters}")
        
        chroma_filter = {}
        if filters:
            conditions = []
            
            # Gender mapping
            if "gender" in filters:
                conditions.append({"gender": {"$eq": filters["gender"]}})
            
            # Article type mapping
            if "article_type" in filters:
                conditions.append({"article_type": {"$eq": filters["article_type"]}})
            
            # Master category mapping
            if "master_category" in filters:
                conditions.append({"master_category": {"$eq": filters["master_category"]}})
            
            # Usage mapping (direct mapping)
            if "usage" in filters:
                conditions.append({"usage": {"$eq": filters["usage"]}})
            
            # Price range filtering
            if "price_range" in filters and isinstance(filters["price_range"], list) and len(filters["price_range"]) == 2:
                min_price, max_price = filters["price_range"]
                price_condition = {}
                if min_price is not None:
                    price_condition["$gte"] = min_price
                if max_price is not None:
                    price_condition["$lte"] = max_price
                if price_condition:
                    conditions.append({"price": price_condition})

            if conditions:
                if len(conditions) > 1:
                    chroma_filter = {"$and": conditions}
                else:
                    chroma_filter = conditions[0]
        
        logger.info(f"Using ChromaDB filter: {chroma_filter}")
        results = self.retriever.vectorstore.similarity_search(query, k=10, filter=chroma_filter or None)
            
        logger.info(f"Retrieved {len(results)} documents from vector store.")
        return results

fashion_retriever = FashionRAGRetriever(vectorstore)

# === LangGraph State and Nodes ===
class FashionRAGState(TypedDict):
    query: str
    user_preferences: Optional[Dict]
    retrieved_products: List[Document]
    recommendation: str

def retrieve_node(state: FashionRAGState) -> FashionRAGState:
    logger.info("Executing 'retrieve' node.")
    query = state["query"]
    user_preferences = state.get("user_preferences", {})
    
    # Clean up user preferences - remove empty/invalid preferences
    cleaned_preferences = {}
    if user_preferences:
        for key, value in user_preferences.items():
            # Skip invalid keys and empty values
            if key != "additionalProp1" and value and value != {}:
                cleaned_preferences[key] = value
    
    logger.info(f"Using cleaned preferences for search: {cleaned_preferences}")
    
    retrieved_docs = fashion_retriever.search(query, cleaned_preferences)
    
    return {
        "query": query,
        "user_preferences": cleaned_preferences,
        "retrieved_products": retrieved_docs,
        "recommendation": ""
    }

def format_products_for_prompt(products: List[Document]) -> str:
    if not products:
        return "No products found matching the criteria."
    
    formatted = []
    for i, doc in enumerate(products, 1):
        metadata = doc.metadata
        product_info = [f"{i}. {metadata.get('product_name', 'Unknown Product')}"]
        if metadata.get('gender'): product_info.append(f"   - Gender: {metadata['gender']}")
        if metadata.get('master_category'): product_info.append(f"   - Category: {metadata['master_category']}")
        if metadata.get('sub_category'): product_info.append(f"   - Type: {metadata['sub_category']}")
        if metadata.get('base_colour'): product_info.append(f"   - Color: {metadata['base_colour']}")
        if metadata.get('usage'): product_info.append(f"   - Usage: {metadata['usage']}")
        if metadata.get('price'): product_info.append(f"   - Price: ₹{metadata['price']:.2f}")
        formatted.append('\n'.join(product_info))
    
    return '\n\n'.join(formatted)

def generate_node(state: FashionRAGState) -> FashionRAGState:
    logger.info("Executing 'generate' node.")
    query = state["query"]
    user_preferences = state.get("user_preferences", {})
    retrieved_products = state["retrieved_products"]

    products_text = format_products_for_prompt(retrieved_products)
    preferences_text = json.dumps(user_preferences, indent=2) if user_preferences else "No specific preferences provided"

    fashion_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional fashion stylist. Your role is to provide personalized fashion recommendations based on retrieved product information. Given a user's query and a list of fashion products, provide thoughtful styling advice. Be specific about why each item works."""),
        ("human", "User Query: {query}\n\nUser Preferences: {preferences}\n\nRetrieved Fashion Products:\n{products}\n\nPlease provide a detailed fashion recommendation.")
    ])
    
    chain = fashion_prompt | llm
    
    try:
        response = chain.invoke({
            "query": query,
            "preferences": preferences_text,
            "products": products_text
        })
        recommendation = str(response.content)
        logger.info("Successfully generated fashion recommendation.")
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}")
        recommendation = "I apologize, but I encountered an error while generating a recommendation. Please try again."

    # The state passed to the next node must conform to the state's TypedDict definition
    return {
        "query": query,
        "user_preferences": user_preferences,
        "retrieved_products": retrieved_products,
        "recommendation": recommendation
    }


# === Build and Compile LangGraph Workflow ===
logger.info("Building LangGraph workflow...")
workflow = StateGraph(FashionRAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
fashion_rag_app = workflow.compile()
logger.info("LangGraph workflow compiled successfully.")

# === FastAPI Endpoints ===
@app.post("/query")
def query_rag(input_data: QueryInput):
    logger.info(f"Received query: '{input_data.query}' with preferences: {input_data.user_preferences}")
    try:
        initial_state: FashionRAGState = {
            "query": input_data.query,
            "user_preferences": input_data.user_preferences or {},
            "retrieved_products": [],
            "recommendation": ""
        }
        result = fashion_rag_app.invoke(initial_state)
        logger.info("Graph workflow completed successfully.")
        
        return {
            "recommendation": result["recommendation"],
            "retrieved_products": [doc.metadata for doc in result["retrieved_products"]]
        }
    except Exception as e:
        logger.exception("Error processing query in /query endpoint.")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Fashion RAG API is running!"}

# === Main Execution ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)