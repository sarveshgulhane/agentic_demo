# 🤖 Agentic AI Assistant

An intelligent AI-powered assistant built with LangGraph, LangChain, and Ollama that can handle weather queries, answer general questions using RAG (Retrieval-Augmented Generation), and evaluate its own responses.

## ✨ Features

- **🌦️ Weather Integration**: Real-time weather information via OpenWeatherMap API
- **📚 RAG Pipeline**: Document retrieval and question answering from uploaded PDFs
- **🧠 Intelligent Routing**: Automatically routes queries to appropriate handlers
- **🎯 Response Evaluation**: Validates and evaluates response quality
- **💬 Chat UI**: Interactive Streamlit web interface with chat history
- **📁 Document Management**: Upload and ingest PDFs to knowledge base
- **🔍 Vector Search**: Semantic search using Qdrant vector database
- **📝 LLM Processing**: Local LLM inference with Ollama (Ministral 3B model)

## 🏗️ Architecture

```
User Query
    ↓
[Decision Node] - Routes query (weather vs general)
    ↓
    ├─ Weather Route
    │   ├─ [City Node] - Extract city name
    │   ├─ [Weather Node] - Fetch weather data
    │   └─ [Answer Node] - Generate response
    │
    └─ General Route
        ├─ [Context Node] - Retrieve relevant documents (RAG)
        └─ [Answer Node] - Generate response
    ↓
[Evaluation Node] - Validate response quality
    ↓
Final Answer
```

## 📋 Prerequisites

- **Python 3.12+**
- **Ollama** (for local LLM inference)
- **Qdrant** (vector database)
- **OpenWeatherMap API Key** (free tier available)


## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/sarveshgulhane/agentic_demo.git
cd agentic_demo
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies (Optional)
```bash
pip install -r requirements-dev.txt
```

### 5. Setup Ollama
```bash
# Install Ollama from https://ollama.ai

# Pull the Ministral 3B model
ollama pull ministral:3b

# Run Ollama server (in a separate terminal)
ollama serve
```

### 6. Setup Qdrant Vector Database
```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 \
  -e QDRANT__SERVICE__HTTP_PORT=6333 \
  qdrant/qdrant

```

### 7. Environment Configuration
Create a `.env` file in the project root:

```bash
# OpenWeatherMap API
OPENWEATHER_API_KEY=your_api_key_here

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333

# LangChain Tracing (Optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=agentic-demo
```

Get your OpenWeatherMap API key:
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for free account
3. Generate API key from dashboard
4. Add to `.env`

## 📁 Project Structure

```
agentic_demo/
├── main.py                 # Streamlit web interface
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
│
├── graph/                 # LangGraph workflow
│   ├── __init__.py
│   ├── state.py          # Agent state definition
│   ├── workflow.py       # Graph workflow setup
│   └── nodes/            # Individual node implementations
│       ├── decision.py   # Route query (weather vs general)
│       ├── city.py       # Extract city from query
│       ├── weather.py    # Fetch weather data
│       ├── context.py    # Retrieve documents (RAG)
│       ├── answer.py     # Generate response
│       └── evaluation.py # Evaluate response quality
│
├── services/             # External service integrations
│   ├── llm_service.py    # Ollama LLM interface
│   ├── embedding_service.py  # HuggingFace embeddings
│   └── weather_service.py    # OpenWeatherMap API
│
├── rag/                  # RAG pipeline
│   ├── ingestion.py      # PDF ingestion and chunking
│   └── retriever.py      # Document retrieval
│
├── documents/            # User-uploaded PDFs
│
├── tests/                # Test suite
│   ├── conftest.py       # Pytest configuration & fixtures
│   ├── test_llm.py       # LLM service tests
│   ├── test_rag.py       # RAG/retrieval tests
│   ├── test_weather.py   # Weather API tests
│   ├── test_decision.py  # Decision routing tests
│   ├── test_embedding.py # Embedding tests
│   ├── test_integration.py  # End-to-end tests
│   └── test_examples.py  # Test pattern examples
│
└── .github/
    └── workflows/        # CI/CD configuration
```

## 🎮 Usage

### Start the Web Application
```bash
streamlit run main.py
```

The application will open at `http://localhost:8501`

### Using the Interface

1. **Upload Documents**:
   - Click "📄 Document Upload" in sidebar
   - Select PDF files
   - Files are automatically ingested to knowledge base
   - Upload confirmation shows completion

2. **Ask Questions**:
   - Type your query in the chat input
   - Press "🚀 Send" or hit Enter
   - View response in chat history
   - Scroll to see previous conversations

3. **Query Examples**:
   - Weather: "What's the weather in New York?"
   - General: "Explain machine learning"
   - Document-based: "What does the PDF say about AI?"

### API Testing
```bash
# Test individual components
python -c "from services.llm_service import get_llm_response; print(get_llm_response('Hello'))"
```

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## 🔧 Configuration Details

### LLM Configuration
- **Model**: Ministral 3B (3 billion parameters)
- **Temperature**: 0.7 (balanced creativity/consistency)
- **Framework**: LangChain + Ollama

### Embeddings Configuration
- **Model**: all-MiniLM-L6-v2 (HuggingFace)
- **Dimensions**: 384
- **Use Case**: Document semantic search

### RAG Pipeline
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top-K Retrieval**: 3 documents
- **Vector Database**: Qdrant

## 🔌 Integration Points

### External APIs
- **OpenWeatherMap**: Weather data
- **HuggingFace Hub**: Embedding models
- **Qdrant API**: Vector operations

### Local Services
- **Ollama**: LLM inference
- **Qdrant**: Vector database

## 📊 Data Flow

### Weather Query Flow
```
User: "What's the weather in Paris?"
    ↓
Decision Node: Detects weather intent
    ↓
City Node: Extracts "Paris"
    ↓
Weather Service: Calls OpenWeatherMap API
    ↓
Answer Node: Generates response
    ↓
Evaluation Node: Validates quality
    ↓
Response: "It's 15°C with clear skies in Paris"
```

### Document Query Flow
```
User: "What is machine learning?"
    ↓
Decision Node: Routes to general/RAG path
    ↓
RAG Retriever: Searches vector database
    ↓
Retrieved Chunks: 3 most relevant documents
    ↓
LLM: Generates answer using documents
    ↓
Evaluation Node: Validates response
    ↓
Response: Answer based on retrieved documents
```


## 🔒 Security Considerations

- Store API keys in `.env` file (not in code)
- Don't commit `.env` to version control
- Use environment variables for sensitive data
- Validate user inputs
- Sanitize file uploads
