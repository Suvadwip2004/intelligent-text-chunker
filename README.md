# Text Chunking and Embedding Project

A Python-based text processing pipeline that intelligently chunks text documents and generates embeddings using the BGE-M3 model for semantic search, RAG (Retrieval-Augmented Generation), and document analysis applications.

## 📋 Table of Contents

- [Where This Project Can Be Used](#where-this-project-can-be-used)
- [Why Use This Project](#why-use-this-project)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Setting Up BGE-M3 Model](#setting-up-bge-m3-model)
- [Project Execution Process](#project-execution-process)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contact](#contact)

## 🎯 Where This Project Can Be Used

This project is designed for various applications that require text processing and semantic understanding:

1. **RAG (Retrieval-Augmented Generation) Systems**
   - Building knowledge bases for AI chatbots
   - Creating context-aware search systems
   - Enhancing LLM responses with relevant document chunks

2. **Semantic Search Applications**
   - Document similarity search
   - Content recommendation systems
   - Intelligent information retrieval

3. **Document Analysis & Processing**
   - Large document preprocessing for ML models
   - Text summarization pipelines
   - Content organization and indexing

4. **Knowledge Management Systems**
   - Corporate knowledge bases
   - Research paper processing
   - Legal document analysis
   - Technical documentation systems

5. **NLP Pipelines**
   - Preprocessing for machine learning models
   - Text vectorization for downstream tasks
   - Content clustering and classification

## 💡 Why Use This Project

1. **Intelligent Chunking**: Uses spaCy's NLP capabilities to create semantically meaningful chunks based on sentence boundaries, ensuring context is preserved.

2. **State-of-the-Art Embeddings**: Leverages BGE-M3 (BAAI General Embedding Model M3), a powerful multilingual embedding model that provides high-quality vector representations.

3. **Efficient Processing**: Processes multiple files in batch, automatically handling the entire pipeline from text input to embedded vectors.

4. **Flexible Output Formats**: Generates both JSON (human-readable) and joblib (efficient binary) formats for different use cases.

5. **Easy Integration**: Simple API that can be easily integrated into larger systems or used as a standalone tool.

6. **Scalable Architecture**: Modular design allows for easy customization and extension of individual components.

## 📊 Project Overview

The project processes text files through three main stages:

1. **Text Chunking** (`create_chunks.py`): Splits text into semantically meaningful chunks using spaCy, with a default maximum of 40 words per chunk.

2. **Embedding Generation** (`process_embedding.py`): Generates vector embeddings for each chunk using the BGE-M3 model via Ollama API.

3. **Data Persistence** (`create_joblib.py`): Saves the processed data in both JSON and joblib formats for easy access and efficient storage.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Ollama installed and running (for BGE-M3 model)

### Step 1: Clone or Navigate to the Project

```bash
cd "create chunks"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- spacy (for NLP and semantic chunking)
- pandas (for data manipulation)
- numpy (for numerical operations)
- joblib (for efficient serialization)
- requests (for API calls to Ollama)

### Step 5: Download spaCy English Model

```bash
python -m spacy download en_core_web_sm
```

## 🤖 Setting Up BGE-M3 Model

This project uses the BGE-M3 embedding model through Ollama. Follow these steps to set it up:

### Step 1: Install Ollama

**Windows:**
1. Download Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer and follow the setup instructions

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Pull the BGE-M3 Model

Open a terminal and run:

```bash
ollama pull bge-m3
```

This will download the BGE-M3 model (approximately 2.3 GB). The download may take a few minutes depending on your internet connection.

### Step 3: Verify Ollama is Running

Start Ollama service (it should start automatically after installation):

**Windows:**
- Ollama should run as a background service automatically
- You can verify by opening: [http://localhost:11434](http://localhost:11434)

**Linux/Mac:**
```bash
ollama serve
```

The API will be available at `http://localhost:11434` by default.

### Step 4: Test the Model

You can test if the model is working correctly:

```bash
ollama run bge-m3 "test embedding"
```

Or test the embedding API directly:

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "bge-m3",
  "input": ["test sentence"]
}'
```

## 📝 Project Execution Process

The project follows this execution flow:

```
Text Files (data/) 
    ↓
[1] Read text files
    ↓
[2] Semantic Chunking (spaCy)
    ├── Split into sentences
    ├── Group sentences into chunks (max 40 words)
    └── Assign chunk IDs
    ↓
[3] Generate JSON structure
    ├── Total chunks count
    └── Chunks array with IDs and text
    ↓
[4] Create Embeddings (BGE-M3 via Ollama)
    ├── Extract text from chunks
    ├── Send batch to Ollama API
    └── Receive vector embeddings
    ↓
[5] Save Results
    ├── JSON file (json_data/)
    └── Joblib file (joblibs/)
    ↓
Complete!
```

### Detailed Process:

1. **File Reading**: The main script reads all text files from the `data/` directory.

2. **Semantic Chunking**: 
   - Uses spaCy's NLP pipeline to parse text
   - Identifies sentence boundaries
   - Groups sentences into chunks with a maximum of 40 words
   - Preserves semantic coherence by respecting sentence boundaries

3. **JSON Creation**: 
   - Creates a structured JSON with metadata (total chunks) and chunk data
   - Each chunk has an ID and text content
   - Saved to `json_data/` directory

4. **Embedding Generation**:
   - Extracts text from all chunks
   - Sends batch request to Ollama API at `http://localhost:11434/api/embed`
   - Uses BGE-M3 model to generate embeddings
   - Attaches embeddings to each chunk

5. **Joblib Export**:
   - Converts the embedded chunks to a pandas DataFrame
   - Saves as a binary joblib file for efficient storage and loading
   - Saved to `joblibs/` directory

## 💻 Usage

### Basic Usage

1. **Prepare Your Text Files**:
   - Place your text files (`.txt` format) in the `data/` directory
   - Ensure Ollama is running with BGE-M3 model loaded

2. **Run the Main Script**:
   ```bash
   python main.py
   ```

3. **Check Output**:
   - JSON files will be in `json_data/` directory
   - Joblib files will be in `joblibs/` directory

### Example Workflow

```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# 2. Ensure Ollama is running with BGE-M3
ollama serve

# 3. Place your text files in data/ directory
# (e.g., document1.txt, document2.txt)

# 4. Run the processing script
python main.py

# 5. Access results
# - json_data/document1.json
# - joblibs/document1.joblib
```

### Loading Processed Data

To load and use the processed data:

```python
import joblib
import pandas as pd

# Load joblib file
df = joblib.load("./joblibs/document1.joblib")

# Access chunks and embeddings
for chunk in df['chunks']:
    chunk_id = chunk['chunk_id']
    text = chunk['text']
    embedding = chunk['embedding']
    # Use embedding for similarity search, etc.
```

## 📁 Project Structure

```
create chunks/
│
├── data/                  # Input text files directory
│   └── *.txt             # Your text files go here
│
├── json_data/            # Output JSON files
│   └── *.json           # Generated JSON chunk files
│
├── joblibs/              # Output joblib files
│   └── *.joblib         # Generated binary files
│
├── src/                  # Source code
│   ├── create_chunks.py      # Semantic chunking module
│   ├── process_embedding.py  # Embedding generation module
│   └── create_joblib.py      # Joblib export module
│
├── main.py               # Main execution script
├── test.py               # Test/utility script
├── requirements.txt      # Python dependencies
├── venv/                 # Virtual environment (not in git)
└── README.md             # This file
```

## 📦 Dependencies

All Python dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

The project requires the following Python packages:

- **spacy**: Natural language processing for semantic chunking
- **pandas**: Data manipulation and DataFrame operations
- **numpy**: Numerical operations for embeddings
- **joblib**: Efficient serialization of Python objects
- **requests**: HTTP requests to Ollama API

### External Services

- **Ollama**: Required for running BGE-M3 embedding model
- **BGE-M3 Model**: Multilingual embedding model (downloaded via Ollama)

## 🔧 Configuration

### Adjusting Chunk Size

To change the maximum words per chunk, modify the `max_words` parameter in `src/create_chunks.py`:

```python
json_chunks = semantic_chunk_text(text, max_words=50)  # Change from default 40
```

### Changing Ollama API Endpoint

If Ollama is running on a different host/port, modify `src/process_embedding.py`:

```python
r = requests.post("http://your-host:your-port/api/embed", json={...})
```

## ⚠️ Troubleshooting

### Ollama Connection Error

If you get connection errors:
- Ensure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama list`
- Check if port 11434 is accessible

### spaCy Model Not Found

If you see spaCy model errors:
```bash
python -m spacy download en_core_web_sm
```

### Memory Issues

For large files:
- Process files individually
- Reduce batch size in embedding requests
- Consider increasing system RAM

## 📧 Contact

For questions, suggestions, or support, please contact:

**Email**: developersuvadwipmaiti@gmail.com

## 📄 License

This project is provided as-is for text processing and embedding generation purposes.

## 🤝 Contributing

Feel free to extend this project for your specific use cases. The modular design makes it easy to customize individual components.

---

**Note**: Make sure Ollama is running and the BGE-M3 model is downloaded before executing the main script. The project expects text files in the `data/` directory and will process all `.txt` files found there.

