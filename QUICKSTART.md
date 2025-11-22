# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key**
```bash
export OPENAI_API_KEY='sk-...'
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

3. **Run the example**
```bash
python example.py
```

This will:
- Load the dataset
- Build the vector database (first run only, ~2-3 minutes)
- Run 3 example diagnoses
- Offer interactive mode

## Web UI (Easiest)

```bash
streamlit run app.py
```

This opens a web interface with:
- Simple symptom input (text or dropdown)
- Visual results display
- Adjustable settings
- Batch diagnosis option

## Interactive Mode

```bash
python rag_pipeline.py
```

Then enter symptoms:
```
Enter symptoms: fever, cough, headache
```

## Files Overview

- `data_preprocessing.py` - Loads and processes the symptom dataset
- `vector_database.py` - Creates embeddings and manages ChromaDB
- `rag_pipeline.py` - Main RAG system with retrieval + LLM generation
- `example.py` - Example usage and demonstrations
- `requirements.txt` - Python dependencies

## How It Works

1. **Preprocessing**: Converts binary symptom data to text format
2. **Embeddings**: Uses sentence transformers to create vector embeddings
3. **Storage**: Stores in ChromaDB for fast similarity search
4. **Retrieval**: Finds top-k similar symptom patterns
5. **Generation**: LLM generates diagnosis based on similar cases

## Customization

Change the LLM model in `rag_pipeline.py`:
```python
rag = MedicalRAG(vector_db, model="gpt-4")  # or "gpt-4-turbo"
```

Change number of similar cases:
```python
result = rag.diagnose(symptoms, n_similar_cases=10)
```

Change embedding model in `vector_database.py`:
```python
vector_db = VectorDatabase(model_name="all-mpnet-base-v2")
```

## Troubleshooting

**"Import could not be resolved"** - Install dependencies:
```bash
pip install -r requirements.txt
```

**API Key Error** - Set environment variable:
```bash
export OPENAI_API_KEY='your-key'
```

**Database Issues** - Clear and rebuild:
```python
vector_db.clear_collection()
documents = preprocessor.prepare_documents()
vector_db.add_documents(documents)
```

## Medical Disclaimer

This is an educational tool. Always consult healthcare professionals for medical advice.
