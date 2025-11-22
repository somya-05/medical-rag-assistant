# Using Groq API

The system now uses **Groq API** for fast LLM inference.

## Your API Key is Set
✅ Your `.env` file contains: `GROQ_API_KEY=gsk_arXf7QujG5yE9PJolFUi...`

## What Changed

### Model
- **Current:** Llama 3.1 8B Instant (via Groq)
- **Alternative models:** Llama 3.1 70B, Mixtral 8x7B, Gemma2 9B

### Benefits
- **Super fast inference** - Groq provides blazing fast LLM responses
- **Free tier available** - Generous free quota
- **High quality models** - Llama 3.1, Mixtral, Gemma
- **Reliable** - Dedicated inference infrastructure

### How It Works
The system now:
1. Retrieves similar cases from vector database (same as before)
2. Calls Groq API with Llama 3.1
3. Generates diagnosis based on context

## Using the System

### Web UI (Recommended)
```bash
.venv/bin/streamlit run app.py
```

### Command Line
```bash
.venv/bin/python rag_pipeline.py
```

### Test API Connection
```bash
.venv/bin/python rag_pipeline.py
```

## Available Models

You can change the model in `rag_pipeline.py`:

```python
# Fast and efficient (default)
rag = MedicalRAG(vector_db, model="llama-3.1-8b-instant")

# Other good options:
rag = MedicalRAG(vector_db, model="llama-3.1-70b-versatile")
rag = MedicalRAG(vector_db, model="mixtral-8x7b-32768")
rag = MedicalRAG(vector_db, model="gemma2-9b-it")
```

## API Key Setup

Get your free API key from: https://console.groq.com/keys

Add it to your `.env` file:
```
GROQ_API_KEY=your_key_here
```

## Rate Limits

Groq's free tier includes:
- 30 requests per minute
- 14,400 requests per day
- Fast inference speeds (typically <1 second)

## First Run Note

The first time you run the system, it will:
1. Load the vector database (or build it if empty)
2. Connect to Groq API
3. Process your query quickly

## Performance

Groq provides extremely fast inference, typically responding in under 1 second. No warm-up time needed!
