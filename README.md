# Medical Symptom Diagnosis RAG System

A simple Retrieval-Augmented Generation (RAG) system for diagnosing medical conditions based on symptoms. This system uses a vector database to find similar symptom patterns and an LLM to generate diagnostic assessments.

## Overview

This tool combines:
- **Data Preprocessing**: Processes the symptom-disease dataset
- **Vector Database**: Stores symptom embeddings using ChromaDB and sentence transformers
- **RAG Pipeline**: Retrieves similar cases and generates diagnosis using OpenAI's GPT models

## Features

- Binary symptom matching with 400+ symptoms
- Vector similarity search for finding similar medical cases
- LLM-powered diagnostic explanations
- Interactive command-line interface
- Simple and straightforward implementation

## Dataset

The system uses the Disease Symptom Knowledge Database from Columbia University (Kaggle dataset). The data contains:
- 400+ binary symptom columns
- Multiple disease categories
- Augmented samples for better coverage

## Installation

1. Clone or navigate to the project directory:
```bash
cd /Users/somyabansal/Documents/symptom-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Project Structure

```
symptom-prediction/
├── data/
│   └── fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv
├── data_preprocessing.py    # Data loading and preparation
├── vector_database.py        # Vector embeddings and retrieval
├── rag_pipeline.py           # Main RAG system
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

### Web UI (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

This opens a user-friendly web interface in your browser with:
- Text input or dropdown selection for symptoms
- Visual display of similar cases
- Formatted diagnostic assessments
- Batch diagnosis support
- Real-time system statistics

### Command Line Interface

Run the interactive diagnosis system:

```bash
python rag_pipeline.py
```

Then enter symptoms separated by commas:
```
Enter symptoms: fever, cough, fatigue
```

### Programmatic Usage

```python
from data_preprocessing import DataPreprocessor
from vector_database import VectorDatabase
from rag_pipeline import MedicalRAG

# Initialize components
preprocessor = DataPreprocessor("data/fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv")
preprocessor.load_data()

# Create vector database
vector_db = VectorDatabase()

# Add documents (first time only)
documents = preprocessor.prepare_documents()
vector_db.add_documents(documents)

# Initialize RAG system
rag = MedicalRAG(vector_db)

# Diagnose symptoms
result = rag.diagnose(["shortness of breath", "chest pain", "fatigue"])
print(result['diagnosis'])
```

### Data Preprocessing Only

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor("data/fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv")
preprocessor.load_data()

# View statistics
print(preprocessor.get_disease_statistics())
print(preprocessor.get_symptom_frequency())
```

### Vector Database Only

```python
from vector_database import VectorDatabase

vector_db = VectorDatabase()

# Search for similar cases
results = vector_db.search("fever, headache, nausea", n_results=5)
```

## How It Works

1. **Data Preprocessing**: 
   - Loads the CSV dataset
   - Extracts active symptoms (binary 1s) for each record
   - Creates text representations combining symptoms and diseases

2. **Vector Database**:
   - Uses sentence transformers (all-MiniLM-L6-v2) to create embeddings
   - Stores embeddings in ChromaDB for efficient similarity search
   - Retrieves top-k similar cases based on symptom patterns

3. **RAG Pipeline**:
   - Takes user symptoms as input
   - Retrieves similar cases from vector database
   - Builds context from retrieved cases
   - Uses OpenAI GPT to generate diagnostic assessment
   - Provides explanation with disclaimers

## Configuration

### Vector Database Options

```python
vector_db = VectorDatabase(
    collection_name="medical_symptoms",
    model_name="all-MiniLM-L6-v2",  # Can use other sentence transformer models
    persist_directory="./chroma_db"
)
```

### RAG System Options

```python
rag = MedicalRAG(
    vector_db=vector_db,
    model="gpt-3.5-turbo",  # Can use "gpt-4", "gpt-4-turbo", etc.
    api_key="your-key"
)
```

## Important Disclaimers

- This is a demonstration tool for educational purposes
- NOT a replacement for professional medical advice
- Always consult qualified healthcare providers for medical issues
- The system provides possible diagnoses based on similarity matching
- Accuracy depends on the dataset and model quality

## Troubleshooting

### API Key Issues
If you get an authentication error:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### ChromaDB Persistence
To clear the database:
```python
vector_db.clear_collection()
```

### Memory Issues
For large datasets, adjust batch_size:
```python
vector_db.add_documents(documents, batch_size=50)
```

## Example Output

```
Diagnosing symptoms: shortness of breath, chest pain, and fatigue
Retrieving similar cases...
Generating diagnosis...

======================================================================
DIAGNOSIS REPORT
======================================================================

Symptoms: shortness of breath, chest pain, and fatigue

----------------------------------------------------------------------
Similar Cases from Database:
----------------------------------------------------------------------
Case 1 (Similarity: 87.32%):
Disease: coronary arteriosclerosis
Symptoms: shortness of breath, chest pain, fatigue, palpitation

Case 2 (Similarity: 84.15%):
Disease: myocardial infarction
Symptoms: chest pain, shortness of breath, fatigue, nausea
...

----------------------------------------------------------------------
Diagnostic Assessment:
----------------------------------------------------------------------
Based on the similar cases retrieved from the medical database, the 
symptoms of shortness of breath, chest pain, and fatigue are strongly 
associated with cardiac conditions...
[Full diagnostic explanation]
======================================================================
```

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- chromadb: Vector database
- sentence-transformers: Text embeddings
- openai: LLM API access
- python-dotenv: Environment variable management

## License

This project is for educational purposes. The dataset is from Kaggle (Columbia University Disease Symptom Knowledge Database).

## Future Improvements

- Add support for other LLM providers (Anthropic, local models)
- Implement confidence scores
- Add symptom validation against known symptom list
- Create web interface
- Add medical knowledge base integration
- Implement few-shot learning examples
