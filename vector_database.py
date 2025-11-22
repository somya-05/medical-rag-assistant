"""
Vector database module for storing and retrieving symptom-disease embeddings.
Uses sentence transformers for embeddings and ChromaDB for vector storage.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorDatabase:
    """Manages vector embeddings and retrieval for symptom-disease data."""
    
    def __init__(self, collection_name: str = "medical_symptoms", 
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize vector database.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the sentence transformer model
            persist_directory: Directory to persist the database
        """
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Medical symptom-disease embeddings"}
        )
        
        print(f"Initialized vector database with model: {model_name}")
        
    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 100):
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document dictionaries with 'text', 'disease', 'symptoms'
            batch_size: Number of documents to process at once
        """
        print(f"Adding {len(documents)} documents to vector database...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare data
            ids = [f"doc_{i + j}" for j in range(len(batch))]
            texts = [doc['text'] for doc in batch]
            metadatas = [
                {
                    'disease': doc['disease'],
                    'symptom_text': doc['symptom_text']
                } 
                for doc in batch
            ]
            
            # Generate embeddings
            embeddings = self.model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            if (i + batch_size) % 1000 == 0:
                print(f"Processed {i + batch_size} documents...")
        
        print(f"Successfully added {len(documents)} documents to database")
        
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar symptom patterns.
        
        Args:
            query: Query text (symptom description)
            n_results: Number of results to return
            
        Returns:
            Dictionary containing results with diseases and similarities
        """
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def get_collection_count(self) -> int:
        """Get number of documents in the collection."""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Medical symptom-disease embeddings"}
        )
        print("Collection cleared")


if __name__ == "__main__":
    # Test the vector database
    from data_preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor(
        "data/fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv"
    )
    preprocessor.load_data()
    documents = preprocessor.prepare_documents()
    
    # Create vector database
    vector_db = VectorDatabase()
    
    # Add documents
    vector_db.add_documents(documents)
    
    print(f"\nTotal documents in database: {vector_db.get_collection_count()}")
    
    # Test search
    test_query = "shortness of breath, chest pain, fatigue"
    print(f"\nSearching for: {test_query}")
    results = vector_db.search(test_query, n_results=3)
    
    print("\nTop 3 matches:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. Disease: {metadata['disease']}")
        print(f"   Similarity: {1 - distance:.3f}")
        print(f"   Symptoms: {metadata['symptom_text'][:100]}...")
