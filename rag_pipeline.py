"""
RAG Pipeline for Medical Symptom Diagnosis.
Combines retrieval from vector database with LLM-based generation using Groq API.
"""

import os
from typing import List, Dict, Optional
from groq import Groq


class MedicalRAG:
    """RAG system for medical diagnosis based on symptoms."""
    
    def __init__(self, vector_db, api_key: Optional[str] = None, 
                 model: str = "llama-3.1-8b-instant"):
        """
        Initialize the RAG system.
        
        Args:
            vector_db: VectorDatabase instance
            api_key: Groq API key (or set GROQ_API_KEY env variable)
            model: Groq model to use for generation (default: llama-3.1-8b-instant)
        """
        self.vector_db = vector_db
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        print(f"Initialized RAG system with Groq model: {model}")
    
    def format_symptoms(self, symptoms: List[str]) -> str:
        """Format symptom list into natural language."""
        if not symptoms:
            return "No symptoms provided"
        
        if len(symptoms) == 1:
            return symptoms[0]
        elif len(symptoms) == 2:
            return f"{symptoms[0]} and {symptoms[1]}"
        else:
            return ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"
    
    def retrieve_similar_cases(self, symptoms: List[str], n_results: int = 5) -> Dict:
        """
        Retrieve similar cases from the vector database.
        
        Args:
            symptoms: List of symptoms
            n_results: Number of similar cases to retrieve
            
        Returns:
            Dictionary with retrieved results
        """
        # Create query from symptoms
        query = self.format_symptoms(symptoms)
        
        # Search vector database
        results = self.vector_db.search(query, n_results=n_results)
        
        return results
    
    def build_context(self, retrieved_results: Dict) -> str:
        """
        Build context string from retrieved results.
        
        Args:
            retrieved_results: Results from vector database search
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (metadata, distance) in enumerate(zip(
            retrieved_results['metadatas'][0],
            retrieved_results['distances'][0]
        )):
            similarity = 1 - distance
            context_parts.append(
                f"Case {i+1} (Similarity: {similarity:.2%}):\n"
                f"Disease: {metadata['disease']}\n"
                f"Symptoms: {metadata['symptom_text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_diagnosis(self, symptoms: List[str], context: str) -> str:
        """
        Generate diagnosis using LLM based on symptoms and retrieved context.
        
        Args:
            symptoms: List of patient symptoms
            context: Context from similar cases
            
        Returns:
            Generated diagnosis explanation
        """
        symptom_text = self.format_symptoms(symptoms)
        
        # Create prompt
        system_prompt = """You are an experienced medical diagnostic assistant. 
Your role is to analyze patient symptoms and provide possible diagnoses based on similar cases from a medical database.

IMPORTANT DISCLAIMERS:
- You are an AI assistant and not a replacement for professional medical advice
- Always recommend consulting with a qualified healthcare provider
- Present information clearly but emphasize the need for professional diagnosis

Based on the similar cases provided, explain:
1. The most likely diagnosis/diagnoses
2. Why these conditions match the symptoms
3. What other symptoms might be expected
4. Recommended next steps (always including seeing a doctor)

Be clear, concise, and professional."""

        user_prompt = f"""Patient presents with the following symptoms:
{symptom_text}

Similar cases from medical database:
{context}

Based on these similar cases, provide a diagnostic assessment."""

        # Use Groq API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error calling Groq API: {str(e)}\n\nPlease check your API key and model availability."
    
    def diagnose(self, symptoms: List[str], n_similar_cases: int = 5, 
                 include_raw_results: bool = False) -> Dict:
        """
        Complete RAG pipeline: retrieve similar cases and generate diagnosis.
        
        Args:
            symptoms: List of patient symptoms
            n_similar_cases: Number of similar cases to retrieve
            include_raw_results: Whether to include raw retrieval results
            
        Returns:
            Dictionary with diagnosis, context, and optionally raw results
        """
        print(f"Diagnosing symptoms: {self.format_symptoms(symptoms)}")
        
        # Retrieval step
        print("Retrieving similar cases...")
        retrieved_results = self.retrieve_similar_cases(symptoms, n_similar_cases)
        
        # Build context
        context = self.build_context(retrieved_results)
        
        # Generation step
        print("Generating diagnosis...")
        diagnosis = self.generate_diagnosis(symptoms, context)
        
        result = {
            'symptoms': symptoms,
            'diagnosis': diagnosis,
            'context': context,
            'n_similar_cases': n_similar_cases
        }
        
        if include_raw_results:
            result['raw_results'] = retrieved_results
        
        return result
    
    def interactive_diagnosis(self):
        """Run interactive diagnosis session."""
        print("\n" + "="*60)
        print("Medical Symptom Diagnosis System (RAG-based)")
        print("="*60)
        print("\nEnter symptoms separated by commas (or 'quit' to exit)")
        print("Example: fever, cough, headache")
        print("="*60 + "\n")
        
        while True:
            user_input = input("\nEnter symptoms: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting diagnosis system. Stay healthy!")
                break
            
            if not user_input:
                print("Please enter at least one symptom.")
                continue
            
            # Parse symptoms
            symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
            
            try:
                # Run diagnosis
                result = self.diagnose(symptoms)
                
                # Display results
                print("\n" + "="*60)
                print("DIAGNOSIS REPORT")
                print("="*60)
                print(f"\nSymptoms: {self.format_symptoms(symptoms)}")
                print("\n" + "-"*60)
                print("Similar Cases from Database:")
                print("-"*60)
                print(result['context'])
                print("\n" + "-"*60)
                print("Diagnostic Assessment:")
                print("-"*60)
                print(result['diagnosis'])
                print("="*60)
                
            except Exception as e:
                print(f"\nError during diagnosis: {str(e)}")
                print("Please try again with different symptoms.")


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from vector_database import VectorDatabase
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable not set")
        print("Please set it before running the RAG system")
        print("Example: Add GROQ_API_KEY to your .env file")
        print("Get your API key from: https://console.groq.com/keys")
    else:
        # Initialize components
        print("Loading data...")
        preprocessor = DataPreprocessor(
            "data/fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv"
        )
        preprocessor.load_data()
        
        print("\nInitializing vector database...")
        vector_db = VectorDatabase()
        
        # Check if database is empty
        if vector_db.get_collection_count() == 0:
            print("Database is empty. Adding documents...")
            documents = preprocessor.prepare_documents()
            vector_db.add_documents(documents)
        else:
            print(f"Using existing database with {vector_db.get_collection_count()} documents")
        
        # Initialize RAG system
        rag = MedicalRAG(vector_db)
        
        # Run interactive session
        rag.interactive_diagnosis()
