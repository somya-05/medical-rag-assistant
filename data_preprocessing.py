"""
Data preprocessing module for medical symptom dataset.
Loads and prepares the symptom-disease dataset for RAG system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class DataPreprocessor:
    """Preprocesses medical symptom dataset for RAG system."""
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with dataset path.
        
        Args:
            data_path: Path to the CSV file containing symptom data
        """
        self.data_path = data_path
        self.df = None
        self.symptom_columns = None
        self.disease_column = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV."""
        self.df = pd.read_csv(self.data_path)
        
        # Last column is the disease/diagnosis
        self.disease_column = self.df.columns[-1]
        # All other columns are symptoms
        self.symptom_columns = self.df.columns[:-1].tolist()
        
        print(f"Loaded dataset: {len(self.df)} rows, {len(self.symptom_columns)} symptoms")
        print(f"Number of unique diseases: {self.df[self.disease_column].nunique()}")
        
        return self.df
    
    def get_symptom_list(self, row: pd.Series) -> List[str]:
        """
        Extract active symptoms from a row.
        
        Args:
            row: DataFrame row containing symptom values
            
        Returns:
            List of symptom names where value is 1
        """
        symptoms = []
        for symptom in self.symptom_columns:
            if row[symptom] == 1:
                symptoms.append(symptom)
        return symptoms
    
    def prepare_documents(self) -> List[Dict[str, str]]:
        """
        Prepare documents for vector database.
        Each document contains symptoms and associated disease.
        
        Returns:
            List of dictionaries with 'text', 'disease', and 'symptoms' keys
        """
        documents = []
        
        for idx, row in self.df.iterrows():
            symptoms = self.get_symptom_list(row)
            disease = row[self.disease_column]
            
            # Create text representation
            symptom_text = ", ".join(symptoms)
            text = f"Symptoms: {symptom_text}. Diagnosis: {disease}"
            
            documents.append({
                'text': text,
                'disease': disease,
                'symptoms': symptoms,
                'symptom_text': symptom_text
            })
        
        return documents
    
    def get_disease_statistics(self) -> pd.DataFrame:
        """Get statistics about diseases in the dataset."""
        disease_counts = self.df[self.disease_column].value_counts()
        return pd.DataFrame({
            'disease': disease_counts.index,
            'count': disease_counts.values,
            'percentage': (disease_counts.values / len(self.df) * 100).round(2)
        })
    
    def get_symptom_frequency(self) -> pd.DataFrame:
        """Get frequency of each symptom across all records."""
        symptom_counts = self.df[self.symptom_columns].sum().sort_values(ascending=False)
        return pd.DataFrame({
            'symptom': symptom_counts.index,
            'count': symptom_counts.values,
            'percentage': (symptom_counts.values / len(self.df) * 100).round(2)
        })


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor(
        "data/fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv"
    )
    
    # Load data
    df = preprocessor.load_data()
    
    # Get statistics
    print("\nTop 10 diseases:")
    print(preprocessor.get_disease_statistics().head(10))
    
    print("\nTop 10 most common symptoms:")
    print(preprocessor.get_symptom_frequency().head(10))
    
    # Prepare sample documents
    docs = preprocessor.prepare_documents()
    print(f"\nPrepared {len(docs)} documents for RAG system")
    print(f"\nSample document:\n{docs[0]}")
