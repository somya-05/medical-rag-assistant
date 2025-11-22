"""
Streamlit Web UI for Medical Symptom Diagnosis RAG System.
Simple and intuitive interface for symptom-based diagnosis.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from data_preprocessing import DataPreprocessor
from vector_database import VectorDatabase
from rag_pipeline import MedicalRAG

# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Medical Symptom Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #1e3a5f;
        border-left: 5px solid #4a90e2;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .result-box {
        background-color: #1e3a5f;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .symptom-chip {
        background-color: #1e3a5f;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system (cached to avoid reloading)."""
    with st.spinner("Loading system components..."):
        # Load data
        preprocessor = DataPreprocessor(
            "data/fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv"
        )
        preprocessor.load_data()
        
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Build database if empty
        if vector_db.get_collection_count() == 0:
            st.info("Building vector database for the first time. This may take 2-3 minutes...")
            documents = preprocessor.prepare_documents()
            vector_db.add_documents(documents, batch_size=100)
            st.success(f"Database built with {len(documents)} documents!")
        
        # Initialize RAG
        rag = MedicalRAG(vector_db)
        
        return rag, preprocessor, vector_db


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Symptom Diagnosis System</h1>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Important Medical Disclaimer:</strong><br>
        This is an AI-powered educational tool and NOT a substitute for professional medical advice.
        Always consult qualified healthcare providers for medical diagnosis and treatment.
        This system is for demonstration purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in environment variables!")
        st.info("Please set your Groq API key:")
        st.code("Get your key from: https://console.groq.com/keys")
        st.info("Or create a .env file with: GROQ_API_KEY=your-key-here")
        st.stop()
    
    # Initialize system
    try:
        rag, preprocessor, vector_db = initialize_system()
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()
    
    # Sidebar - System Info
    with st.sidebar:
        st.header("ℹ️ System Information")
        st.metric("Documents in Database", vector_db.get_collection_count())
        st.metric("Total Symptoms", len(preprocessor.symptom_columns))
        st.metric("Unique Diseases", preprocessor.df[preprocessor.disease_column].nunique())
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        n_cases = st.slider(
            "Number of similar cases to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More cases provide more context but may slow down processing"
        )
        
        show_cases = st.checkbox("Show retrieved cases", value=True)
        
        st.markdown("---")
        st.header("📊 Dataset Statistics")
        
        if st.button("Show Top Diseases"):
            st.dataframe(
                preprocessor.get_disease_statistics().head(10),
                width="stretch"
            )
        
        if st.button("Show Common Symptoms"):
            st.dataframe(
                preprocessor.get_symptom_frequency().head(10),
                width="stretch"
            )
    
    # Main content
    st.header("Enter Patient Symptoms")
    
    # Tab interface
    tab1, tab2 = st.tabs(["🔍 Single Diagnosis", "📋 Batch Diagnosis"])
    
    with tab1:
        # Input methods
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio(
                "Input Method:",
                ["Text Input", "Select from List"],
                horizontal=True
            )
        
        symptoms = []
        
        if input_method == "Text Input":
            symptom_input = st.text_area(
                "Enter symptoms (comma-separated):",
                placeholder="e.g., fever, cough, headache, fatigue",
                height=100,
                help="Enter symptoms separated by commas"
            )
            
            if symptom_input:
                symptoms = [s.strip() for s in symptom_input.split(',') if s.strip()]
        
        else:
            # Multi-select from available symptoms
            common_symptoms = preprocessor.get_symptom_frequency().head(50)['symptom'].tolist()
            
            symptoms = st.multiselect(
                "Select symptoms:",
                options=common_symptoms,
                help="Select one or more symptoms from the list"
            )
        
        # Display selected symptoms
        if symptoms:
            st.markdown("**Selected Symptoms:**")
            symptom_html = " ".join([f'<span class="symptom-chip">{s}</span>' for s in symptoms])
            st.markdown(symptom_html, unsafe_allow_html=True)
        
        # Diagnosis button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            diagnose_button = st.button("🔬 Get Diagnosis", type="primary", width="stretch")
        with col2:
            if st.button("🔄 Clear", width="stretch"):
                st.rerun()
        
        # Run diagnosis
        if diagnose_button and symptoms:
            with st.spinner("Analyzing symptoms..."):
                try:
                    result = rag.diagnose(
                        symptoms, 
                        n_similar_cases=n_cases,
                        include_raw_results=True
                    )
                    
                    # Display results
                    st.success("✅ Analysis Complete")
                    
                    # Similar cases
                    if show_cases:
                        st.markdown("---")
                        st.subheader("📚 Similar Cases from Database")
                        
                        for i, (metadata, distance) in enumerate(zip(
                            result['raw_results']['metadatas'][0],
                            result['raw_results']['distances'][0]
                        ), 1):
                            similarity = (1 - distance) * 100
                            
                            with st.expander(f"Case {i}: {metadata['disease']} (Similarity: {similarity:.1f}%)"):
                                st.write(f"**Disease:** {metadata['disease']}")
                                st.write(f"**Similarity Score:** {similarity:.2f}%")
                                st.write(f"**Symptoms:** {metadata['symptom_text']}")
                    
                    # Diagnosis
                    st.markdown("---")
                    st.subheader("🩺 Diagnostic Assessment")
                    st.markdown(f'<div class="result-box">{result["diagnosis"]}</div>', 
                               unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during diagnosis: {str(e)}")
        
        elif diagnose_button and not symptoms:
            st.warning("⚠️ Please enter at least one symptom")
    
    with tab2:
        st.info("Upload a CSV file with symptoms or enter multiple cases")
        
        batch_input = st.text_area(
            "Enter multiple cases (one per line, symptoms comma-separated):",
            placeholder="fever, cough\nheadache, nausea, dizziness\nchest pain, shortness of breath",
            height=200
        )
        
        if st.button("🔬 Batch Diagnose", type="primary"):
            if batch_input:
                cases = [line.strip() for line in batch_input.split('\n') if line.strip()]
                
                with st.spinner(f"Processing {len(cases)} cases..."):
                    for i, case in enumerate(cases, 1):
                        symptoms = [s.strip() for s in case.split(',') if s.strip()]
                        
                        st.markdown(f"### Case {i}")
                        st.write(f"**Symptoms:** {', '.join(symptoms)}")
                        
                        try:
                            result = rag.diagnose(symptoms, n_similar_cases=3)
                            
                            with st.expander("View Diagnosis", expanded=False):
                                st.write(result['diagnosis'])
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                        
                        st.markdown("---")
            else:
                st.warning("Please enter at least one case")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p>Powered by RAG (Retrieval-Augmented Generation) | 
        Using Groq Llama 3.1 & ChromaDB | 
        Built with Streamlit</p>
        <p><strong>Remember:</strong> This is an educational tool. Always consult healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
