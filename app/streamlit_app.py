"""
Streamlit Demo App for Document Intelligence System
Simple web interface for document classification and extraction
"""

import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import DocumentClassifier
from src.ocr import extract_text
from src.ner_kv import NERExtractor, KeyValueExtractor
from src.preprocessing import clean_text
from src.document_processor import DocumentProcessor


# Page config
st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 Document Intelligence System")
st.markdown("Classify documents and extract structured information")

# Sidebar
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input(
    "Model Path",
    value="models/document_classifier.pkl",
    help="Path to trained classifier model"
)

# Initialize components
@st.cache_resource
def load_classifier(model_path):
    """Load classifier with caching."""
    if os.path.exists(model_path):
        return DocumentClassifier(model_path=model_path)
    return None

@st.cache_resource
def load_extractors():
    """Load NER and KV extractors."""
    return NERExtractor(), KeyValueExtractor()

classifier = load_classifier(model_path)
ner_extractor, kv_extractor = load_extractors()

# Main interface
tab1, tab2 = st.tabs(["📝 Text Input", "📁 File Upload"])

with tab1:
    st.header("Enter Text")
    text_input = st.text_area(
        "Document Text",
        height=200,
        placeholder="Paste your document text here..."
    )
    
    if st.button("Process Text", type="primary"):
        if text_input:
            with st.spinner("Processing..."):
                # Clean text
                cleaned_text = clean_text(text_input)
                
                # Classify
                if classifier and classifier.model is not None:
                    predictions = classifier.predict([cleaned_text])
                    probabilities = classifier.predict_proba([cleaned_text])
                    
                    predicted_class = predictions[0]
                    confidence = float(max(probabilities[0]))
                    
                    st.success(f"**Predicted Class:** {predicted_class} (Confidence: {confidence:.2%})")
                    
                    # Show all probabilities
                    if classifier.classes_ is not None:
                        st.subheader("Class Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': classifier.classes_,
                            'Probability': probabilities[0]
                        }).sort_values('Probability', ascending=False)
                        st.bar_chart(prob_df.set_index('Class'))
                else:
                    st.warning("No classifier model loaded. Train a model first.")
                
                # Extract entities
                entities = ner_extractor.extract_entities(cleaned_text)
                if entities:
                    st.subheader("Extracted Entities")
                    for entity_type, entity_list in entities.items():
                        st.write(f"**{entity_type}:** {', '.join(entity_list[:10])}")
                
                # Extract key-values
                kv_pairs = kv_extractor.extract_key_values(cleaned_text)
                if kv_pairs:
                    st.subheader("Key-Value Pairs")
                    kv_df = pd.DataFrame(list(kv_pairs.items()), columns=['Key', 'Value'])
                    st.dataframe(kv_df, use_container_width=True)
                
                # Show cleaned text
                with st.expander("View Cleaned Text"):
                    st.text(cleaned_text)

with tab2:
    st.header("Upload File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx', 'xlsx', 'pptx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a document, spreadsheet, presentation, or image for processing"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        os.makedirs("temp", exist_ok=True)
        temp_path = os.path.join("temp", uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show file info
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        st.info(f"File uploaded: {uploaded_file.name} ({file_ext[1:].upper()} - {uploaded_file.size/1024:.1f} KB)")
        
        if st.button("Process File", type="primary"):
            with st.spinner("Processing file..."):
                # Extract text using the document processor
                processor = DocumentProcessor()
                text = processor.extract_text_from_file(temp_path)
                
                # Show preview for supported file types
                if file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    st.image(uploaded_file, caption="Uploaded Image", width=400)
                elif file_ext == '.pdf':
                    try:
                        # Show first page of PDF as preview
                        images = convert_from_path(temp_path, first_page=1, last_page=1)
                        if images:
                            st.image(images[0], caption="First Page Preview", width=400)
                    except Exception as e:
                        st.warning(f"Couldn't generate PDF preview: {e}")
                
                cleaned_text = clean_text(text)
                
                # Classify
                if classifier and classifier.model is not None:
                    predictions = classifier.predict([cleaned_text])
                    probabilities = classifier.predict_proba([cleaned_text])
                    
                    predicted_class = predictions[0]
                    confidence = float(max(probabilities[0]))
                    
                    st.success(f"**Predicted Class:** {predicted_class} (Confidence: {confidence:.2%})")
                    
                    # Show probabilities
                    if classifier.classes_ is not None:
                        st.subheader("Class Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': classifier.classes_,
                            'Probability': probabilities[0]
                        }).sort_values('Probability', ascending=False)
                        st.bar_chart(prob_df.set_index('Class'))
                else:
                    st.warning("No classifier model loaded.")
                
                # Extract entities
                entities = ner_extractor.extract_entities(cleaned_text)
                if entities:
                    st.subheader("Extracted Entities")
                    for entity_type, entity_list in entities.items():
                        st.write(f"**{entity_type}:** {', '.join(entity_list[:10])}")
                
                # Extract key-values
                kv_pairs = kv_extractor.extract_key_values(cleaned_text)
                if kv_pairs:
                    st.subheader("Key-Value Pairs")
                    kv_df = pd.DataFrame(list(kv_pairs.items()), columns=['Key', 'Value'])
                    st.dataframe(kv_df, use_container_width=True)
                
                # Show extracted text
                with st.expander("View Extracted Text"):
                    st.text(cleaned_text)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This demo uses:
- **TF-IDF + RandomForest** for classification
- **spaCy** for NER
- **Tesseract OCR** for image text extraction
""")

if not classifier or classifier.model is None:
    st.sidebar.warning("⚠️ No model loaded. Train a model first using `python src/train.py`")

