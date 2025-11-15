# Document Intelligence System

An end-to-end pipeline for processing unstructured documents (invoices, receipts, resumes, contracts) using OCR, NLP, and machine learning.

## Features

- **OCR**: Extract text from images using Tesseract
- **Classification**: Document type classification using TF-IDF + RandomForest
- **NER**: Named Entity Recognition using spaCy
- **Key-Value Extraction**: Structured information extraction
- **Regression**: Predict numerical values from documents
- **Visualization**: Plotting and analysis tools
- **Web Demo**: Streamlit-based interactive interface

## Project Structure

```
ML_Project/
├── data/
│   ├── examples/          # Sample documents/images
│   └── labels/
│       └── train.csv      # Training data with labels
├── src/
│   ├── ocr.py            # OCR text extraction
│   ├── preprocessing.py  # Text cleaning utilities
│   ├── features.py       # Feature extraction
│   ├── classifier.py     # Document classification
│   ├── ner_kv.py         # NER and key-value extraction
│   ├── regression.py     # Regression models
│   ├── visualize.py      # Visualization tools
│   ├── train.py          # Training script
│   └── infer.py          # Inference script
├── app/
│   └── streamlit_app.py  # Web demo interface
├── models/               # Saved models (created after training)
├── plots/                # Generated plots (created after training)
├── train_layoutlm.py     # LayoutLM/BERT fine-tuning scaffold
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location: `C:\Program Files\Tesseract-OCR\`
- The code will auto-detect the installation path

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### 2. Create Python Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Training the Classifier

1. Prepare your training data in `data/labels/train.csv` with columns:
   - `filename`: Name of the file (or use `text` column directly)
   - `label`: Document type (invoice, receipt, resume, contract, etc.)
   - `text`: (Optional) Text content of the document

2. Run training:
```bash
python src/train.py --data data/labels/train.csv --model models/document_classifier.pkl
```

The script will:
- Load training data
- Train the classifier
- Save the model to `models/document_classifier.pkl`
- Generate plots in `plots/` directory

### Running Inference

Classify documents and extract information:

```bash
# Process a single file
python src/infer.py path/to/document.txt --model models/document_classifier.pkl

# Process a directory
python src/infer.py data/examples --model models/document_classifier.pkl --output results.json

# Verbose output
python src/infer.py data/examples --verbose
```

### Streamlit Web Demo

Launch the interactive web interface:

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to the URL shown (typically http://localhost:8501).

## Example Training Data Format

The `data/labels/train.csv` file should have the following format:

```csv
filename,label,text
invoice_001.txt,invoice,"Invoice #12345 Date: 01/15/2024 Total: $100.00"
receipt_001.txt,receipt,"Receipt Thank you for your purchase Date: 01/15/2024"
resume_001.txt,resume,"John Doe Resume Experience: Software Engineer"
contract_001.txt,contract,"Contract Agreement between Party A and Party B"
```

## Upgrading to LayoutLM/BERT

For better accuracy, especially with complex document layouts, you can fine-tune LayoutLM or BERT models. See `train_layoutlm.py` for a scaffold implementation.

**Requirements for LayoutLM:**
- Annotated dataset with bounding boxes and labels
- Format: JSON with text, bounding boxes, and entity labels
- Requires `transformers` and `torch` libraries (already in requirements.txt)

The scaffold file includes:
- Dataset loading structure
- Model initialization
- Training loop skeleton
- Comments on required data format

## Model Files

After training, the following files are created:
- `models/document_classifier.pkl`: Trained classifier
- `models/document_classifier_vectorizer.pkl`: TF-IDF vectorizer
- `plots/confusion_matrix.png`: Classification performance
- `plots/class_distribution.png`: Training data distribution

## Supported Document Types

- **Invoices**: Extract invoice numbers, dates, totals, vendor info
- **Receipts**: Extract purchase details, amounts, dates
- **Resumes**: Extract contact info, experience, education, skills
- **Contracts**: Extract parties, dates, terms, values

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `pytesseract`: OCR
- `spacy`: NLP and NER
- `scikit-learn`: Machine learning
- `streamlit`: Web interface
- `transformers`, `torch`: Optional, for LayoutLM/BERT

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please ensure code follows PEP 8 style guidelines.

## Troubleshooting

**Tesseract not found:**
- Ensure Tesseract is installed and in PATH
- On Windows, the code auto-detects common installation paths
- Verify installation: `tesseract --version`

**spaCy model not found:**
- Run: `python -m spacy download en_core_web_sm`
- The code will fall back to regex-based extraction if model is missing

**Model file not found:**
- Train a model first: `python src/train.py`
- Or download a pre-trained model to `models/` directory

## Future Enhancements

- Support for PDF documents
- Multi-language OCR support
- Advanced layout analysis
- Fine-tuned LayoutLM/BERT models
- API endpoint for production use

