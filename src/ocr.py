"""
OCR Module for Document Intelligence System
Extracts text from images using Tesseract OCR
"""

import os
import sys
import pytesseract
from PIL import Image
import cv2
import numpy as np


# Auto-detect Tesseract path for Windows
if sys.platform == 'win32':
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


def preprocess_image(image_path):
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised


def extract_text(image_path, preprocess=True):
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_path: Path to image file
        preprocess: Whether to preprocess image before OCR
        
    Returns:
        Extracted text as string
    """
    try:
        if preprocess:
            img = preprocess_image(image_path)
            pil_image = Image.fromarray(img)
        else:
            pil_image = Image.open(image_path)
        
        # Extract text with Tesseract
        text = pytesseract.image_to_string(pil_image, lang='eng')
        
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""


def extract_text_with_boxes(image_path):
    """
    Extract text with bounding box information.
    
    Args:
        image_path: Path to image file
        
    Returns:
        List of dictionaries with 'text', 'left', 'top', 'width', 'height'
    """
    try:
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Confidence > 0
                boxes.append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })
        
        return boxes
    except Exception as e:
        print(f"Error extracting text boxes from {image_path}: {e}")
        return []


if __name__ == "__main__":
    # Test OCR on a sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        text = extract_text(image_path)
        print("Extracted text:")
        print(text)

