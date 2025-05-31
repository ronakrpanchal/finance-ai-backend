import cv2
import pytesseract
import json
import base64
import numpy as np
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
from datetime import datetime
from pymongo import MongoClient

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")  # Default to a strong model
MONGO_URI = os.environ.get("MONGO_URI")

# 📦 Define Product & Receipt data structures
class Product(BaseModel):
    name: str
    price: float

class ParsedReceiptData(BaseModel):
    products: List[Product]
    store_name: Optional[str] = Field(default=None)
    date: Optional[str] = Field(default=None)
    total: Optional[float] = Field(default=None)

# 🔗 Enhanced LangChain LLM pipeline with better prompt
def load_model():
    llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0,
        api_key=API_KEY
    )

    parser = JsonOutputParser(pydantic_object=ParsedReceiptData)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert receipt parser. Extract information from the receipt text into a JSON format.

Follow these specific guidelines:
1. For each product in the receipt, identify its exact name and price.
2. Ignore subtotals, taxes, and other non-product entries.
3. If there are partial words or unclear text, use your best judgment to infer the complete product name.
4. Make sure all prices are numeric values (not text).
5. If possible, also extract the store name, receipt date, and total amount.

Format the output as:
```json
{
    "products": [
        {"name": "Product Name 1", "price": 10.99},
        {"name": "Product Name 2", "price": 5.49}
    ],
    "store_name": "Store Name (if available)",
    "date": "YYYY-MM-DD (if available)",
    "total": 16.48 (if available)
}
```

The most important task is to correctly identify product names and their corresponding prices.
If some text is unclear, focus on the parts that are clearly product entries.
"""),
        ("user", "Here is the receipt text to parse:\n\n{input}")
    ])

    return prompt | llm | parser

# 🖼️ Enhanced image preprocessing with multiple approaches
def preprocess_receipt(image):
    """Apply multiple preprocessing techniques to improve OCR results"""
    # Convert to grayscale if image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create multiple processed versions with different techniques
    processed_images = []
    
    # Version 1: Basic adaptive thresholding
    adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    processed_images.append(adaptive1)
    
    # Version 2: Different parameters for adaptive thresholding
    adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 5)
    processed_images.append(adaptive2)
    
    # Version 3: Otsu's thresholding after Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # Version 4: Increase contrast using histogram equalization
    equalized = cv2.equalizeHist(gray)
    processed_images.append(equalized)
    
    # Version 5: Bilateral filter to preserve edges but remove noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    processed_images.append(bilateral)
    
    return processed_images

# 🧾 Improved OCR text extraction that tries multiple preprocessing approaches
def extract_text_from_receipt(image):
    """Extract text using multiple preprocessing techniques and combine results"""
    processed_images = preprocess_receipt(image)
    
    # Extract text from each processed image
    all_texts = []
    
    # Try with default image
    default_text = pytesseract.image_to_string(image)
    if default_text.strip():
        all_texts.append(default_text)
    
    # Try each processed version
    for processed_img in processed_images:
        try:
            # Try different OCR configurations
            # -l eng: English language
            # --oem 1: Use LSTM OCR Engine
            # --psm 4: Assume a single column of text of variable sizes
            text1 = pytesseract.image_to_string(processed_img, config='--oem 1 --psm 4')
            if text1.strip():
                all_texts.append(text1)
            
            # Try with different PSM mode for receipts
            # --psm 6: Assume a single uniform block of text
            text2 = pytesseract.image_to_string(processed_img, config='--oem 1 --psm 6')
            if text2.strip():
                all_texts.append(text2)
        except:
            continue
    
    # Find the text with the most useful content (most likely to contain prices)
    best_text = ""
    max_price_count = 0
    
    price_pattern = r'\$?\d+\.\d{2}'  # Pattern to match prices like 10.99 or $10.99
    
    for text in all_texts:
        # Count potential price matches
        price_matches = re.findall(price_pattern, text)
        if len(price_matches) > max_price_count:
            max_price_count = len(price_matches)
            best_text = text
    
    # If we couldn't find a good candidate, use the longest text
    if not best_text and all_texts:
        best_text = max(all_texts, key=len)
    
    # Clean up the text
    best_text = best_text.replace('\n\n', '\n').strip()
    
    return best_text

# 🔍 Enhanced LLM query with fallback strategy
def query_llm(extracted_text):
    """Query the LLM with fallback mechanism for better results"""
    try:
        chain = load_model()
        result = chain.invoke({"input": extracted_text})
        
        # Validate result
        if not result or not result.get('products'):
            # Try to extract at least some basic product information using regex
            fallback_products = extract_products_with_regex(extracted_text)
            if fallback_products:
                return {"products": fallback_products}
        
        return result
    except Exception as e:
        print(f"LLM error: {str(e)}")
        # Fallback to regex extraction
        fallback_products = extract_products_with_regex(extracted_text)
        return {"products": fallback_products}

# 📋 Regex-based fallback product extraction
def extract_products_with_regex(text):
    """Extract products and prices using regex patterns as a fallback method"""
    products = []
    
    # Common patterns in receipts
    # Look for lines with a price at the end
    line_pattern = r'(.*?)\s+(\$?\d+\.\d{2})\s*$'
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to match the pattern
        match = re.search(line_pattern, line)
        if match:
            name = match.group(1).strip()
            price_str = match.group(2).strip()
            
            # Clean up price string
            price_str = price_str.replace('$', '')
            try:
                price = float(price_str)
                
                # Skip likely non-products (very high prices, totals, etc.)
                if price > 0 and price < 1000 and not any(x in name.lower() for x in ['total', 'tax', 'subtotal', 'sum', 'amount']):
                    products.append({"name": name, "price": price})
            except:
                pass
    
    return products

# 💾 Save JSON response to file
def save_json_to_file(data, filename="llm_response.json"):
    """Save structured data to a JSON file"""
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# 🖼️ Image loading functions
def load_image_from_path(image_path):
    """Load image from a file path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return img

def decode_base64_image(base64_string):
    """Decode base64 string to OpenCV image"""
    try:
        # If the string contains data URI scheme, extract the base64 part
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image from base64 string")
        
        return img
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {str(e)}")

# 🧠 Main pipeline with enhanced error handling
def receipt_model(image_data):
    """
    Process receipt image and extract structured information
    
    Args:
        image_data (str): Can be either a file path or base64 encoded image string
        
    Returns:
        dict: Extracted receipt information
    """
    try:
        # Check if image_data is a base64 string or a file path
        if isinstance(image_data, str) and (image_data.startswith('data:image') or ';base64,' in image_data):
            # It's a base64 string
            image = decode_base64_image(image_data)
        else:
            # Assume it's a file path
            image = load_image_from_path(image_data)
        
        # Print image dimensions for debugging
        print(f"Image dimensions: {image.shape}")
        
        # Extract text using OCR
        extracted_text = extract_text_from_receipt(image)
        
        # Print extracted text for debugging
        print(f"Extracted text length: {len(extracted_text)}")
        print("Sample text (first 200 chars):", extracted_text[:200])
        
        # Process with LLM if we have text
        if extracted_text.strip():
            llm_response = query_llm(extracted_text)
        else:
            print("Warning: No text extracted from image")
            llm_response = {"products": []}
        
        # Save response to file (optional)
        save_json_to_file(llm_response)
        
        return llm_response
        
    except Exception as e:
        print(f"Error in receipt_model: {str(e)}")
        # Return a minimal valid structure in case of error
        return {"products": []}

# 🗃️ Save to MongoDB
def save_receipt_in_mongodb(user_id, llm_response, date, category):
    """Save receipt data to MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client["finance_ai"]
        collection = db["transactions"]
        
        # Handle both dictionary and object formats
        if isinstance(llm_response, dict):
            products = llm_response.get('products', [])
        else:
            # Assuming it's a Pydantic model or similar
            products = llm_response.products
        
        documents = []
        for item in products:
            # Handle both dictionary and object formats
            if isinstance(item, dict):
                name = item.get('name', 'Unknown Item')
                price = float(item.get('price', 0.0))
            else:
                name = item.name
                price = float(item.price)
                
            doc = {
                "user_id": user_id,
                "transaction_date": date,
                "amount": price,
                "amount_type": "debit",
                "category": category,
                "description": name
            }
            documents.append(doc)

        # Only insert if there are documents to insert
        if documents:
            collection.insert_many(documents)
            print(f"✅ Successfully added {len(documents)} items to MongoDB")
        else:
            print("⚠️ No products found to save")
            
        client.close()
        return True
    except Exception as e:
        print(f"MongoDB Error: {str(e)}")
        return False