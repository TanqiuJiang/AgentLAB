#!/usr/bin/env python3
"""
Convert WebShop product files to Lucene-indexable JSON format.

This script reads the items_shuffle.json file and converts each product
to the JSON format expected by Anserini/Lucene for indexing.

Place this file in: search_engine/convert_product_file_format.py
Run from search_engine directory: python convert_product_file_format.py
"""

import json
import os
import random
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'data'

# Input files to try (in order of preference)
PRODUCT_FILES = [
    'items_shuffle.json',        # Full dataset
    'items_shuffle_1000.json',   # Small dataset
]

ATTR_FILES = [
    'items_ins_v2.json',         # Full attributes
    'items_ins_v2_1000.json',    # Small attributes
]


def load_products(data_dir):
    """Load product data from JSON file."""
    for filename in PRODUCT_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"Loading products from {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                products = json.load(f)
            print(f"  Loaded {len(products)} products")
            return products, filename
    
    raise FileNotFoundError(f"No product file found in {data_dir}. "
                           f"Expected one of: {PRODUCT_FILES}")


def load_attributes(data_dir):
    """Load product attributes from JSON file."""
    for filename in ATTR_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"Loading attributes from {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                attrs = json.load(f)
            print(f"  Loaded attributes for {len(attrs)} products")
            return attrs
    
    print("  Warning: No attribute file found, proceeding without attributes")
    return {}


def product_to_doc(asin, product, attributes=None):
    """
    Convert a single product to Lucene document format.
    
    The format expected by Anserini's JsonCollection:
    {
        "id": "document_id",
        "contents": "searchable text content"
    }
    """
    contents_parts = []
    
    # Handle different product data structures
    if isinstance(product, dict):
        # Product name/title
        name = product.get('name', product.get('Title', product.get('title', '')))
        if name:
            contents_parts.append(name)
            # Add name twice for higher weight
            contents_parts.append(name)
        
        # Description
        desc = product.get('Description', product.get('description', ''))
        if isinstance(desc, list):
            contents_parts.extend([str(d) for d in desc])
        elif desc:
            contents_parts.append(str(desc))
        
        # Features/bullet points
        features = product.get('Features', product.get('features', 
                              product.get('BulletPoints', product.get('bullet_points', []))))
        if isinstance(features, list):
            contents_parts.extend([str(f) for f in features if f])
        elif features:
            contents_parts.append(str(features))
        
        # Category
        category = product.get('category', product.get('Category', ''))
        if category:
            contents_parts.append(str(category))
        
        # Brand
        brand = product.get('brand', product.get('Brand', ''))
        if brand:
            contents_parts.append(str(brand))
        
        # Price (as searchable text)
        price = product.get('price', product.get('Price', ''))
        if price:
            contents_parts.append(f"price {price}")
        
        # Options (colors, sizes, etc.)
        options = product.get('options', product.get('Options', {}))
        if isinstance(options, dict):
            for opt_name, opt_values in options.items():
                if isinstance(opt_values, list):
                    contents_parts.extend([str(v) for v in opt_values])
                else:
                    contents_parts.append(str(opt_values))
        
        # Main image (for completeness, though not searchable)
        # contents_parts might include image URLs but they're not useful for search
        
    elif isinstance(product, str):
        contents_parts.append(product)
    
    # Add attributes if available
    if attributes and asin in attributes:
        attr = attributes[asin]
        if isinstance(attr, dict):
            for key, value in attr.items():
                if isinstance(value, list):
                    contents_parts.extend([str(v) for v in value])
                elif value:
                    contents_parts.append(str(value))
        elif isinstance(attr, list):
            contents_parts.extend([str(a) for a in attr])
    
    # Clean and join content
    contents = ' '.join(str(p).strip() for p in contents_parts if p)
    # Remove excessive whitespace
    contents = ' '.join(contents.split())
    
    return {
        "id": asin,
        "contents": contents
    }


def write_docs(docs, output_dir, batch_size=10000):
    """
    Write documents to JSONL file(s) for Lucene indexing.
    
    Anserini expects JSONL format (one JSON object per line).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'docs.json'
    
    print(f"Writing {len(docs)} documents to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"  Done!")
    return len(docs)


def convert_products(products, attributes, output_dir, limit=None):
    """Convert products to Lucene format and write to output directory."""
    product_list = list(products.items())
    
    if limit:
        # Random sample for subset
        if limit < len(product_list):
            random.seed(42)  # Reproducible sampling
            product_list = random.sample(product_list, limit)
        print(f"  Using {len(product_list)} products (limit: {limit})")
    
    docs = []
    for asin, product in product_list:
        doc = product_to_doc(asin, product, attributes)
        if doc['contents']:  # Only add if there's actual content
            docs.append(doc)
    
    if not docs:
        print(f"  Warning: No documents generated for {output_dir}")
        return 0
    
    return write_docs(docs, output_dir)


def main():
    print("=" * 60)
    print("WebShop Product Converter")
    print("=" * 60)
    
    # Load product data
    try:
        products, source_file = load_products(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease download the data files first:")
        print("  Option 1: Run ./setup.sh -d small (for 1000 products)")
        print("  Option 2: Run ./setup.sh -d all (for full dataset)")
        print("  Option 3: Manually download from Google Drive:")
        print("    - items_shuffle_1000.json: https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib")
        print("    - items_ins_v2_1000.json: https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu")
        return 1
    
    # Load attributes
    attributes = load_attributes(DATA_DIR)
    
    # Define output configurations
    # (output_dir, limit)
    configs = [
        ('resources', None),           # Full dataset
        ('resources_1k', 1000),         # 1K subset
        ('resources_100', 100),         # 100 subset
        ('resources_100k', 100000),     # 100K subset
    ]
    
    total_products = len(products)
    print(f"\nTotal products available: {total_products}")
    print("\nConverting products to Lucene format...")
    print("-" * 60)
    
    for output_dir, limit in configs:
        output_path = BASE_DIR / output_dir
        
        # Skip if limit exceeds available products
        if limit and limit > total_products:
            print(f"\n{output_dir}: Skipping (limit {limit} > available {total_products})")
            continue
        
        print(f"\n{output_dir}:")
        count = convert_products(products, attributes, output_path, limit)
        print(f"  Created {count} documents")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("\nNext steps:")
    print("  1. Verify files: ls -la resources*/")
    print("  2. Run indexing: ./run_indexing.sh")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
