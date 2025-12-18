#!/usr/bin/env python
# coding=utf-8

import logging
import os
import json
from collections import Counter
from tqdm import tqdm
import textstat

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
)

# Configuration
OUTPUT_DIR = "outputs_readability_sft"
STYLES = [
    {"style_key": "elementary", "text_file": "1/generated_predictions.txt"},
    {"style_key": "middle", "text_file": "2/generated_predictions.txt"},
    {"style_key": "high", "text_file": "3/generated_predictions.txt"},
    {"style_key": "college", "text_file": "4/generated_predictions.txt"},
]

# Classify text based on Flesch score
def classify_flesch(text):
    """Calculate Flesch Reading Ease score and classify into a category."""
    try:
        flesch_score = textstat.flesch_reading_ease(text)
        if flesch_score >= 80:
            return "elementary"
        elif 60 <= flesch_score < 80:
            return "middle"
        elif 40 <= flesch_score < 60:
            return "high"
        else:
            return "college"
    except Exception as e:
        logger.error(f"Error calculating Flesch score for text: '{text[:50]}...': {e}")
        return "unknown"

# Compute category predictions
def compute_category_predictions(texts, style_key):
    """Compute predicted readability categories and return their counts."""
    predicted_categories = []
    
    # Progress bar for processing texts
    for text in tqdm(texts, desc=f"Classifying texts for {style_key}", total=len(texts), leave=False):
        predicted_category = classify_flesch(text)
        predicted_categories.append(predicted_category)
    
    # Count occurrences of each predicted category
    category_counts = Counter(predicted_categories)
    
    # Convert to a dictionary with all possible categories, ensuring zero counts for missing ones
    all_categories = ["elementary", "middle", "high", "college", "unknown"]
    result = {cat: category_counts.get(cat, 0) for cat in all_categories}
    
    return {"intended_label": style_key, "category_counts": result}

# Read texts from file
def read_texts(file_path):
    """Read texts from a .txt file, one text per line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    except FileNotFoundError:
        logger.error(f"Text file {file_path} not found")
        return []
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return []

def main():
    """Main function to compute category prediction statistics for each style."""
    # Dictionary to store results for all styles
    all_results = {}
    
    # Process each style with progress bar
    for style_info in tqdm(STYLES, desc="Processing styles"):
        style_key = style_info["style_key"]
        text_file = style_info["text_file"]
        text_file_path = os.path.join(OUTPUT_DIR, text_file)
        
        logger.info(f"Processing style: {style_key} (file: {text_file})")
        
        # Read generated texts
        texts = read_texts(text_file_path)
        if not texts:
            logger.warning(f"No texts loaded for style {style_key}; skipping")
            continue
        
        logger.info(f"Loaded {len(texts)} texts for style {style_key}")
        
        # Compute category predictions
        metrics = compute_category_predictions(texts, style_key)
        
        # Store results
        all_results[style_key] = metrics
        
        # Save results
        style_output_dir = os.path.join(OUTPUT_DIR, style_key)
        os.makedirs(style_output_dir, exist_ok=True)
        output_metrics_file = os.path.join(style_output_dir, "category_metrics.json")
        
        with open(output_metrics_file, "w") as writer:
            json.dump(metrics, writer, indent=4)
        
        logger.info(f"Metrics for style {style_key}: {metrics}")
        logger.info(f"Results saved to {output_metrics_file}")
    
    # Print summary of category predictions
    logger.info("\nSummary of Predicted Readability Categories (Counts):")
    for style_key in all_results:
        logger.info(f"\nStyle: {style_key} (Intended Label: {all_results[style_key]['intended_label']})")
        logger.info("Category Counts:")
        for category, count in all_results[style_key]["category_counts"].items():
            logger.info(f"  {category}: {count}")

if __name__ == "__main__":
    main()