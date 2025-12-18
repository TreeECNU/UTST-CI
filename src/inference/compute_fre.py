#!/usr/bin/env python
# coding=utf-8

import os
import statistics
import textstat
from tqdm import tqdm
import json

# Configuration
OUTPUT_DIR = "outputs_readability_only_ppo_word"
STYLES = [
    {"style_key": "elementary", "text_file": "1/ppo_generated_predictions.txt"},
    {"style_key": "middle", "text_file": "2/ppo_generated_predictions.txt"},
    {"style_key": "high", "text_file": "3/ppo_generated_predictions.txt"},
    {"style_key": "college", "text_file": "4/ppo_generated_predictions.txt"},
]

# Read texts from file
def read_texts(file_path):
    """Read texts from a .txt file, one text per line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    except FileNotFoundError:
        print(f"Text file {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return []

# Compute FRE statistics for a list of texts
def compute_fre_stats(texts, style_key):
    """Compute the average and variance of Flesch Reading Ease scores for given texts."""
    fre_scores = []
    
    for text in tqdm(texts, desc=f"Calculating FRE for {style_key}", total=len(texts), leave=False):
        try:
            score = textstat.flesch_reading_ease(text)
            fre_scores.append(score)
        except Exception as e:
            print(f"Error calculating FRE score for text in {style_key}: '{text[:50]}...': {e}")
    
    if not fre_scores:
        return {"style_key": style_key, "average_fre": 0, "variance_fre": 0, "text_count": 0}
    
    average_fre = statistics.mean(fre_scores)
    variance_fre = statistics.variance(fre_scores) if len(fre_scores) > 1 else 0
    
    return {
        "style_key": style_key,
        "average_fre": average_fre,
        "variance_fre": variance_fre,
        "text_count": len(fre_scores)
    }

def main():
    """Main function to compute FRE statistics for each style's texts."""
    all_results = []
    
    # Process each style
    for style_info in tqdm(STYLES, desc="Processing styles"):
        style_key = style_info["style_key"]
        text_file_path = os.path.join(OUTPUT_DIR, style_info["text_file"])
        
        # Read texts
        texts = read_texts(text_file_path)
        if not texts:
            print(f"No texts loaded for style {style_key} from {text_file_path}")
            all_results.append({
                "style_key": style_key,
                "average_fre": 0,
                "variance_fre": 0,
                "text_count": 0
            })
            continue
        
        print(f"Loaded {len(texts)} texts for style {style_key} from {text_file_path}")
        
        # Compute FRE statistics
        stats = compute_fre_stats(texts, style_key)
        all_results.append(stats)
        
        # Save individual style results
        style_output_dir = os.path.join(OUTPUT_DIR, style_key)
        os.makedirs(style_output_dir, exist_ok=True)
        output_file = os.path.join(style_output_dir, "fre_stats.json")
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Results for {style_key} saved to {output_file}")
    
    # Print summary
    print("\nFlesch Reading Ease Statistics Summary:")
    for result in all_results:
        print(f"\nStyle: {result['style_key']}")
        print(f"  Number of texts: {result['text_count']}")
        print(f"  Average FRE: {result['average_fre']:.2f}")
        print(f"  Variance FRE: {result['variance_fre']:.2f}")

if __name__ == "__main__":
    main()