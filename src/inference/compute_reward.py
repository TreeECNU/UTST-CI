#!/usr/bin/env python
# coding=utf-8

import logging
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
TFIDF_LEXICON_DIR = "readability_style_differences"
OUTPUT_DIR = "outputs_readability_sft"
# STYLES = [
#     {"style_key": "elementary", "text_file": "1/generated_predictions.txt"},
#     {"style_key": "middle", "text_file": "2/generated_predictions.txt"},
#     {"style_key": "high", "text_file": "3/generated_predictions.txt"},
#     {"style_key": "college", "text_file": "4/generated_predictions.txt"},
# ]
# STYLES = [
#     {"style_key": "elementary", "text_file": "1/ppo_generated_predictions.txt"},
#     {"style_key": "middle", "text_file": "2/ppo_generated_predictions.txt"},
#     {"style_key": "high", "text_file": "3/ppo_generated_predictions.txt"},
#     {"style_key": "college", "text_file": "4/ppo_generated_predictions.txt"},
# ]
MAX_WORDS = 1000  # Limit to top 100 words per lexicon

# Target Flesch scores for each style (from train_readability_ppo.py)
CATEGORY_RANGES = {
    "elementary": 90,
    "middle": 70,
    "high": 50,
    "college": 20
}

# Gaussian normalization parameters (from train_readability_ppo.py)
SIGMA = 10
GAUSSIAN_CONSTANT = 0.039894228040143274  # 1 / (sqrt(2 * pi) * sigma)

def calc_nd(value, mean):
    """Calculate normalized Gaussian density for Flesch score."""
    return (1 / (SIGMA * np.sqrt(2 * np.pi)) * 
            np.exp(- (value - mean) ** 2 / (2 * SIGMA ** 2)) / GAUSSIAN_CONSTANT)

def get_flesch(text):
    """Calculate Flesch Reading Ease score."""
    try:
        score = textstat.flesch_reading_ease(text)
        return score
    except:
        return 0

# Load TF-IDF lexicons from CSV, limited to top 100 words
def load_tfidf_lexicons(directory):
    """Load TF-IDF lexicons from CSV files, selecting top 100 words by TF-IDF score."""
    lexicons = {}
    for style in [s["style_key"] for s in STYLES]:
        file_path = os.path.join(directory, f"{style}.csv")
        try:
            data = pd.read_csv(file_path)
            if "word" not in data.columns or "tfidf" not in data.columns:
                logger.warning(f"TF-IDF lexicon file {file_path} missing 'word' or 'tfidf' column")
                continue
            # Sort by tfidf in descending order and select top 100 words
            data = data.sort_values(by="tfidf", ascending=False).head(MAX_WORDS)
            lexicons[style] = dict(zip(data["word"].astype(str), data["tfidf"].astype(float)))
            logger.info(f"Loaded {len(lexicons[style])} words for style {style} from {file_path}")
        except FileNotFoundError:
            logger.warning(f"TF-IDF lexicon file {file_path} not found")
        except Exception as e:
            logger.warning(f"Error loading TF-IDF lexicon file {file_path}: {e}")
    return lexicons

# Compute Style Match Score (SMS) and Flesch-based reward
def compute_style_and_reward(texts, style_key, lexicons):
    """Compute Style Match Score (0-1), Flesch score, and combined reward."""
    if not lexicons or style_key not in lexicons:
        logger.warning(f"No TF-IDF lexicon for style {style_key}; returning zero scores")
        return {
            "avg_sms": 0,
            "avg_flesch": 0,
            "avg_reward": 0,
            "scores": [{"sms": 0, "flesch": 0, "reward": 0} for _ in texts]
        }
    
    # Prepare TF-IDF lexicon vector
    lexicon = lexicons[style_key]
    words = list(lexicon.keys())
    lexicon_vector = np.array([lexicon.get(word, 0) for word in words])
    
    # Compute text vectors using lexicon vocabulary
    vectorizer = CountVectorizer(vocabulary=words)
    text_vectors = vectorizer.fit_transform(texts).toarray()
    
    sms_scores = []
    flesch_scores = []
    reward_scores = []
    target_flesch = CATEGORY_RANGES[style_key]
    
    # Progress bar for processing texts
    for text, text_vector in tqdm(zip(texts, text_vectors), 
                                desc=f"Computing scores for {style_key} texts", 
                                total=len(texts), 
                                leave=False):
        # Compute SMS (normalized to 0-1)
        if np.sum(text_vector) == 0 or np.sum(lexicon_vector) == 0:
            sms = 0
        else:
            sms = cosine_similarity([text_vector], [lexicon_vector])[0][0]
        sms_scores.append(sms)
        
        # Compute Flesch score
        flesch_raw = get_flesch(text)
        flesch_normalized = calc_nd(flesch_raw, target_flesch)
        flesch_scores.append(flesch_normalized)
        
        # Compute combined reward
        reward = 0.5 * sms + 0.5 * flesch_normalized
        reward_scores.append(reward)
    
    return {
        "avg_sms": round(np.mean(sms_scores), 4) if sms_scores else 0,
        "avg_flesch": round(np.mean(flesch_scores), 4) if flesch_scores else 0,
        "avg_reward": round(np.mean(reward_scores), 4) if reward_scores else 0,
        "scores": [
            {
                "sms": round(sms, 4),
                "flesch": round(flesch, 4),
                "reward": round(reward, 4)
            }
            for sms, flesch, reward in zip(sms_scores, flesch_scores, reward_scores)
        ]
    }

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
    """Main function to compute SMS, Flesch, and reward for each style."""
    # Load TF-IDF lexicons
    logger.info("Loading TF-IDF lexicons...")
    lexicons = load_tfidf_lexicons(TFIDF_LEXICON_DIR)
    if not lexicons:
        logger.error("No TF-IDF lexicons loaded; exiting")
        return
    
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
        
        # Compute SMS, Flesch, and reward
        metrics = compute_style_and_reward(texts, style_key, lexicons)
        
        # Save results
        style_output_dir = os.path.join(OUTPUT_DIR, style_key)
        os.makedirs(style_output_dir, exist_ok=True)
        output_metrics_file = os.path.join(style_output_dir, "new_metrics.json")
        
        with open(output_metrics_file, "w") as writer:
            json.dump(metrics, writer, indent=4)
        
        logger.info(f"Metrics for style {style_key}: {metrics}")
        logger.info(f"Results saved to {output_metrics_file}")

if __name__ == "__main__":
    main()