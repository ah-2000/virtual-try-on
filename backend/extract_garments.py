import os
import sys
import argparse
from PIL import Image
import numpy as np
import torch

# Setup paths to find fashnvton packages
project_root = os.path.dirname(os.path.abspath(__file__))
fashn_src_path = os.path.join(project_root, "fashnvton", "src")
sys.path.append(fashn_src_path)

from fashn_human_parser import FashnHumanParser, CATEGORY_TO_BODY_COVERAGE
from fashn_vton.preprocessing import BODY_COVERAGE_TO_FASHN_LABELS, FASHN_LABELS_TO_IDS

def extract_garments(input_dir, output_dir, category):
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FashnHumanParser on {device}...")
    hp_model = FashnHumanParser(device=device)
    
    body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
    labels_to_segment = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
    labels_to_segment_indices = [FASHN_LABELS_TO_IDS[label] for label in labels_to_segment]
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    print(f"Extracting {category} from images in {input_dir}...")
    
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue
            
        img_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, f"extracted_{os.path.splitext(filename)[0]}.png")
        
        try:
            print(f"Processing: {filename}")
            # Load and ensure RGB
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            
            # Predict segmentation
            seg_pred = hp_model.predict(img_np)
            
            # Create mask of the garment
            mask = np.isin(seg_pred, labels_to_segment_indices)
            
            # Create RGBA image with transparent background (or white if prefered)
            # Fashn-VTON works well with white backgrounds for garments, but transparent is also good. 
            # We will use a white background to mimic flat-lay photos.
            result_np = np.ones_like(img_np) * 255  # White background
            result_np[mask] = img_np[mask]
            
            result_img = Image.fromarray(result_np)
            result_img.save(out_path)
            print(f"  -> Saved to {out_path}")
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract garments from model photos")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing model photos")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for extracted garment images")
    parser.add_argument("--category", "-c", type=str, choices=["tops", "bottoms", "one-pieces"], default="one-pieces", help="Garment category to extract")
    
    args = parser.parse_args()
    extract_garments(args.input, args.output, args.category)
