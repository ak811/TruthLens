import openai
import os
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm

def parse_args():
    """
    Command-line argument parser with default paths.
    """
    parser = argparse.ArgumentParser(description="Image authenticity analysis based on descriptions")
    parser.add_argument("--description_file", required=True, 
                       default="json file containing descriptions", 
                       help="Path to the image descriptions file.")
    parser.add_argument("--output_dir", required=True,
                       default="data path", 
                       help="Path to save the analysis JSON files.")
    parser.add_argument("--api_key", required=True,
                       default="PUT YOUR API KEY", 
                       help="OpenAI API key.")
    
    return parser.parse_args()

def clean_file_path(file_path):
    """
    Extracts just the filename without extension from the full path
    """
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]

def safe_parse_response(response_text):
    """
    Safely parse the response text to extract verdict and justification
    """
    try:
        return ast.literal_eval(response_text)
    except:
        try:
            return json.loads(response_text)
        except:
            response_text = response_text.lower()
            if "fake" in response_text:
                verdict = "FAKE"
            else:
                verdict = "REAL"
            
            justification_parts = response_text.split("justification")
            if len(justification_parts) > 1:
                justification = justification_parts[1].strip(": '\"").strip()
            else:
                justification = response_text
            
            return {
                "verdict": verdict,
                "justification": justification
            }

def analyze_authenticity(description_file, image_files, output_dir):
    """
    Analyzes image descriptions to determine authenticity using OpenAI GPT-3.
    """
    verdicts = []
    
    for file_path in tqdm(image_files, desc="Analyzing images"):
        try:
            key = file_path
            clean_name = clean_file_path(file_path)
            description = description_file[key]

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are a forensic image analyst detecting AI-generated or manipulated images.
                        Analyze the image description and determine if it's real or fake.
                        
                        Key indicators to analyze:
                        1. Lighting and shadows consistency
                        2. Skin texture and natural imperfections
                        3. Facial feature symmetry and proportions
                        4. Reflections and highlights
                        5. Eye and pupil details
                        
                        Provide your analysis in this exact format (ensure it's valid JSON):
                        {
                            "verdict": "FAKE",
                            "justification": "Brief explanation of key factors"
                        }
                        
                        Use "FAKE" or "REAL" for verdict (all caps).
                        Keep justification concise but informative.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this image description and determine if it's real or fake: {description}"
                    }
                ]
            )
            
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = safe_parse_response(response_message)
            
            json_file_path = os.path.join(output_dir, f"{clean_name}_analysis.json")
            with open(json_file_path, "w") as f:
                json.dump(response_dict, f, indent=4)
                
            verdicts.append(response_dict['verdict'])
            
        except Exception as e:
            print(f"\nError processing {clean_name}: {str(e)}")
            print("Waiting 2 minutes before continuing...")
            time.sleep(120)
            continue
    
    return verdicts

def calculate_metrics(all_verdicts, total_images):
    """
    Calculate accuracy metrics for the analysis.
    """
    fake_count = sum(1 for verdict in all_verdicts if verdict == "FAKE")
    real_count = total_images - fake_count
    
    accuracy = (fake_count / total_images) * 100
    
    metrics = {
        "total_images": total_images,
        "detected_fake": fake_count,
        "incorrectly_classified_real": real_count,
        "accuracy": accuracy,
        "error_rate": 100 - accuracy
    }
    
    return metrics

def main():
    """
    Main function to control the flow of the program.
    """
    args = parse_args()
    
    print("Reading description file...")
    with open(args.description_file) as file:
        descriptions = json.load(file)
    
    image_files = list(descriptions.keys())
    total_images = len(image_files)
    
    print(f"Found {total_images} images to analyze")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setting OpenAI API key
    openai.api_key = args.api_key
    
    print("Starting analysis...")
    all_verdicts = analyze_authenticity(descriptions, image_files, args.output_dir)
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(all_verdicts, total_images)
    
    # Saving metrics
    metrics_file = os.path.join(args.output_dir, "analysis_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nAnalysis Metrics:")
    print(f"Total Images Analyzed: {metrics['total_images']}")
    print(f"Images Detected as Fake: {metrics['detected_fake']}")
    print(f"Images Incorrectly Classified as Real: {metrics['incorrectly_classified_real']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Error Rate: {metrics['error_rate']:.2f}%")

if __name__ == "__main__":
    main()