import json

def concatenate_json_descriptions(json_file_paths):
    """
    Concatenate text descriptions from multiple JSON files for the same image paths.
    
    Args:
        json_file_paths (list): List of paths to JSON files
    
    Returns:
        dict: Combined dictionary with concatenated descriptions for each image
    """
    combined_data = {}
    
    for file_path in json_file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for img_path, description in data.items():
                    if img_path in combined_data:
                        combined_data[img_path] = combined_data[img_path] + " | " + description
                    else:
                        combined_data[img_path] = description
                        
            print(f"Successfully processed: {file_path}")
            
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {file_path}: {e}")
    
    output_path = 'combined_descriptions.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4)
        print(f"\nCombined data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving combined data: {e}")
    
    return combined_data

# Example usage
if __name__ == "__main__":
    json_paths = [
        "eyes_chat_univi_results_fake1000.json",
        "faceattributes_chat_univi_results_fake1000.json",
        "facialhair_chat_univi_results_fake1000.json",
        "realism_2_chat_univi_results_fake1000.json",
        "reflections_chat_univi_results_fake1000.json",
        "symmetry_2_chat_univi_results_fake1000.json",
        "texture_chat_univi_results_fake1000.json"
    ]
    
    combined_result = concatenate_json_descriptions(json_paths)
    print(f"\nTotal number of unique images: {len(combined_result)}")
    
    if combined_result:
        example_key = next(iter(combined_result))
        print(f"\nExample combined description for {example_key}:")
        print(combined_result[example_key])