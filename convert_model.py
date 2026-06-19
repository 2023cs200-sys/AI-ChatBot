import json
import torch

def convert():
    print("Loading data.pth...")
    data = torch.load("data.pth", map_location="cpu")
    
    # Extract properties
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]
    
    # Convert weights & biases (tensors) to nested lists
    converted_state = {}
    for key, tensor in model_state.items():
        converted_state[key] = tensor.tolist()
        print(f"Converted {key} with shape {list(tensor.shape)}")
        
    json_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
        "model_state": converted_state
    }
    
    print("Saving to data.json...")
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
        
    print("Successfully converted data.pth to data.json!")

if __name__ == "__main__":
    convert()
