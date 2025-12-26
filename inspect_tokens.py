import torch
from src.chatterbox_.models.tokenizers import MTLTokenizer
import os

def inspect_file(file_path, tokenizer_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    # Load data
    data = torch.load(file_path, map_location='cpu')
    
    print(f"--- Inspection of {os.path.basename(file_path)} ---")
    
    if isinstance(data, dict):
        for key in data.keys():
            val = data[key]
            if torch.is_tensor(val):
                print(f"{key}: Tensor of shape {list(val.shape)}")
            else:
                print(f"{key}: {type(val)}")
        
        # Text tokens
        if "text_tokens" in data:
            text_tokens = data["text_tokens"]
            print(f"\nText Token IDs: {text_tokens.tolist()}")
            
            try:
                tokenizer = MTLTokenizer(tokenizer_path)
                
                # Check for [be] token
                be_id = tokenizer.tokenizer.token_to_id("[be]")
<<<<<<< HEAD
                
=======
>>>>>>> fba7d4db8b6862cc7f921959a1a183c93566f449
                print(f"Token ID for '[be]': {be_id}")
                
                # Check first token
                first_id = text_tokens[0].item()
                first_token = tokenizer.tokenizer.id_to_token(first_id)
                print(f"First ID in file ({first_id}) corresponds to token: {first_token}")

                decoded_text = tokenizer.decode(text_tokens)
                print(f"Decoded Text: {decoded_text}")
            except Exception as e:
                print(f"Could not decode or find tokens: {e}")
                
        # Speech tokens (briefly)
        if "speech_tokens" in data:
            speech_tokens = data["speech_tokens"]
            print(f"Speech Tokens (first 10): {speech_tokens[:10].tolist()} ...")

    else:
        print(f"Data is not a dictionary, it's a {type(data)}")
        print(data)

if __name__ == "__main__":
    file_to_inspect = r"D:\BeTTS\Chatterbox TTS\finetune\MyTTSDataset\preprocess\common_voice_be_28980742.pt"
    tokenizer_json = r"D:\BeTTS\Chatterbox TTS\finetune\tokenizer.json"
    inspect_file(file_to_inspect, tokenizer_json)
