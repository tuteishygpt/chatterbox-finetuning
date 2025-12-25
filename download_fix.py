
import requests
import os
import sys

# URL provided by user (converted to resolve link for direct download)
url = "https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_mtl23ls_v2.safetensors"
dest_dir = r"d:\BeTTS\Chatterbox TTS\finetune\pretrained_models"
dest_file = os.path.join(dest_dir, "t3_mtl23ls_v2.safetensors")

# Ensure directory exists
os.makedirs(dest_dir, exist_ok=True)

print(f"Downloading from: {url}")
print(f"Saving to: {dest_file}")

try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB chunks

    with open(dest_file, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", end="")
                else:
                    print(f"\rProgress: {downloaded / (1024*1024):.1f} MB", end="")
    
    print("\nDownload complete!")

except Exception as e:
    print(f"\nError downloading file: {e}")
    sys.exit(1)
