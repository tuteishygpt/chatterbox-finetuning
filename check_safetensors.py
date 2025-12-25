
import struct
import os
import json

file_path = r"d:\BeTTS\Chatterbox TTS\finetune\pretrained_models\t3_mtl23ls_v2.safetensors"

try:
    file_size = os.path.getsize(file_path)
    print(f"Actual File size: {file_size} bytes")

    with open(file_path, "rb") as f:
        # Read the header size (first 8 bytes)
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        print(f"Header Length: {header_len}")
        
        # Read the header
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        
        # Calculate expected size from metadata
        max_end_offset = 0
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if "data_offsets" in value:
                start, end = value["data_offsets"]
                if end > max_end_offset:
                    max_end_offset = end
        
        expected_size = 8 + header_len + max_end_offset
        print(f"Expected File Size (calculated from header): {expected_size} bytes")
        
        if file_size != expected_size:
            print("-" * 30)
            print(f"CRITICAL ERROR: File is truncated or corrupted!")
            print(f"Missing bytes: {expected_size - file_size}")
            print("-" * 30)
        else:
            print("File size matches expected size. The file structure appears correct.")

except Exception as e:
    print(f"Error checking file: {e}")
