import gdown
import os

files = {
    "common_voice_be_28980742.wav": "1Z32qMj3s93CQk1a24ZXIQLvhLC8MFL1o",
    "common_voice_be_28980744.wav": "1lSIRp0OVbtE6BpDoaS0rQY8Hx1b03ndh",
    "common_voice_be_28980762.wav": "1AWPQrttjkQmLoYx6mzDo5VvqcaZq19kS",
    "common_voice_be_28980766.wav": "1--X1juStaVPj8NUzTH2ZRAEOVRBNT4wX",
    "common_voice_be_28980769.wav": "1XTLkdOYNl655u0Z_GdtBNJaGC7BomzCR",
    "common_voice_be_28980789.wav": "18qsfNFuBSPS4fUcsjj7IpDF8L3AZxrON",
    "common_voice_be_28980790.wav": "1fScxWsoRF5I9E-CxxaxN8xzyD98QNnI5",
    "common_voice_be_28980815.wav": "1tESLIOX2QDWc6_2ZNZOmJLdufVXUiaTy",
    "common_voice_be_28980818.wav": "1PZd0MyBMy7WZFE1ix29-neaPNaA-6LpB",
    "common_voice_be_28980819.wav": "16J8HKQ4AOUtUOJok65F7j_3O2s19oIXh",
    "common_voice_be_28980830.wav": "1aOyyDQN5BiRnz4Waf0oc9qRfpvT5h5RE",
    "common_voice_be_28980834.wav": "1LHRKOJb1zMVVNIXMV8yzgf5L2lNe4wIw",
    "common_voice_be_28980838.wav": "1Uw2i2uArGyG3toVfeaXmBTTG6viTHErx",
    "common_voice_be_28980860.wav": "1Vcsl-IN7d8dfmC8qSbl1CmtNdlGkqXrj",
    "common_voice_be_28980862.wav": "1qRS8Xz1zj5HCS9irHkZB0AWHbL7DBfmo",
    "common_voice_be_28980890.wav": "1BktCqD5HvXJyxS2AODG8fJQeONYmBkQW",
    "common_voice_be_28980894.wav": "1A6n9xaGaAD9X6GvVd703dsWe1hVD3l4x",
    "common_voice_be_28980895.wav": "1zYcnNa1cM9vasGXu18NFItHzkzNT5_BS",
    "common_voice_be_28980907.wav": "12XsAOPlU6B8XiTBF-81tU9Yxmw5XmUTD"
}

output_dir = r"D:\BeTTS\Chatterbox TTS\finetune\MyTTSDataset\wavs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename, file_id in files.items():
    output_path = os.path.join(output_dir, filename)
    print(f"Downloading {filename}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
