"""
Test script for Belarusian language support in Chatterbox TTS.

This script demonstrates how to use Belarusian (be) language with the fine-tuned model.
"""

import os
from pathlib import Path
import torch
import soundfile as sf
from safetensors.torch import load_file

from src.utils import setup_logger
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.models.tokenizers import MTLTokenizer
from src.chatterbox_.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


logger = setup_logger("BelarusianTest")

# Configuration
BASE_MODEL_DIR = "./pretrained_models"
FINETUNED_WEIGHTS = "./pretrained_models/t3_mtl23ls_v2.safetensors"
NEW_VOCAB_SIZE = 2457


def load_model(device):
    """Load the fine-tuned multilingual model."""
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    # Load Base Engine
    tts_engine = ChatterboxTTS.from_local(BASE_MODEL_DIR, device="cpu")
    
    # Swap tokenizer for Multilingual support
    tokenizer_path = Path(BASE_MODEL_DIR) / "tokenizer.json"
    if tokenizer_path.exists():
        tts_engine.tokenizer = MTLTokenizer(str(tokenizer_path))
    else:
        logger.warning(f"Tokenizer not found at {tokenizer_path}")
    
    # Configure New T3 Model
    logger.info(f"Initializing new T3 with vocab size: {NEW_VOCAB_SIZE}")
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = NEW_VOCAB_SIZE
    new_t3 = T3(hp=t3_config)
    
    # Load Fine-Tuned Weights
    if os.path.exists(FINETUNED_WEIGHTS):
        logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
        state_dict = load_file(FINETUNED_WEIGHTS)
        new_t3.load_state_dict(state_dict, strict=False)
        logger.info("Fine-tuned weights loaded successfully.")
    else:
        logger.warning(f"Fine-tuned file not found at {FINETUNED_WEIGHTS}")

    # Convert to Multilingual Engine
    mtl_engine = ChatterboxMultilingualTTS(
        t3=new_t3,
        s3gen=tts_engine.s3gen,
        ve=tts_engine.ve,
        tokenizer=tts_engine.tokenizer,
        device=device,
        conds=tts_engine.conds
    )
    
    mtl_engine.to(device)
    mtl_engine.t3.eval()
    
    return mtl_engine


def test_belarusian():
    """Test Belarusian language synthesis."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    
    # Check if Belarusian is supported
    logger.info(f"Supported languages: {sorted(SUPPORTED_LANGUAGES.keys())}")
    
    if 'be' not in SUPPORTED_LANGUAGES:
        logger.error("Belarusian (be) is not in supported languages!")
        return
    
    logger.info(f"✓ Belarusian language is supported: {SUPPORTED_LANGUAGES['be']}")
    
    # Load model
    engine = load_model(device)
    
    # Belarusian test text
    belarusian_text = "Добры дзень! Як справы?"  # "Good afternoon! How are you?"
    
    # Reference audio path (adjust this to your reference audio)
    audio_prompt = r"D:\BeTTS\Chatterbox TTS\finetune\speaker_reference\reference.wav"
    
    if not os.path.exists(audio_prompt):
        logger.error(f"Reference audio not found at: {audio_prompt}")
        logger.info("Please provide a valid reference audio file path")
        return
    
    logger.info(f"Synthesizing Belarusian text: {belarusian_text}")
    
    # Generation parameters
    params = {
        "language_id": "be",
        "temperature": 0.8,
        "exaggeration": 1.2,
        "cfg_weight": 0.3,
        "repetition_penalty": 2.0,
    }
    
    # Generate audio
    try:
        wav_tensor = engine.generate(
            text=belarusian_text,
            audio_prompt_path=audio_prompt,
            **params
        )
        
        # Save output
        output_file = "output_belarusian.wav"
        wav_np = wav_tensor.squeeze().cpu().numpy()
        sf.write(output_file, wav_np, engine.sr)
        
        logger.info(f"✓ Audio generated successfully!")
        logger.info(f"✓ Saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_belarusian()
