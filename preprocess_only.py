"""
Preprocess dataset without training.
Creates .pt files in the preprocessed directory.
"""
import torch
from src.config import TrainConfig
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.preprocess_file_based import preprocess_dataset_file_based
from src.chatterbox_.tts import ChatterboxTTS
from src.utils import setup_logger

logger = setup_logger("Preprocess")

def main():
    cfg = TrainConfig()
    
    logger.info("--- Starting Preprocessing Only ---")
    logger.info(f"Model Directory: {cfg.model_dir}")
    logger.info(f"Language ID: {cfg.language_id}")
    logger.info(f"WAV Directory: {cfg.wav_dir}")
    logger.info(f"Output Directory: {cfg.preprocessed_dir}")
    
<<<<<<< HEAD
    # Load TTS engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading TTS engine on {device}...")
    tts_engine = ChatterboxTTS.from_local(cfg.model_dir, device=device, load_t3=False)
=======
    # Load TTS engine on CPU first
    logger.info("Loading TTS engine...")
    tts_engine = ChatterboxTTS.from_local(cfg.model_dir, device="cpu")
>>>>>>> fba7d4db8b6862cc7f921959a1a183c93566f449
    
    # Run preprocessing
    if cfg.ljspeech:
        logger.info("Using LJSpeech format (metadata.csv)")
        preprocess_dataset_ljspeech(cfg, tts_engine)
    else:
        logger.info("Using file-based format (wav + txt pairs)")
        preprocess_dataset_file_based(cfg, tts_engine)
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
