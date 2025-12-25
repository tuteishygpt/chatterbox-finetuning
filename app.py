import os
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import random
import re
import gradio as gr
from safetensors.torch import load_file

from src.utils import setup_logger, trim_silence_with_vad
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.models.tokenizers import MTLTokenizer
from src.chatterbox_.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


logger = setup_logger("ChatterboxInference")

# --- Configuration ---
BASE_MODEL_DIR = "./pretrained_models"
FINETUNED_WEIGHTS = "./pretrained_models/t3_mtl23ls_v2.safetensors"
NEW_VOCAB_SIZE = 2457  # Must match the training vocab size

# Global model instance
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_finetuned_engine(device):
    """
    Loads the Chatterbox engine and replaces the T3 module with the fine-tuned version.
    """
    
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    # 1. Load Base Engine (CPU first)
    tts_engine = ChatterboxTTS.from_local(BASE_MODEL_DIR, device="cpu")
    
    # Swap tokenizer for Multilingual support
    tokenizer_path = Path(BASE_MODEL_DIR) / "tokenizer.json"
    if tokenizer_path.exists():
        tts_engine.tokenizer = MTLTokenizer(str(tokenizer_path))
    else:
        logger.warning(f"Tokenizer not found at {tokenizer_path}, keeping default.")
    
    # 2. Configure New T3 Model
    logger.info(f"Initializing new T3 with vocab size: {NEW_VOCAB_SIZE}")
    
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = NEW_VOCAB_SIZE
    
    # Create fresh T3 instance
    new_t3 = T3(hp=t3_config)
    
    # 3. Load Fine-Tuned Weights
    if os.path.exists(FINETUNED_WEIGHTS):
        
        logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
        
        state_dict = load_file(FINETUNED_WEIGHTS)
        
        # Load weights (strict=False enables loading even if some metadata differs)
        try:
            
            new_t3.load_state_dict(state_dict, strict=False)
            logger.info("Fine-tuned weights loaded successfully.")
            
        except RuntimeError as e:
            logger.error(f"Weight mismatch: {e}")
            raise e
        
    else:
        
        logger.warning(f"Fine-tuned file not found at {FINETUNED_WEIGHTS}. Using random init (Garbage output expected).")


    # 4. Convert to Multilingual Engine
    # We reconstruct as ChatterboxMultilingualTTS to get correct punc_norm and generate logic
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


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already."""
    global MODEL
    if MODEL is None:
        logger.info("Model not loaded, initializing...")
        try:
            MODEL = load_finetuned_engine(DEVICE)
            logger.info(f"Model loaded successfully on device: {DEVICE}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return MODEL


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")

    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])

    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)

{line1}

{line2}
"""


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    cfg_weight_input: float = 0.5,
    repetition_penalty_input: float = 2.0,
    seed_num_input: int = 0,
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model.
    
    Args:
        text_input: The text to synthesize into speech
        language_id: The language code for synthesis (e.g., en, fr, de, es, it, pt, hi)
        audio_prompt_path_input: File path to the reference audio file
        exaggeration_input: Controls speech expressiveness (0.25-2.0)
        temperature_input: Controls randomness in generation (0.05-5.0)
        cfg_weight_input: CFG weight controlling generation guidance (0.0-1.0)
        repetition_penalty_input: Penalty for repetition (1.0-3.0)
        seed_num_input: Random seed for reproducible results (0 for random)
        
    Returns:
        tuple[int, np.ndarray]: Sample rate and generated audio waveform
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    logger.info(f"Generating audio for text: '{text_input[:50]}...'")
    logger.info(f"Language: {language_id}")

    if not audio_prompt_path_input:
        raise ValueError("Please provide a reference audio file.")

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfg_weight_input,
        "repetition_penalty": repetition_penalty_input,
        "audio_prompt_path": audio_prompt_path_input,
    }

    logger.info(f"Using audio prompt: {audio_prompt_path_input}")

    wav = current_model.generate(
        text_input,
        language_id=language_id,
        **generate_kwargs
    )
    
    logger.info("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).cpu().numpy())


# --- Gradio Interface ---
with gr.Blocks(title="Chatterbox Multilingual TTS - Fine-tuned") as demo:
    gr.Markdown(
        """
        # 🎙️ Chatterbox Multilingual TTS - Fine-tuned
        Generate high-quality multilingual speech from text with reference audio styling.
        """
    )

    # Display supported languages
    gr.Markdown(get_supported_languages_display())
    
    with gr.Row():
        with gr.Column():
            initial_lang = "en"
            
            text = gr.Textbox(
                value="Hello, this is a test of the Chatterbox text to speech system.",
                label="Text to synthesize",
                max_lines=5,
                placeholder="Enter the text you want to synthesize..."
            )

            language_id = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value=initial_lang,
                label="Language",
                info="Select the language for text-to-speech synthesis"
            )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Required)",
            )

            gr.Markdown(
                "💡 **Note**: Ensure that the reference clip matches the specified language tag. "
                "Otherwise, language transfer outputs may inherit the accent of the reference clip's language. "
                "To mitigate this, set the CFG weight to 0.",
            )

            exaggeration = gr.Slider(
                0.25, 2, step=0.05, 
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", 
                value=1.2
            )
            
            cfg_weight = gr.Slider(
                0.0, 1.0, step=0.05, 
                label="CFG Weight (0 for language transfer)", 
                value=0.3
            )
            
            repetition_penalty = gr.Slider(
                1.0, 3.0, step=0.1,
                label="Repetition Penalty",
                value=2.0
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)

            run_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio")

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            cfg_weight,
            repetition_penalty,
            seed_num,
        ],
        outputs=[audio_output],
    )

    gr.Markdown(
        """
        ---
        ### 📝 Tips:
        - Upload a clear reference audio file (3-10 seconds recommended)
        - Match the reference audio language with the selected language for best results
        - Adjust exaggeration for more expressive speech
        - Use seed for reproducible outputs
        """
    )


if __name__ == "__main__":
    logger.info(f"Starting Gradio interface on device: {DEVICE}")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
