"""
Check if the tokenizer vocabulary includes the [be] token for Belarusian.
This script helps verify if the tokenizer needs to be updated.
"""

import sys
from pathlib import Path
from tokenizers import Tokenizer

# Tokenizer path
TOKENIZER_PATH = "./pretrained_models/tokenizer.json"

def check_tokenizer():
    """Check if Belarusian token exists in tokenizer vocabulary."""
    
    if not Path(TOKENIZER_PATH).exists():
        print(f"❌ Tokenizer not found at: {TOKENIZER_PATH}")
        print("Please ensure the pretrained models are downloaded.")
        return False
    
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab = tokenizer.get_vocab()
    
    print(f"\n📊 Tokenizer Statistics:")
    print(f"   Total vocabulary size: {len(vocab)}")
    
    # Check for language tokens
    language_tokens = [key for key in vocab.keys() if key.startswith('[') and key.endswith(']') and len(key) == 4]
    print(f"   Language tokens found: {len(language_tokens)}")
    print(f"   Language tokens: {sorted(language_tokens)}")
    
    # Check specifically for [be]
    print(f"\n🔍 Checking for Belarusian token:")
    if "[be]" in vocab:
        token_id = vocab["[be]"]
        print(f"   ✅ [be] token EXISTS (ID: {token_id})")
        return True
    else:
        print(f"   ⚠️  [be] token NOT FOUND in vocabulary")
        print(f"\n💡 Recommendation:")
        print(f"   The tokenizer doesn't have the [be] token yet.")
        print(f"   The model will still work, but for optimal performance:")
        print(f"   1. Add the [be] token to the tokenizer vocabulary")
        print(f"   2. Consider fine-tuning the model with Belarusian data")
        return False


def test_encoding():
    """Test encoding Belarusian text."""
    from src.chatterbox_.models.tokenizers import MTLTokenizer
    
    print(f"\n🧪 Testing Belarusian text encoding:")
    
    try:
        tokenizer = MTLTokenizer(TOKENIZER_PATH)
        
        # Test text
        belarusian_text = "Добры дзень!"
        
        # Encode without language ID
        tokens_no_lang = tokenizer.encode(belarusian_text, language_id=None)
        print(f"   Text: '{belarusian_text}'")
        print(f"   Without language_id: {len(tokens_no_lang)} tokens")
        
        # Encode with Belarusian language ID
        tokens_with_lang = tokenizer.encode(belarusian_text, language_id="be")
        print(f"   With language_id='be': {len(tokens_with_lang)} tokens")
        
        # Decode to verify
        decoded = tokenizer.decode(tokens_with_lang)
        print(f"   Decoded: '{decoded}'")
        
        print(f"\n   ✅ Encoding test successful!")
        
    except Exception as e:
        print(f"\n   ❌ Error during encoding test: {e}")
        import traceback
        traceback.print_exc()


def show_supported_languages():
    """Display all supported languages."""
    from src.chatterbox_.mtl_tts import SUPPORTED_LANGUAGES
    
    print(f"\n🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total):")
    
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        marker = "✨" if code == "be" else "  "
        print(f"   {marker} {code:3s} - {name}")
    
    if "be" in SUPPORTED_LANGUAGES:
        print(f"\n   ✅ Belarusian (be) is in the supported languages list!")
    else:
        print(f"\n   ❌ Belarusian (be) is NOT in the supported languages list!")


if __name__ == "__main__":
    print("="*60)
    print("Belarusian Language Support - Verification Script")
    print("="*60)
    
    # Check supported languages
    show_supported_languages()
    
    # Check tokenizer
    print("\n" + "="*60)
    has_be_token = check_tokenizer()
    
    # Test encoding
    print("\n" + "="*60)
    test_encoding()
    
    print("\n" + "="*60)
    print("Verification Complete!")
    print("="*60)
