# Quick Start: Using Belarusian Language in Chatterbox TTS

## ✅ Implementation Status

Belarusian language support has been successfully added! Here's what you can do now:

---

## 🚀 Quick Usage Examples

### Example 1: Using the Web Interface (Recommended)

```bash
# Start the Gradio interface
python app.py
```

Then in the web browser:
1. Select **"be"** from the Language dropdown (you'll see "Belarusian" among 24+ languages)
2. Enter Belarusian text:
   ```
   Добры дзень! Як справы? Гэта тэст сістэмы сінтэзу беларускай мовы.
   ```
3. Upload your reference audio
4. Click "🎵 Generate Speech"

### Example 2: Modify inference.py

Open `inference.py` and change these lines:

```python
# Line 143-144
TEXT_TO_SAY = "Добры дзень! Гэта тэст сістэмы сінтэзу мовы."
LANGUAGE_ID = "be"  # Change from "en" to "be"
```

Then run:
```bash
python inference.py
```

### Example 3: Programmatic Usage

```python
from src.chatterbox_.mtl_tts import ChatterboxMultilingualTTS

# Load model
device = "cuda"  # or "cpu"
model = ChatterboxMultilingualTTS.from_local("./pretrained_models", device)

# Generate Belarusian speech
audio = model.generate(
    text="Вітаю! Добры дзень!",
    language_id="be",
    audio_prompt_path="path/to/reference.wav",
    exaggeration=1.2,
    cfg_weight=0.3
)

# Save audio
import soundfile as sf
sf.write("output.wav", audio.squeeze(0).cpu().numpy(), model.sr)
```

---

## 📝 Belarusian Text Examples

### Greetings
```python
texts = [
    "Добры дзень!",        # Good afternoon!
    "Добрай раніцы!",      # Good morning!
    "Добры вечар!",        # Good evening!
    "Вітаю!",              # Hello!
]
```

### Common Phrases
```python
texts = [
    "Як справы?",          # How are you?
    "Дзякуй!",             # Thank you!
    "Калі ласка",          # Please
    "Да пабачэння!",       # Goodbye!
    "Прабачте",            # Excuse me/Sorry
]
```

### Sentences
```python
texts = [
    "Мяне клічуць Іван.",                    # My name is Ivan
    "Я размаўляю па-беларуску.",             # I speak Belarusian
    "Гэта цудоўны дзень.",                   # This is a wonderful day
    "Дзякуй за дапамогу!",                   # Thank you for your help!
    "Беларусь - гэто прыгожая краіна.",      # Belarus is a beautiful country
]
```

---

## 🔧 Verification

Run the verification script to confirm everything is set up correctly:

```bash
python check_belarusian_support.py
```

This will:
- ✅ Check if "be" is in SUPPORTED_LANGUAGES
- ✅ Verify tokenizer configuration
- ✅ Test encoding/decoding Belarusian text
- ⚠️  Warn if [be] token is missing from vocabulary (optional, model still works)

---

## ⚙️ Parameters for Best Results

When generating Belarusian speech, these parameters work well:

```python
params = {
    "language_id": "be",
    "temperature": 0.8,        # Lower = more stable, Higher = more variation
    "exaggeration": 1.2,       # 0.5 = neutral, 1.5-2.0 = very expressive
    "cfg_weight": 0.3,         # 0 = language transfer, 1 = match reference
    "repetition_penalty": 2.0, # Prevents repetition
}
```

**Tips:**
- Use `cfg_weight=0` if your reference audio is NOT in Belarusian (for language transfer)
- Use `cfg_weight=0.3-0.5` if your reference audio IS in Belarusian
- Adjust `exaggeration` for more/less expressive speech
- Lower `temperature` (0.5-0.7) for more consistent output

---

## 📂 Files Modified

✅ `src/chatterbox_/mtl_tts.py` - Added "be" to SUPPORTED_LANGUAGES  
✅ `src/chatterbox_/models/tokenizers/tokenizer.py` - Added Belarusian normalization  
📄 `test_belarusian.py` - Test script  
📄 `check_belarusian_support.py` - Verification script  
📄 `BELARUSIAN_LANGUAGE_SUPPORT.md` - Full documentation  

---

## 🎯 What Works Now

✅ **app.py (Gradio web interface)** - Belarusian appears in language dropdown  
✅ **inference.py** - Can set `LANGUAGE_ID = "be"`  
✅ **Direct API usage** - Can use `language_id="be"` in generate()  
✅ **Text normalization** - Automatic preprocessing for Belarusian text  
✅ **Language validation** - System validates "be" as supported language  

---

## 🔮 Future Enhancements (Optional)

For production-quality Belarusian TTS:

1. **Fine-tune on Belarusian data** (10-50 hours of Belarusian speech)
2. **Update tokenizer vocabulary** to include `[be]` token explicitly
3. **Custom stress markers** for Belarusian (if library becomes available)

---

## 🎄 З Калядамі! (Merry Christmas!)

You can now synthesize speech in Belarusian using Chatterbox TTS!

**Quick test:**
```bash
python test_belarusian.py
```

Or start the web interface and try:
```
Вітаю! З Калядамі і Новым годам!
```
(Hello! Merry Christmas and Happy New Year!)
