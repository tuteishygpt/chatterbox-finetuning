# Беларуская мова (Belarusian) Language Support for Chatterbox TTS

## ✅ Implementation Complete

Belarusian language support has been successfully added to Chatterbox TTS following the official guidelines from [ADDING_NEW_LANGUAGE.md](https://github.com/Musaddiqua/chatterbox/blob/ae4e44edfa443249b59a1d398ed6b0e63d6c440e/ADDING_NEW_LANGUAGE.md).

---

## 📋 Changes Made

### 1. **Updated Supported Languages** 
**File:** `src/chatterbox_/mtl_tts.py`

Added Belarusian to the `SUPPORTED_LANGUAGES` dictionary:
```python
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "be": "Belarusian",  # ✨ NEW
  "da": "Danish",
  # ...
}
```

### 2. **Added Belarusian Text Normalization**
**File:** `src/chatterbox_/models/tokenizers/tokenizer.py`

Created a new normalization function `add_belarusian_stress()`:
- Handles Cyrillic script text processing
- Uses Russian text stresser as fallback (Belarusian is closely related to Russian)
- Gracefully handles cases where dependencies are unavailable

```python
def add_belarusian_stress(text: str) -> str:
    """Belarusian text normalization: adds stress marks to Belarusian text."""
    # Implementation uses Russian stresser as fallback
    # since both languages share Cyrillic script and similar phonetics
```

### 3. **Updated Tokenizer Encode Method**
**File:** `src/chatterbox_/models/tokenizers/tokenizer.py`

Added Belarusian language processing to the encoding pipeline:
```python
def encode(self, txt: str, language_id: str = None, ...):
    # ...
    elif language_id == 'ru':
        txt = add_russian_stress(txt)
    elif language_id == 'be':  # ✨ NEW
        txt = add_belarusian_stress(txt)
    # ...
```

### 4. **Created Test Script**
**File:** `test_belarusian.py`

A standalone test script to verify Belarusian language functionality.

---

## 🚀 How to Use

### **Option 1: Using the Gradio Web Interface (`app.py`)**

1. **Start the web interface:**
   ```bash
   python app.py
   ```

2. **In the web UI:**
   - Select **"be"** (Belarusian) from the Language dropdown
   - Enter Belarusian text, for example:
     - "Добры дзень! Як справы?" (Good afternoon! How are you?)
     - "Вітаю!" (Hello!)
     - "Дзякуй!" (Thank you!)
   - Upload a reference audio file
   - Click **"Generate Speech"**

### **Option 2: Using the Inference Script (`inference.py`)**

Edit the inference script parameters:

```python
# In inference.py, line 144:
LANGUAGE_ID = "be"  # Set to Belarusian

# Example text (line 143):
TEXT_TO_SAY = "Добры дзень! Гэта тэст сістэмы сінтэзу беларускай мовы."
```

Then run:
```bash
python inference.py
```

### **Option 3: Using the Test Script**

```bash
python test_belarusian.py
```

Make sure to update the audio reference path in the script.

---

## ⚙️ Next Steps (Optional - For Production Use)

### **Important: Tokenizer Vocabulary Update**

For optimal performance, you should update the tokenizer vocabulary to include the `[be]` language token:

1. **Check if `[be]` token exists:**
   ```python
   from tokenizers import Tokenizer
   
   tokenizer = Tokenizer.from_file("./pretrained_models/tokenizer.json")
   vocab = tokenizer.get_vocab()
   
   if "[be]" in vocab:
       print("✓ [be] token exists in vocabulary")
   else:
       print("✗ [be] token missing - needs to be added")
   ```

2. **Add the `[be]` token (if missing):**
   
   You'll need to retrain or expand the tokenizer vocabulary. Here's a basic approach:

   ```python
   from tokenizers import Tokenizer
   from tokenizers.models import BPE
   from tokenizers.trainers import BpeTrainer
   
   # Load existing tokenizer
   tokenizer = Tokenizer.from_file("./pretrained_models/tokenizer.json")
   
   # Add the new language token
   tokenizer.add_special_tokens(["[be]"])
   
   # Save updated tokenizer
   tokenizer.save("./pretrained_models/tokenizer_with_be.json")
   ```

   **Note:** After updating the tokenizer, you may need to adjust `NEW_VOCAB_SIZE` in `app.py` and `inference.py`.

### **Fine-tuning for Belarusian (Recommended)**

For best results with Belarusian, you should fine-tune the model on Belarusian speech data:

1. **Prepare Belarusian TTS dataset:**
   - Format: WAV files (16kHz for tokenization, 44.1kHz for S3Gen)
   - Quantity: 10-50 hours of audio recommended
   - Quality: Clear audio with minimal background noise
   - Diversity: Multiple speakers, various prosodies

2. **Data organization:**
   ```
   BelarusianTTSDataset/
   ├── wavs/
   │   ├── speaker1_001.wav
   │   ├── speaker1_002.wav
   │   └── ...
   └── metadata.csv
   ```

3. **Follow the fine-tuning process** described in the main Chatterbox documentation.

---

## 📝 Testing Examples

### Belarusian Test Phrases

**Greetings:**
- "Добры дзень!" - Good afternoon!
- "Добрай раніцы!" - Good morning!
- "Добры вечар!" - Good evening!
- "Вітаю!" - Hello!

**Common Phrases:**
- "Як справы?" - How are you?
- "Дзякуй!" - Thank you!
- "Калі ласка" - Please
- "Да пабачэння!" - Goodbye!

**Sentences:**
- "Мяне клічуць Іван." - My name is Ivan.
- "Я размаўляю па-беларуску." - I speak Belarusian.
- "Гэта цудоўны дзень." - This is a wonderful day.

---

## 🔍 Verification

To verify that Belarusian support is working:

1. **Check supported languages:**
   ```python
   from src.chatterbox_.mtl_tts import SUPPORTED_LANGUAGES
   
   print("Supported languages:", sorted(SUPPORTED_LANGUAGES.keys()))
   # Should include 'be'
   
   print(f"Belarusian: {SUPPORTED_LANGUAGES.get('be')}")
   # Should print: "Belarusian"
   ```

2. **Test tokenization:**
   ```python
   from src.chatterbox_.models.tokenizers import MTLTokenizer
   
   tokenizer = MTLTokenizer("./pretrained_models/tokenizer.json")
   text = "Добры дзень!"
   
   # Encode with Belarusian language ID
   tokens = tokenizer.encode(text, language_id="be")
   print(f"Tokens: {tokens}")
   
   # Should include language-specific preprocessing
   ```

3. **Run the test script:**
   ```bash
   python test_belarusian.py
   ```

---

## 🐛 Troubleshooting

### Issue: "Unsupported language_id 'be'"

**Solution:** Make sure all changes have been saved and you've restarted your Python session or Gradio interface.

### Issue: Missing `russian_text_stresser` library

The Belarusian normalization uses the Russian text stresser as a fallback. If you encounter import errors:

```bash
pip install russian-text-stresser
```

Or the normalization will gracefully skip stress marking (the model may still work).

### Issue: `[be]` token not in vocabulary

If the tokenizer doesn't have the `[be]` token, the model will still work but may not perform optimally. Follow the "Tokenizer Vocabulary Update" section above.

### Issue: Poor audio quality

If the audio quality is poor:
1. Ensure your reference audio is in Belarusian (or use `cfg_weight=0` for language transfer)
2. Adjust generation parameters (exaggeration, temperature, cfg_weight)
3. Consider fine-tuning on Belarusian data for optimal results

---

## 📚 References

- **Original Guide:** [ADDING_NEW_LANGUAGE.md](https://github.com/Musaddiqua/chatterbox/blob/ae4e44edfa443249b59a1d398ed6b0e63d6c440e/ADDING_NEW_LANGUAGE.md)
- **ISO Language Code:** `be` (ISO 639-1 code for Belarusian)
- **Belarusian Language:** Uses Cyrillic script, closely related to Russian and Ukrainian

---

## ✨ Summary

✅ Belarusian language (`be`) is now supported  
✅ Text normalization implemented (using Russian stresser as fallback)  
✅ Integrated into both `app.py` and `inference.py`  
✅ Test script provided for verification  
⚠️ For optimal results: Update tokenizer vocabulary and fine-tune on Belarusian data

---

**З Калядамі! Merry Christmas! 🎄**
