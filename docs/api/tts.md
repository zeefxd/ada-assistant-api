# 🗣️ Text-to-Speech API

The Text-to-Speech module uses XTTS-v2 to generate natural-sounding Polish speech.

---

## 🚀 API Endpoints

### **POST** `/tts/synthesize`

Transform your Polish text into lifelike speech using a custom voice sample.

#### 📥 Request Body

```json
{
    "text": "Text to convert to speech",
    "voice_sample": "polish_female_voice.wav",
    "language": "pl"
}
```

| Parameter      | Type   | Description                                                                 |
| -------------- | ------ | --------------------------------------------------------------------------- |
| `text`         | string | Text to convert to speech (Polish only)                                     |
| `voice_sample` | string | Reference voice sample filename (must be in the `assets` directory)         |
| `language`     | string | Language code (default: `"pl"` for Polish)                                  |

#### 📤 Response

- Returns: An audio file (`.wav`) containing the synthesized speech.

---

## ⚙️ Implementation Details

- Loads the **XTTS-v2** model on first use for efficient performance.
- Utilizes your provided voice sample for **voice cloning**.
- Synthesizes speech with the cloned voice profile for a personalized experience.
- Delivers the result as a downloadable audio file.

---

> **Tip:** For best results, use a clear, high-quality `.wav` voice sample in Polish.