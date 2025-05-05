# 🎤 Speech-to-Text API

The Speech-to-Text module uses [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) to transcribe audio files. 

---

## 🚀 API Endpoints

### `POST /stt/transcribe`

Transcribe your audio files effortlessly and receive instant text results.

#### 📥 Request Parameters

| Name        | Type     | Description                                               | Example      |
|-------------|----------|-----------------------------------------------------------|--------------|
| `file`      | File     | Audio file to transcribe (`.wav`, `.mp3`, etc.)           | `audio.wav`  |
| `language`  | String   | Language code (`"auto"` for auto-detect, `"PL"` for Polish, etc.) | `"auto"`     |
| `use_json`  | Boolean  | If `true`, output will be formatted as JSON               | `true`       |

#### 📤 Response

Returns a **Server-Sent Events (SSE)** stream with transcription segments as they become available.

---
