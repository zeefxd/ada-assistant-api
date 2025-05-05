## 🚀 Required Models

The application **automatically downloads** the following:

- **Whisper.cpp `large-v3-turbo`** model for speech recognition  
- **FFmpeg** (if not already installed)

---
## 🛠️ Manual Dependency Setup

If the automatic setup fails, you can manually install required dependencies:

### Whisper.cpp

1. **Clone the Repository:**  
    ```sh
    git clone https://github.com/ggerganov/whisper.cpp.git /tmp/whisper.cpp-src
    cd /tmp/whisper.cpp-src
    ```

2. **Build the Executable:**  
    ```sh
    make
    ```
    > 💡 *For GPU acceleration (CUDA, OpenCL, etc.), see the [official build guide](https://github.com/ggerganov/whisper.cpp#build).*

3. **Copy the Executable:**  
    - Create a `whisper.cpp` directory in your project root.
    - Copy the built `whisper-cli.exe` into `whisper.cpp/`.

    ```sh
    mkdir -p /your/project/path/whisper.cpp
    cp ./whisper-cli.exe /your/project/path/whisper.cpp/
    ```

4. **Verify:**  
    Ensure `whisper.cpp/whisper-cli.exe` exists in your project directory.
---

### FFmpeg

If FFmpeg isn't installed automatically:

1. **Download FFmpeg** from [ffmpeg.org](https://ffmpeg.org/download.html).
2. **Extract to:**  
    ```
    model/ffmpeg/
    ```
3. **Ensure:**  
    ```
    model/ffmpeg/bin/ffmpeg.exe
    ```
    exists.

---

## 🗣️ Speech Recognition (STT) Configuration

You can change the Whisper model in `stt.py`:

```python
# Change this to use a different model
WHISPER_CPP_MODEL_NAME = "large-v3-turbo-q8-v3"
```

**Available models:**

- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v1`, `large-v2`, `large-v3`

---

## 🗨️ Text-to-Speech (TTS) Configuration

To use TTS with a Polish voice:

1. **Record or download** a Polish female voice sample.
2. **Save as WAV** in the `assets` directory.
3. **Reference the file** in your API calls.

---

## 🕹️ Command Detection

The system recognizes commands like:

- Calendar events
- Reminders
- Music controls

**Customize** these by editing `command_detector.py`.

---