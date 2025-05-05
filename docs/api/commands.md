# 🚀 Command Detection

The **Command Detection** module provides natural language command parsing for Hey Ada.

---

## 🎯 Supported Command Types

Hey Ada understands a variety of command types:

| 🏷️ Command Type | 📝 Description                | 💬 Example                                 |
|:---------------:|:-----------------------------|:-------------------------------------------|
| `calendar`      | Create calendar events        | _"Utwórz spotkanie jutro o 15:00"_         |
| `reminder`      | Set reminders                 | _"Przypomnij mi o lekarzu o 17:00"_        |
| `music`         | Control music playback        | _"Odtwórz utwór "_                          |

---

## ⚙️ How It Works

The command detection workflow:

1. **Transcription:** User speech is transcribed via the Speech-to-Text API.
2. **Analysis:** The text is analyzed by the LLM and `CommandDetector` class.
3. **Pattern Matching:** If a command pattern is detected, parameters are extracted.
4. **Response:** A structured command object is returned to the client.

---

## 🧩 Command Object Structure

Each recognized command includes:

- **Type:** (e.g., `calendar`, `reminder`, `music`)
- **Parameters:** Details specific to the command type

---