# 🚀 Language Model API

The Language Model component uses **Ollama** to provide natural language processing capabilities.

---

## 📚 API Endpoints

### 🔹 `POST /llm/generate`

Generate intelligent responses to user queries using a large language model.

#### 📝 Request Body

```json
{
    "prompt": "User query text",
    "system_prompt": "Optional system instructions",
    "temperature": 0.7,
    "max_tokens": 1024
}
```

| Parameter      | Type     | Description                                                      |
| -------------- | -------- | ---------------------------------------------------------------- |
| `prompt`       | string   | The user's query or message                                      |
| `system_prompt`| string   | (Optional) System instructions for the LLM                       |
| `temperature`  | float    | Controls randomness (0.0–1.0, default: 0.7)                      |
| `max_tokens`   | integer  | Maximum tokens to generate in the response (default: 1024)        |

#### 🟢 Example Response

```json
{
    "response": "Generated text response",
    "is_command": false,
    "command_type": null,
    "command_params": null,
    "processing_time": "0.45s"
}
```

---

### 🔹 `GET /llm/info`

Retrieve information about the currently loaded language model.

#### 🟢 Example Response

```json
{
    "model": "gemma3:4b",
    "status": "loaded"
}
```
## 🧩 Command Detection Integration

The Language Model API seamlessly integrates with the **Command Detection** module to identify and process specific user commands within natural language queries.

---

> **Tip:** For best results, provide clear prompts and adjust the `temperature` parameter to fine-tune response creativity.