# Goal: Voice Conversation

## Objective
Handle a single conversational turn: receive user input (audio or text), generate a response with optional voice cloning, and return both text and spoken audio.

## Trigger
User speaks into the microphone (stop_recording event) or submits text via the text box.

## Inputs
- **Audio input**: Gradio `(sample_rate, numpy_array)` tuple from microphone, OR
- **Text input**: String from the text box
- **Current voice ref**: Optional numpy array of 16kHz mono voice reference audio
- **Session settings**: temperature, max_new_tokens from the Settings tab

## Process

### 1. Normalize input
- Convert audio to float32 mono 16kHz (resample with librosa if needed)
- Text input passes through directly

### 2. Check for voice commands
- Run `detect_voice_command(text)` against regex patterns
- Supported commands: "change voice to X", "switch to X's voice", "use X voice", "sound like X", "speak like X", "use default voice"
- If command detected:
  - Look up voice via `voice_manager.get_voice(name, fuzzy_threshold)`
  - If exact match: switch voice, confirm
  - If fuzzy match: switch voice, inform user of the match
  - If no match: report available voices
  - If "default": clear voice ref

### 3. Generate response
- Build message list: `[{"role": "user", "content": [input]}]`
- If voice ref is set, model_manager prepends a system message with the reference audio
- Call `model_manager.chat()` with session temperature and max_new_tokens
- Audio response is saved to `.tmp/response.wav`

### 4. Return results
- Text transcript appended to chat history
- Audio tuple `(sample_rate, numpy_array)` returned for Gradio autoplay
- Status bar updated with current voice name

## Tools Used
- `tools/model/model_manager.py` — `chat()`
- `tools/audio/voice_manager.py` — `get_voice()`, `list_voices()`

## Edge Cases
- No audio received → return "No audio received" status
- Empty text → return "No text entered" status
- Voice command with no matching voice → list available voices
- Fuzzy voice match → inform user which voice was selected
- Model generates text but no audio file → return text only (no playback)
