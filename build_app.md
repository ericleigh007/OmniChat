# OmniChat — Build Document

## Overview
Multimodal voice assistant powered by MiniCPM-o 4.5 (9B parameters, unquantized).
Talk via microphone, hear responses through speakers, switch to cloned voices on command,
and send images/video for the model to interpret. Web UI via Gradio.

## Hardware Requirements
- GPU: NVIDIA with >= 20GB VRAM (tested on RTX PRO 6000 Blackwell 96GB)
- CUDA 12.x with PyTorch 2.x
- Python 3.12+

## Architecture

```
OmniChat/
├── main.py                     # Entry point — loads model, builds Gradio UI (3 tabs)
├── setup.py                    # Idempotent first-run setup
├── launch.bat                  # Windows launcher
├── requirements.txt            # Python dependencies
├── build_app.md                # This file
├── args/
│   └── settings.yaml           # Model config, voice settings, server options
├── voices/                     # Voice reference samples (16kHz WAV)
├── tools/
│   ├── manifest.md             # Index of all tool scripts
│   ├── model/
│   │   └── model_manager.py    # Singleton model loader + chat/image/video inference
│   ├── audio/
│   │   └── voice_manager.py    # Voice sample lookup with fuzzy matching
│   ├── vision/
│   │   └── process_media.py    # Image/document/video analysis + format detection
│   └── output/
│       └── save_output.py      # Save to markdown, text, or Excel
├── outputs/                    # Saved scan/OCR results
├── goals/
│   ├── manifest.md             # Index of goal workflows
│   └── voice_conversation.md   # Voice chat process definition
├── context/
│   └── system_prompt.md        # Default assistant personality
└── .venv/                      # Per-project virtual environment
```

## Build Phases (ATLAS)

### Phase A — Foundation
- Scaffold directory structure
- Create setup.py, launch.bat, requirements.txt
- model_manager.py — singleton model loader
- **Verified**: model loads on GPU, responds to text prompts

### Phase B — Audio Core
- voice_manager.py — voice sample management with fuzzy matching
- Gradio audio chat page (mic → model → speaker)
- Voice command detection via regex patterns
- **Verified**: Gradio UI launches, voice commands parse correctly

### Phase C — Vision Pipeline
- process_media.py — image/document/video analysis with auto format detection
- save_output.py — markdown/text/Excel output saving
- **Verified**: format detection and table parsing work, all save formats functional

### Phase D — Full UI
- Three-tab Gradio interface: Voice Chat, Vision, Settings
- Vision tab: image upload (with webcam), document OCR mode, video upload
- Settings tab: temperature, max tokens, output format, voice management
- **Verified**: all three tabs render, HTTP 200

### Phase E — Documentation
- tools/manifest.md, goals/manifest.md, goals/voice_conversation.md
- This build document

## Key Design Decisions

### Voice Cloning
MiniCPM-o 4.5 supports voice cloning via audio system prompt. A 5-15 second voice
sample is passed as part of the system message, and the model mimics that voice in
its spoken responses. No external TTS system needed.

### Fuzzy Voice Matching
`difflib.get_close_matches()` with configurable threshold (default 0.6) handles
imprecise voice requests like "Morgan" matching "Morgan Freeman".

### Format Auto-Detection
Vision output is classified as 'excel' (pipe/tab tables), 'markdown' (headings,
lists, code blocks), or 'text' (plain) to choose the best save format automatically.

### VideoSource Abstraction (v2)
Designed for future smartphone camera streaming. The vision pipeline accepts any
input that provides frames/audio — adding `WiFiCameraSource` (RTSP/HTTP MJPEG)
later requires only a new source class and a settings entry for the camera URL.

## Configuration (args/settings.yaml)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| model.name | - | openbmb/MiniCPM-o-4_5 | HuggingFace model ID |
| model.dtype | - | bfloat16 | Model precision |
| audio.input_sample_rate | - | 16000 | Input audio sample rate |
| audio.output_sample_rate | - | 24000 | Output audio sample rate |
| audio.default_voice | - | null | Default voice name (null = built-in) |
| voice_commands.enabled | - | true | Enable voice-switch commands |
| voice_commands.fuzzy_threshold | - | 0.6 | Fuzzy matching cutoff |
| inference.temperature | - | 0.7 | Sampling temperature |
| inference.max_new_tokens | - | 2048 | Max response length |
| output.default_format | - | auto | Save format: auto, markdown, text, excel |
| server.host | - | 127.0.0.1 | Gradio server host |
| server.port | - | 7860 | Gradio server port |
| server.share | - | false | Create public Gradio URL |
