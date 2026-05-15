# OmniChat — Build Document

## Overview
Multimodal voice assistant powered by selectable local and remote model profiles. Supported local paths now include MiniCPM-o 4.5, Gemma 4 E4B IT, Gemma 4 E4B IT with the MTP assistant drafter, and llama.cpp-backed Qwen3.5 profiles, plus Qwen3-Omni local/remote profiles.
Talk via microphone, hear responses through speakers, switch to cloned voices on command,
and send images/video for the model to interpret. Web UI via Gradio and native RT UI via PySide6.

## Hardware Requirements
- GPU: NVIDIA with >= 20GB VRAM (tested on RTX PRO 6000 Blackwell 96GB [cu120])
- CUDA 12.x with PyTorch 2.x
- Python 3.12+

## Architecture

```
OmniChat/
├── main.py                     # Entry point — loads model, builds Gradio UI
├── rt_main.py                  # PySide6 desktop entry point
├── rt_app.py                   # PySide6 desktop window and UI logic
├── rt_audio.py                 # Desktop mic/VAD/model/speaker pipeline
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
│   │   ├── model_manager.py    # Singleton model loader + profile runtime config
│   │   └── backends/           # MiniCPM, Qwen, Gemma, llama.cpp backend adapters
│   ├── audio/
│   │   └── voice_manager.py    # Voice sample lookup with fuzzy matching
│   ├── vision/
│   │   └── process_media.py    # Image/document/video analysis + format detection
│   └── output/
│       └── save_output.py      # Save to markdown, text, or Excel
├── outputs/                    # Saved scan/OCR results
├── benchmarks/                 # Quantization and Gemma MTP speed/quality benchmarks
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

### Phase F — Multi-Backend Runtime
- Profile-driven runtime selection through `args/model_profiles.json`
- Shared `tools/shared/session.py` model runtime configurator for Gradio, RT, demos, probes, and benchmarks
- Gemma 4 E4B IT local Transformers backend with native text/image/audio/video input
- MiniCPM streaming TTS bridge for models that need spoken output
- **Verified**: baseline Gemma, Gemma+MiniCPM TTS, Gemma+MTP, and Gemma+MTP+MiniCPM TTS profiles load and run

### Phase G — Gemma MTP Acceleration
- Added optional Gemma MTP assistant model loading via `google/gemma-4-E4B-it-assistant`
- Added profile-level switching between old Gemma and MTP Gemma
- Runtime status reports `assistant_loaded` and `last_mtp_used`
- **Verified**: 50-case multimodal speed/quality benchmark shows Gemma+MTP is 1.58x faster overall with no deterministic rubric-quality loss
- **Verified**: full PySide6 app-borne seven-act demo passes with Gemma+MTP+MiniCPM TTS

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

### Gemma Input With MiniCPM Output
Gemma profiles use Gemma for native text, image, audio, and video input. If spoken output is requested, OmniChat hands the generated text to MiniCPM streaming TTS. This keeps Gemma as the reasoning and perception model while using MiniCPM only for the audio generation capability Gemma does not provide.

### MTP Speculative Decoding
The MTP profile keeps the same Gemma 4 target checkpoint and adds the official assistant drafter. The assistant proposes tokens, but the target Gemma model remains authoritative. This is why the quality benchmark can compare speed and output parity directly against the old Gemma profile.

## Benchmarks And Verification

Current major verification artifacts:

| Artifact | Purpose |
|----------|---------|
| `benchmark_outputs/gemma4_mtp_multimodal_2026-05-15/report/speed_quality_report.md` | README-ready speed and quality summary for old Gemma versus Gemma+MTP. |
| `benchmark_outputs/gemma4_mtp_multimodal_2026-05-15/comparison.json` | Raw timing and response-preview comparison. |
| `benchmark_outputs/gemma4_mtp_multimodal_2026-05-15/quality_comparison.json` | Deterministic rubric quality scores. |
| `demo_outputs/rt_app_full_demo_2026-05-15/rt_full_demo_probe.json` | Full PySide6 app-borne seven-act demo result. |

Commands:

```bash
.venv/Scripts/python.exe -m benchmarks.gemma_mtp_multimodal --max-new-tokens 96 --temperature 0.0 --output-dir benchmark_outputs/gemma4_mtp_multimodal_YYYY-MM-DD
.venv/Scripts/python.exe -m benchmarks.evaluate_gemma_mtp_quality --bench-dir benchmark_outputs/gemma4_mtp_multimodal_YYYY-MM-DD
.venv/Scripts/python.exe outputs/debug/rt_full_demo_probe.py --profile gemma4_e4b_transformers_mtp_mincpm_tts --output-dir demo_outputs/rt_app_full_demo_YYYY-MM-DD
```

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
