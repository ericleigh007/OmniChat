# Tools Manifest

Index of all available tool scripts in the OmniChat system.
Always check this file before writing a new script — a tool may already exist.

## Model (`tools/model/`)

| Script | Description |
|--------|-------------|
| `model_manager.py` | Singleton model manager and backend router for MiniCPM, Qwen, Gemma, and llama.cpp profiles; exposes `chat()`, `chat_streaming()`, `chat_streaming_with_playback()`, `process_image()`, and `process_video()`. |
| `backends/gemma_transformers_backend.py` | Gemma 4 E4B IT local Transformers backend with native text/image/audio/video input, optional MTP assistant drafter, and optional MiniCPM streaming TTS bridge for spoken output. |

## Audio (`tools/audio/`)

| Script | Description |
|--------|-------------|
| `voice_manager.py` | Manage voice reference samples — list, lookup (fuzzy), add, delete voice clips |
| `extract_voice.py` | Extract voice sample from MP4/video files using ffmpeg, saves as 16kHz mono WAV |
| `streaming_player.py` | Queue-based streaming audio player using sounddevice.OutputStream for real-time playback during generation |
| `conversation.py` | Continuous voice chat state machine with Silero-VAD, three interaction modes (auto-detect, push-to-talk, click-per-turn) |

## Shared (`tools/shared/`)

| Script | Description |
|--------|-------------|
| `session.py` | Shared helpers for both Gradio and PySide6 apps — settings loader, model profile runtime configuration, voice command detection, audio normalization, voice ref truncation |

## Vision (`tools/vision/`)

| Script | Description |
|--------|-------------|
| `process_media.py` | Image analysis, document OCR, and video understanding with auto format detection |
| `pdf_processor.py` | PDF-to-image rendering (PyMuPDF), per-page OCR via scan_document, table aggregation across pages |

## Output (`tools/output/`)

| Script | Description |
|--------|-------------|
| `save_output.py` | Save processed content as markdown, plain text, CSV, TSV, or Excel; supports auto-format detection and explicit Save As paths |

## Tests (`tests/`)

| Script | Description |
|--------|-------------|
| `conftest.py` | Shared pytest fixtures — temp dirs, sample audio arrays |
| `test_session.py` | Shared session helpers — settings loading, audio normalization, voice ref truncation |
| `test_rt_audio.py` | Mock-based tests for PySide6 audio pipeline — MicInputStream, ModelInferenceThread, AudioPipeline |
| `test_voice_commands.py` | Voice command regex detection from transcribed speech |
| `test_voice_manager.py` | Voice sample fuzzy matching, name normalization, CRUD operations |
| `test_audio_processing.py` | Audio normalization (dtype, channels, sample rate) and fade-in |
| `test_format_detection.py` | Output format auto-detection and table parsing |
| `test_save_output.py` | Markdown, text, CSV, TSV, Excel file saving, explicit path (Save As) support |
| `test_model_manager.py` | Mock-based tests for model reset logic, message construction, and streaming preprocessing |
| `test_streaming_player.py` | Mock-based tests for StreamingAudioPlayer queue/callback logic |
| `test_integration.py` | GPU integration tests — real model, multi-turn echo, audio gen, voice switching, streaming audio, streaming audio input |
| `test_conversation.py` | Mock-based tests for ConversationManager state machine, VAD integration, mode switching, and turn detection |
| `test_gradio_streaming.py` | Streaming helpers (WAV bytes, chunk normalization, ffmpeg config, audio chunking, pydub monkeypatch, ADTS pipeline) |
| `test_vad_integration.py` | Real Silero-VAD tests — model loading, speech detection, conversation flow, Gradio mic simulation, audio normalization |
| `test_pdf_processor.py` | PDF rendering (real tiny PDFs) and OCR aggregation (mocked model) |
| `test_demo_smoke.py` | Pytest wrapper — runs full demo in headless mode as regression test |
| `test_rt_full_demo_live.py` | GPU live test — drives the real PySide6 app through the full app-borne demo probe |
| `test_gemma_mtp_multimodal_benchmark.py` | Tests Gemma MTP benchmark case generation, report writing, and deterministic quality evaluation |

## Benchmarks (`benchmarks/`)

| Script | Description |
|--------|-------------|
| `run_benchmark.py` | Orchestrator — spawns one subprocess per quantization level, then generates comparison report |
| `run_single_quant.py` | Worker — loads model at specified quantization, runs all prompts, saves raw (pre-leveling) text and WAV outputs |
| `analyze_results.py` | Post-processor — computes audio metrics (RMS, spectral centroid, ZCR), generates mel spectrogram comparisons, writes markdown report |
| `gemma_mtp_multimodal.py` | 50-case old Gemma versus Gemma+MTP multimodal speed benchmark with raw outputs and README-ready reports |
| `evaluate_gemma_mtp_quality.py` | Deterministic rubric scorer for saved Gemma MTP benchmark outputs; writes quality and combined speed/quality reports |
| `prompts.py` | Fixed prompt definitions for benchmark reproducibility (echo + free prompts) |
| `benchmark.bat` | Windows launcher for the benchmark suite |

## Demos (`demos/`)

| Script | Description |
|--------|-------------|
| `run_demo.py` | Live capabilities showcase — 7 acts with audio playback, image display, and pass/fail reporting |
| `demo_assets.py` | Pillow-based test image generators (geometric scene, fake invoice) |
| `demo_narrative.py` | ASCII-safe terminal presentation helpers (banners, narration, summary table) |
