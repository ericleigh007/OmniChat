# OmniChat — Setup Guide

Step-by-step instructions to get OmniChat running from a fresh clone.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.12 or later |
| **NVIDIA GPU** | 10+ GB VRAM (quantized) or 20+ GB (full precision) |
| **CUDA** | 12.x with a compatible PyTorch build |
| **OS** | Windows 11 (tested). Linux should work but is untested. |
| **Disk space** | ~40 GB for the model (downloaded on first run, cached by HuggingFace) |

Tested on: NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), Windows 11, Python 3.12.

## Step 1: Clone the Repo

```bash
git clone https://github.com/ericleigh007/OmniChat.git
cd OmniChat
```

## Step 2: Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

## Step 3: Install PyTorch with CUDA

PyTorch must be installed with CUDA support **before** the other dependencies. Check https://pytorch.org/get-started/locally/ for the command matching your CUDA version.

For CUDA 12.x:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is working:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

You should see your GPU name. If CUDA is not available, the model will fall back to CPU (unusably slow for a 9B parameter model).

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **transformers** (4.51.0) + **accelerate** — HuggingFace model loading
- **minicpmo-utils** — MiniCPM-o tokenizer and audio utilities
- **gradio** — web UI framework
- **PySide6** — desktop app framework
- **librosa**, **soundfile**, **sounddevice** — audio I/O and processing
- **openpyxl** — Excel output
- **Pillow** — image handling
- **pyyaml** — settings file parsing
- **pytest**, **pytest-qt** — testing

## Step 5: Download the Model

The model downloads automatically on first launch (~40 GB, cached at `~/.cache/huggingface/`). To pre-download it:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM-o-4_5', allow_patterns=['*.json', '*.py', '*.safetensors', '*.txt', '*.model'])"
```

Or just run setup (does all of the above checks):
```bash
python setup.py
```

## Step 6: Apply Gradio Patches (Gradio UI only)

The Gradio web UI requires 4 bug fixes to Gradio 6.6.0's streaming audio. These are local patches to your venv — see [README.md](README.md#gradio-patches) for details.

The patched files are in the `ericleigh007/gradio` fork (branch `fix/streaming-audio-multi-turn`). To apply:

```bash
# From the OmniChat directory:
# 1. Patch blocks.py (backend)
# 2. Patch the minified JS bundle (frontend)
```

**Note:** The desktop app (PySide6) does not use Gradio and does not need these patches. If you only plan to use the desktop app, skip this step.

*TODO: Automate patch application in setup.py.*

## Step 7: Add Voice Samples (Optional)

Voice samples are not included in the repo. To use voice cloning:

1. Record or find a 3-5 second WAV clip of the voice you want (clear speech, minimal background noise)
2. Convert to 16 kHz mono WAV (the app auto-resamples if needed, but 16kHz is ideal)
3. Place it in the `voices/` directory (or configure a different path — see below)

```bash
# Name files as: firstname_lastname.wav
voices/morgan_freeman.wav
voices/custom_voice.wav
```

Or configure a custom voices directory:
```yaml
# In args/settings.yaml:
audio:
  voices_dir: "/path/to/my/voices"
```

Or pass it on the command line:
```bash
python main.py --voices-dir /path/to/my/voices
python rt_main.py --voices-dir /path/to/my/voices
```

You can also upload voices through the Settings tab in either app.

## Step 8: Launch

### Gradio Web UI
```bash
# Windows
launch.bat

# Or directly
python main.py
```
Opens at http://localhost:7860. First launch takes ~30-60 seconds (model loading).

### PySide6 Desktop App
```bash
# Windows
launch_rt.bat

# Or directly
python rt_main.py
```
A splash screen shows while the model loads, then the main window opens.

**Only one app can run at a time** — they share the same GPU model singleton.

**Quantization:** If your GPU has less than 20 GB VRAM, add `--quantization int8` or `--quantization int4`:
```bash
launch.bat --quantization int8
launch_rt.bat --quantization int4
```

## Step 9: Verify with Tests

```bash
# Unit tests (no GPU needed, ~4 seconds)
python -m pytest tests/ -v -m "not gpu"

# GPU integration tests (model must be loaded, ~3 minutes)
python -m pytest tests/test_integration.py -v -s
```

All 276 unit tests should pass. The 23 GPU tests require the model to be loaded and a working CUDA setup.

## Configuration

All settings are in `args/settings.yaml`. Key settings to adjust:

| Setting | Default | What It Does |
|---------|---------|-------------|
| `model.name` | `openbmb/MiniCPM-o-4_5` | HuggingFace model ID |
| `model.dtype` | `bfloat16` | Model precision (bf16 needs ~19 GB VRAM) |
| `model.quantization` | `none` | `none` (bf16), `int8` (~10-12 GB), `int4` (~11 GB) |
| `audio.voices_dir` | `voices` | Path to voice WAV samples |
| `audio.voice_sample_length_s` | `5.0` | Seconds of voice clip sent for cloning |
| `inference.temperature` | `0.7` | Sampling randomness (0.0 = deterministic) |
| `inference.repetition_penalty` | `1.5` | Reduces voice sample parroting |
| `server.port` | `7860` | Gradio web UI port |
| `server.share` | `false` | Set `true` for a public Gradio URL |

See the full [settings.yaml](args/settings.yaml) for all options including audio leveling, VAD tuning, and conversation mode parameters.

## Troubleshooting

### "CUDA not available"
- Verify CUDA is installed: `nvidia-smi` should show your GPU
- Verify PyTorch sees it: `python -c "import torch; print(torch.cuda.is_available())"`
- You may need to reinstall PyTorch with the correct CUDA version from https://pytorch.org

### "Out of memory" on model load
- MiniCPM-o 4.5 needs ~19 GB VRAM in bf16 (full precision)
- Close other GPU-using applications
- Use `--quantization int8` for GPUs with 12-20 GB VRAM (~10-12 GB usage)
- Use `--quantization int4` for GPUs under 12 GB (~11 GB usage)
- Or set `model.quantization` in `args/settings.yaml`

### Audio sounds garbled or broken with quantization
- Only the LLM transformer layers are quantized — audio, vision, TTS, and projection modules are kept in bf16 to preserve multimodal quality
- Both INT8 and INT4 use bitsandbytes on the same base model (no separate checkpoint)
- **Text chat works fine** in all quantization modes — audio/speech is the risk area
- When in doubt, use `--quantization none` (the default, bf16)

### Model downloads are slow
- The model is ~40 GB. First download takes time.
- Downloads are cached at `~/.cache/huggingface/hub/`. Subsequent runs are fast.
- Set `HF_HOME` environment variable to change the cache location.

### Gradio streaming audio doesn't play
- Make sure the Gradio patches are applied (Step 6)
- Or use the desktop app instead — it bypasses Gradio entirely

### No sound output (desktop app)
- Check that `sounddevice` can see your audio device: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- The default output device is used automatically

### Voice cloning sounds like the original sample
- Reduce `voice_sample_length_s` to 3-5 seconds
- Increase `repetition_penalty` (default 1.5, try 1.7-2.0)
- Use clips with neutral content (counting, reading random words) rather than distinctive phrases
