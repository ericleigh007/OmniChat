# Voice Samples

Place `.wav` files here to use as voice references for OmniChat.

This directory is the default location. You can change it via:
- `audio.voices_dir` in `args/settings.yaml`
- `--voices-dir /path/to/voices` command-line flag

## Requirements
- Format: WAV, 16kHz sample rate, mono
- Duration: 3-5 seconds of clear speech works best (longer clips are truncated to the configured sample length before sending to the model)
- Naming: use the person's name as filename (e.g., `morgan_freeman.wav`)

## Usage
Type in the text box:
- "Change voice to morgan freeman"
- "Switch to sarah's voice"
- "Use the default voice"

Or select a voice from the dropdown. Voice commands must be typed, not spoken.

OmniChat will fuzzy-match the name against files in this directory.

## Note
Voice WAV files are excluded from the git repo via `.gitignore`. Supply your own samples.
