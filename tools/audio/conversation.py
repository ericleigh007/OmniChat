"""ConversationManager — continuous voice chat with VAD.

State machine + Silero-VAD for hands-free voice conversation.
Three interaction modes:
  - Auto-detect: VAD detects speech start/end, silence = end of turn
  - Push-to-talk: hold button to record, release to send
  - Click per turn: explicit record → send cycle

Anti-vox echo suppression:
  During and after model speech, the VAD threshold is raised so that only
  strong nearby speech (the user) triggers detection, while weaker speaker
  bleed is ignored.  This replaces a hard mic-mute timer.

Barge-in:
  The user can interrupt the model mid-response.  During MODEL_SPEAKING,
  VAD runs with an elevated threshold.  If speech is detected for several
  consecutive chunks, a barge_in flag is returned so the caller can stop
  playback and process the new user turn.

Usage:
    conv = ConversationManager(config)
    conv.start()

    # In Gradio stream handler (called every ~0.5s):
    result = conv.on_audio_chunk(chunk_16k)
    if result.barge_in:
        # Stop current model output, process new speech
        ...
    elif result.audio_ready is not None:
        # Process accumulated speech through the model
        ...

    conv.stop()
"""

import enum
import time
from typing import Optional

import numpy as np


# ── Data types ────────────────────────────────────────────────────────────────


class ConversationState(enum.Enum):
    """States in the conversation state machine."""
    OFF = "off"                      # Not active
    LISTENING = "listening"          # Mic active, waiting for speech
    USER_SPEAKING = "user_speaking"  # VAD detected speech, buffering
    PROCESSING = "processing"        # Speech ended, model will process
    MODEL_SPEAKING = "model_speaking"  # Model generating response
    IDLE = "idle"                    # Response done, brief transition


class ConversationMode(enum.Enum):
    """Interaction mode for the conversation."""
    AUTO_DETECT = "auto_detect"
    PUSH_TO_TALK = "push_to_talk"
    CLICK_PER_TURN = "click_per_turn"


class ChunkResult:
    """Result from processing an audio chunk."""
    __slots__ = ("state", "audio_ready", "barge_in")

    def __init__(
        self,
        state: ConversationState,
        audio_ready: Optional[np.ndarray] = None,
        barge_in: bool = False,
    ):
        self.state = state
        self.audio_ready = audio_ready  # accumulated speech when turn ends
        self.barge_in = barge_in        # True = user interrupted model


# ── Conversation manager ──────────────────────────────────────────────────────


class ConversationManager:
    """State machine for continuous voice conversation with VAD."""

    State = ConversationState
    Mode = ConversationMode

    # (color, label) for each state — used by format_state_html()
    STATE_DISPLAY = {
        ConversationState.OFF:            ("#888888", "Conversation off"),
        ConversationState.LISTENING:      ("#22c55e", "Listening..."),
        ConversationState.USER_SPEAKING:  ("#3b82f6", "You're speaking..."),
        ConversationState.PROCESSING:     ("#eab308", "Thinking..."),
        ConversationState.MODEL_SPEAKING: ("#a855f7", "Responding..."),
        ConversationState.IDLE:           ("#f97316", "Ready"),
    }

    MODE_LABELS = {
        ConversationMode.AUTO_DETECT:    "Auto-detect",
        ConversationMode.PUSH_TO_TALK:   "Push-to-talk",
        ConversationMode.CLICK_PER_TURN: "Click per turn",
    }

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.state = self.State.OFF
        self.mode = self.Mode.AUTO_DETECT

        # VAD settings
        self._vad_model = None
        self._vad_threshold: float = config.get("vad_threshold", 0.5)
        self._vad_sr: int = 16000
        self._vad_chunk_size: int = 512  # 32ms at 16kHz

        # Speech buffering
        self._audio_buffer: list[np.ndarray] = []
        self._silence_chunks: int = 0
        self._silence_threshold_s: float = config.get("silence_threshold_s", 1.5)

        # Anti-vox echo suppression: instead of hard-muting the mic after
        # model speech, we raise the VAD threshold so only strong nearby
        # speech (the user) triggers, while weaker speaker bleed is ignored.
        self._echo_cooldown_s: float = config.get("echo_cooldown_s", 1.5)
        self._cooldown_until: float = 0.0
        self._antivox_boost: float = config.get("antivox_boost", 0.25)

        # Barge-in: let the user interrupt model speech.
        # During MODEL_SPEAKING, VAD runs with an elevated threshold.
        # After N consecutive speech-detected chunks, we signal barge-in.
        self._barge_in_enabled: bool = config.get("barge_in_enabled", True)
        self._barge_in_threshold: float = config.get("barge_in_threshold", 0.75)
        self._barge_in_chunks: int = config.get("barge_in_chunks", 3)
        self._barge_in_count: int = 0

        # Wake word (future enhancement)
        self._wake_word: str = config.get("wake_word", "omnichat")
        self._dormant_timeout_s: float = config.get("dormant_timeout_s", 30)
        self._last_activity: float = 0.0

        # Push-to-talk
        self._ptt_recording: bool = False

    # ── VAD lazy loading ──────────────────────────────────────────────

    def _ensure_vad(self) -> None:
        """Lazy-load Silero-VAD. Runs on CPU, ~5 MB."""
        if self._vad_model is not None:
            return
        import torch
        model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        self._vad_model = model

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start conversation mode. Loads VAD if auto-detect."""
        if self.mode == self.Mode.AUTO_DETECT:
            self._ensure_vad()
            self._vad_model.reset_states()
        self.state = self.State.LISTENING
        self._audio_buffer = []
        self._silence_chunks = 0
        self._barge_in_count = 0
        self._last_activity = time.time()

    def stop(self) -> None:
        """Stop conversation mode. Clears all buffers."""
        self.state = self.State.OFF
        self._audio_buffer = []
        self._silence_chunks = 0
        self._barge_in_count = 0
        self._ptt_recording = False

    @property
    def active(self) -> bool:
        """True if conversation mode is running."""
        return self.state != self.State.OFF

    # ── Audio chunk processing ────────────────────────────────────────

    def on_audio_chunk(self, chunk_16k: np.ndarray) -> ChunkResult:
        """Process a 16 kHz float32 mono audio chunk.

        Called by Gradio's stream handler every ~0.5s. Returns a ChunkResult
        with the current state, optionally the accumulated speech audio
        when a turn boundary is detected, and a barge_in flag if the user
        interrupted the model.

        Args:
            chunk_16k: Audio samples at 16 kHz, float32, 1-D.

        Returns:
            ChunkResult with .state, optionally .audio_ready, and .barge_in.
        """
        if self.state == self.State.OFF:
            return ChunkResult(self.state)

        if self.state == self.State.PROCESSING:
            return ChunkResult(self.state)

        # During model speech: check for barge-in (user interruption)
        if self.state == self.State.MODEL_SPEAKING:
            if self._barge_in_enabled:
                return self._handle_barge_in(chunk_16k)
            return ChunkResult(self.state)

        # Anti-vox: during echo cooldown, use elevated threshold.
        # Strong nearby speech (user) still gets through; weaker speaker
        # bleed is filtered out.
        if time.time() < self._cooldown_until:
            boosted_threshold = self._vad_threshold + self._antivox_boost
            has_speech = self._run_vad(chunk_16k, threshold=boosted_threshold)
            if not has_speech:
                return ChunkResult(self.state)
            # Strong speech detected — user is talking, end cooldown early
            self._cooldown_until = 0.0
            print(f"  [antivox] Strong speech during cooldown — resuming normal detection")

        if self.mode == self.Mode.AUTO_DETECT:
            return self._handle_auto_detect(chunk_16k)
        elif self.mode == self.Mode.PUSH_TO_TALK:
            return self._handle_push_to_talk(chunk_16k)
        else:
            # Click-per-turn doesn't use streaming VAD
            return ChunkResult(self.state)

    def _handle_auto_detect(self, chunk: np.ndarray) -> ChunkResult:
        """VAD-based speech boundary detection."""
        has_speech = self._run_vad(chunk)

        if self.state == self.State.LISTENING:
            if has_speech:
                self.state = self.State.USER_SPEAKING
                self._audio_buffer = [chunk]
                self._silence_chunks = 0
                self._last_activity = time.time()
            return ChunkResult(self.state)

        if self.state == self.State.USER_SPEAKING:
            self._audio_buffer.append(chunk)
            if has_speech:
                self._silence_chunks = 0
            else:
                self._silence_chunks += 1
                chunk_duration = len(chunk) / self._vad_sr
                silence_s = self._silence_chunks * chunk_duration
                if silence_s >= self._silence_threshold_s:
                    # Turn boundary — collect buffered speech
                    audio = np.concatenate(self._audio_buffer)
                    self._audio_buffer = []
                    self._silence_chunks = 0
                    self.state = self.State.PROCESSING
                    self._last_activity = time.time()
                    self._vad_model.reset_states()
                    return ChunkResult(self.state, audio_ready=audio)
            return ChunkResult(self.state)

        if self.state == self.State.IDLE:
            self.state = self.State.LISTENING
            if self._vad_model is not None:
                self._vad_model.reset_states()
            return ChunkResult(self.state)

        return ChunkResult(self.state)

    def _handle_barge_in(self, chunk: np.ndarray) -> ChunkResult:
        """Check for user interruption during model speech.

        Uses an elevated VAD threshold so that only strong nearby speech
        (the user speaking into the mic) triggers, not the model's own
        audio playing through speakers.
        """
        has_speech = self._run_vad(chunk, threshold=self._barge_in_threshold)

        if has_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._barge_in_chunks:
                # User is interrupting — transition to USER_SPEAKING
                print(f"  [barge-in] User interrupted after "
                      f"{self._barge_in_count} consecutive speech chunks")
                self._barge_in_count = 0
                self.state = self.State.USER_SPEAKING
                self._audio_buffer = [chunk]
                self._silence_chunks = 0
                self._cooldown_until = 0.0  # cancel any pending cooldown
                return ChunkResult(self.state, barge_in=True)
        else:
            self._barge_in_count = 0

        return ChunkResult(self.state)

    def _handle_push_to_talk(self, chunk: np.ndarray) -> ChunkResult:
        """Buffer audio while PTT flag is set."""
        if self._ptt_recording:
            self._audio_buffer.append(chunk)
        return ChunkResult(self.state)

    # ── Push-to-talk controls ─────────────────────────────────────────

    def ptt_start(self) -> None:
        """Start recording (push-to-talk mode)."""
        if self.state == self.State.OFF:
            return
        self.state = self.State.USER_SPEAKING
        self._ptt_recording = True
        self._audio_buffer = []

    def ptt_stop(self) -> Optional[np.ndarray]:
        """Stop recording (push-to-talk mode). Returns accumulated audio."""
        self._ptt_recording = False
        if not self._audio_buffer:
            self.state = self.State.LISTENING
            return None
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        self.state = self.State.PROCESSING
        return audio

    # ── Model lifecycle callbacks ─────────────────────────────────────

    def on_model_start(self) -> None:
        """Called when model begins generating a response."""
        self.state = self.State.MODEL_SPEAKING
        self._barge_in_count = 0

    def on_model_done(self) -> None:
        """Called when model finishes generating. Resumes listening."""
        self.state = self.State.IDLE
        self._last_activity = time.time()
        self._barge_in_count = 0
        # Anti-vox cooldown: use elevated threshold for a short window
        # after model finishes, so speaker tail-end doesn't trigger VAD.
        self._cooldown_until = time.time() + self._echo_cooldown_s
        print(f"  [conversation] Anti-vox cooldown: {self._echo_cooldown_s}s "
              f"(threshold boosted by +{self._antivox_boost})")
        # In auto-detect, transition directly to LISTENING
        if self.mode == self.Mode.AUTO_DETECT:
            self.state = self.State.LISTENING
            if self._vad_model is not None:
                self._vad_model.reset_states()

    # ── Mode switching ────────────────────────────────────────────────

    def set_mode(self, mode) -> None:
        """Change interaction mode by enum or display label string."""
        if isinstance(mode, ConversationMode):
            new_mode = mode
        else:
            label_map = {v: k for k, v in self.MODE_LABELS.items()}
            new_mode = label_map.get(mode, self.Mode.AUTO_DETECT)
        self.mode = new_mode
        # Load VAD if switching to auto-detect while active
        if new_mode == self.Mode.AUTO_DETECT and self.state != self.State.OFF:
            self._ensure_vad()
            self._vad_model.reset_states()

    # ── VAD inference ─────────────────────────────────────────────────

    _vad_log_count = 0  # class-level diagnostic counter

    def _run_vad(self, chunk: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Run Silero-VAD on a 16 kHz chunk. Returns True if speech detected.

        Processes the chunk in 512-sample (32 ms) windows. Returns True if
        any window exceeds the speech probability threshold.

        Args:
            chunk: Audio samples at 16 kHz, float32, 1-D.
            threshold: Override VAD threshold. Defaults to self._vad_threshold.
        """
        if self._vad_model is None:
            return False

        if threshold is None:
            threshold = self._vad_threshold

        import torch

        speech_found = False
        max_prob = 0.0
        pos = 0
        while pos + self._vad_chunk_size <= len(chunk):
            sub = chunk[pos : pos + self._vad_chunk_size]
            tensor = torch.from_numpy(sub.copy()).float()
            prob = self._vad_model(tensor, self._vad_sr).item()
            max_prob = max(max_prob, prob)
            if prob >= threshold:
                speech_found = True
            pos += self._vad_chunk_size

        # Log first 10 chunks and then any speech detection for diagnostics
        ConversationManager._vad_log_count += 1
        if ConversationManager._vad_log_count <= 10 or speech_found:
            windows = pos // self._vad_chunk_size
            print(f"  [VAD] chunk={len(chunk)} windows={windows} "
                  f"max_prob={max_prob:.3f} thr={threshold:.2f} "
                  f"speech={speech_found} state={self.state.value}")
        return speech_found

    # ── UI helpers ────────────────────────────────────────────────────

    def format_state_html(self) -> str:
        """Render current state as an HTML indicator bar."""
        color, text = self.STATE_DISPLAY.get(
            self.state, ("#888", "Unknown"),
        )
        # Pulsing animation for LISTENING state
        pulse_css = ""
        if self.state == self.State.LISTENING:
            pulse_css = "animation:convpulse 1.5s ease-in-out infinite;"

        mode_label = self.MODE_LABELS.get(self.mode, "")

        return (
            "<style>@keyframes convpulse{0%,100%{opacity:1}50%{opacity:0.4}}</style>"
            '<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
            'background:#1a1a2e;border-radius:8px;font-family:sans-serif;">'
            f'<span style="width:12px;height:12px;border-radius:50%;'
            f'background:{color};{pulse_css}display:inline-block;"></span>'
            f'<span style="color:#e0e0e0;font-size:14px;font-weight:500;">{text}</span>'
            f'<span style="color:#888;font-size:12px;margin-left:auto;">{mode_label}</span>'
            "</div>"
        )
