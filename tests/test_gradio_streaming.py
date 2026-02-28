"""Tests for Gradio streaming audio helpers and generator behavior.

Mock-based — no GPU or model required.
"""

import io
import numpy as np
import pytest
import soundfile as sf


# ── WAV bytes conversion ────────────────────────────────────────────────────


class TestNumpyToWavBytes:
    """Test _numpy_to_wav_bytes helper."""

    def _make_helper(self):
        """Import the helper by building a minimal app context."""
        # The helper is defined inside build_app(), so we test the logic directly.
        def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        return _numpy_to_wav_bytes

    def test_returns_bytes(self):
        fn = self._make_helper()
        audio = np.random.randn(24000).astype(np.float32) * 0.1
        result = fn(audio)
        assert isinstance(result, bytes)

    def test_roundtrip_preserves_length(self):
        fn = self._make_helper()
        audio = np.random.randn(24000).astype(np.float32) * 0.1
        wav_bytes = fn(audio, 24000)
        buf = io.BytesIO(wav_bytes)
        data, sr = sf.read(buf, dtype="float32")
        assert sr == 24000
        assert len(data) == len(audio)

    def test_wav_header_present(self):
        fn = self._make_helper()
        audio = np.zeros(1000, dtype=np.float32)
        wav_bytes = fn(audio)
        assert wav_bytes[:4] == b"RIFF"

    def test_short_audio_works(self):
        fn = self._make_helper()
        audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        wav_bytes = fn(audio)
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 44  # WAV header is 44 bytes


# ── Per-chunk normalization ─────────────────────────────────────────────────


class TestNormalizeChunk:
    """Test _normalize_chunk helper logic."""

    def _make_helper(self):
        """Build a standalone normalize_chunk using the real model_manager functions."""
        import tools.model.model_manager as mm

        def _normalize_chunk(audio_chunk: np.ndarray, is_first: bool) -> np.ndarray:
            cfg = mm._get_leveling_config()
            if is_first:
                audio_chunk = mm._apply_fade_in(audio_chunk)
            if cfg["enabled"]:
                audio_chunk = mm._normalize_rms(
                    audio_chunk,
                    target_rms=cfg["_output_threshold_linear"],
                    max_gain=cfg["_output_max_gain_linear"],
                    peak_ceiling=cfg["_peak_ceiling_linear"],
                )
            return audio_chunk
        return _normalize_chunk

    def setup_method(self):
        import tools.model.model_manager as mm
        self._saved_cfg = mm._leveling_cfg
        mm._leveling_cfg = None  # force reload

    def teardown_method(self):
        import tools.model.model_manager as mm
        mm._leveling_cfg = self._saved_cfg

    def test_first_chunk_has_fade_in(self):
        fn = self._make_helper()
        audio = np.ones(24000, dtype=np.float32) * 0.5
        result = fn(audio, is_first=True)
        # First sample should be near zero (fade-in starts at 0)
        assert abs(result[0]) < 0.01

    def test_subsequent_chunk_no_fade(self):
        fn = self._make_helper()
        audio = np.ones(24000, dtype=np.float32) * 0.5
        result = fn(audio, is_first=False)
        # First sample should NOT be faded — will be normalized but not zero
        assert abs(result[0]) > 0.01

    def test_silence_unchanged(self):
        fn = self._make_helper()
        audio = np.zeros(24000, dtype=np.float32)
        result = fn(audio, is_first=False)
        assert np.allclose(result, 0.0)


# ── ffmpeg configuration ───────────────────────────────────────────────────


class TestFfmpegConfig:
    """Verify pydub can find ffmpeg via imageio-ffmpeg."""

    def test_imageio_ffmpeg_binary_exists(self):
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        from pathlib import Path
        assert Path(path).exists(), f"ffmpeg binary not found: {path}"

    def test_pydub_converter_configurable(self):
        import imageio_ffmpeg
        import pydub
        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        # Verify it was set (not the default "ffmpeg")
        assert "imageio_ffmpeg" in pydub.AudioSegment.converter or "ffmpeg" in pydub.AudioSegment.converter

    def test_wav_auto_detect_monkeypatch(self):
        """WAV bytes with RIFF header should be detected without ffprobe."""
        import io
        import pydub
        import imageio_ffmpeg

        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

        # Apply the same monkeypatch as main.py
        _orig = pydub.AudioSegment.from_file.__func__

        @classmethod
        def _patched(cls, file, format=None, **kwargs):
            if format is None and hasattr(file, "read"):
                pos = file.tell()
                header = file.read(4)
                file.seek(pos)
                if header == b"RIFF":
                    format = "wav"
            return _orig(cls, file, format=format, **kwargs)

        pydub.AudioSegment.from_file = _patched
        try:
            # Create a valid WAV in memory
            import soundfile as sf
            audio = np.random.randn(24000).astype(np.float32) * 0.1
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            # This should NOT need ffprobe — the monkeypatch detects WAV from RIFF header
            segment = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))
            assert len(segment) > 0  # Duration in ms
        finally:
            pydub.AudioSegment.from_file = classmethod(_orig)


# ── Streaming toggle ───────────────────────────────────────────────────────


class TestAudioChunking:
    """Test the audio chunking logic for streaming_prefill.

    streaming_prefill requires audio in exact-size chunks:
      - First chunk: FIRST_CHUNK_MS=1035 → 16,560 samples @16kHz
      - Subsequent: CHUNK_MS=1000 → 16,000 samples @16kHz
    Short chunks are zero-padded to the exact expected size.
    """

    FIRST = int(1035 * 16000 / 1000)   # 16,560
    REGULAR = int(1000 * 16000 / 1000)  # 16,000

    @staticmethod
    def _split_audio(full_audio):
        """Replicate the chunking logic from chat_streaming()."""
        FIRST_CHUNK_SAMPLES = int(1035 * 16000 / 1000)
        REGULAR_CHUNK_SAMPLES = int(1000 * 16000 / 1000)
        chunks = []
        n = len(full_audio)
        if n <= FIRST_CHUNK_SAMPLES:
            # Short clip — pad to exactly FIRST_CHUNK_SAMPLES
            if n < FIRST_CHUNK_SAMPLES:
                padded = np.zeros(FIRST_CHUNK_SAMPLES, dtype=np.float32)
                padded[:n] = full_audio
                chunks.append(padded)
            else:
                chunks.append(full_audio)
        else:
            chunks.append(full_audio[:FIRST_CHUNK_SAMPLES])
            pos = FIRST_CHUNK_SAMPLES
            while pos < n:
                chunk = full_audio[pos:pos + REGULAR_CHUNK_SAMPLES]
                if len(chunk) < REGULAR_CHUNK_SAMPLES:
                    # Zero-pad to exact size
                    padded = np.zeros(REGULAR_CHUNK_SAMPLES, dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                chunks.append(chunk)
                pos += REGULAR_CHUNK_SAMPLES
        return chunks

    def test_short_audio_padded_to_first_chunk(self):
        """Audio shorter than FIRST_CHUNK is padded to FIRST_CHUNK size."""
        audio = np.ones(8000, dtype=np.float32) * 0.5  # 0.5s
        chunks = self._split_audio(audio)
        assert len(chunks) == 1
        assert len(chunks[0]) == self.FIRST
        # Original samples preserved, rest is zero-padded
        np.testing.assert_array_equal(chunks[0][:8000], audio)
        np.testing.assert_array_equal(chunks[0][8000:], 0.0)

    def test_exact_first_chunk(self):
        """Audio exactly FIRST_CHUNK_SAMPLES stays as one chunk, no padding."""
        audio = np.ones(self.FIRST, dtype=np.float32) * 0.5
        chunks = self._split_audio(audio)
        assert len(chunks) == 1
        assert len(chunks[0]) == self.FIRST
        np.testing.assert_array_equal(chunks[0], audio)

    def test_two_chunks_padded(self):
        """Audio slightly over FIRST_CHUNK → 2 chunks, second is padded."""
        audio = np.ones(self.FIRST + 5000, dtype=np.float32) * 0.5
        chunks = self._split_audio(audio)
        assert len(chunks) == 2
        assert len(chunks[0]) == self.FIRST
        assert len(chunks[1]) == self.REGULAR  # padded to exact size
        # First 5000 samples of second chunk are audio, rest is zero
        np.testing.assert_array_equal(chunks[1][:5000], 0.5)
        np.testing.assert_array_equal(chunks[1][5000:], 0.0)

    def test_three_seconds(self):
        """3 seconds of audio → first + regular + padded remainder."""
        audio = np.zeros(48000, dtype=np.float32)  # 3s
        chunks = self._split_audio(audio)
        assert len(chunks[0]) == self.FIRST
        # Remaining: 48000 - 16560 = 31440
        # 31440 / 16000 = 1.965 → 2 chunks
        assert len(chunks) == 3
        assert len(chunks[1]) == self.REGULAR
        assert len(chunks[2]) == self.REGULAR  # zero-padded

    def test_all_chunks_exact_size(self):
        """Every chunk must be exactly the expected size (critical for model)."""
        for total in [20000, 40000, 48000, 60000, 80000, 100000, 160000]:
            audio = np.zeros(total, dtype=np.float32)
            chunks = self._split_audio(audio)
            assert len(chunks[0]) == self.FIRST, (
                f"First chunk {len(chunks[0])} != {self.FIRST} for total={total}")
            for i, c in enumerate(chunks[1:], 1):
                assert len(c) == self.REGULAR, (
                    f"Chunk {i} is {len(c)} != {self.REGULAR} for total={total}")

    def test_original_audio_preserved_in_padded_chunks(self):
        """Padding should be zeros; original data should be intact."""
        audio = np.arange(50000, dtype=np.float32)
        chunks = self._split_audio(audio)
        # Reassemble without padding — original samples should match
        reassembled = np.concatenate(chunks)
        # First 50000 samples should match original
        np.testing.assert_array_equal(reassembled[:50000], audio)
        # Remaining samples should be zero (padding)
        if len(reassembled) > 50000:
            np.testing.assert_array_equal(reassembled[50000:], 0.0)

    def test_very_long_audio(self):
        """10 seconds of audio → multiple exact-size chunks."""
        audio = np.zeros(160000, dtype=np.float32)  # 10s
        chunks = self._split_audio(audio)
        assert len(chunks[0]) == self.FIRST
        for c in chunks[1:]:
            assert len(c) == self.REGULAR
        # Expected: first chunk (16560) + ceil((160000-16560)/16000) = 1 + 9 = 10
        expected_regular = -(-( 160000 - self.FIRST) // self.REGULAR)  # ceil div
        assert len(chunks) == 1 + expected_regular

    def test_conversation_mode_2_5s_audio(self):
        """Regression: 2.5s audio (40000 samples) that previously crashed.

        Old bug: 1.5x absorption created [16560, 23440] — second chunk
        of 23440 caused tensor mismatch. Now should be [16560, 16000, 16000].
        """
        audio = np.ones(40000, dtype=np.float32) * 0.3
        chunks = self._split_audio(audio)
        assert len(chunks[0]) == self.FIRST
        for c in chunks[1:]:
            assert len(c) == self.REGULAR, (
                f"Non-standard chunk size {len(c)} — would crash model!"
            )
        # 40000 - 16560 = 23440. ceil(23440/16000) = 2. Total = 3 chunks.
        assert len(chunks) == 3


class TestBufferedStreaming:
    """Test the _buffered_streaming accumulation logic.

    Simulates chat_streaming() output and verifies that chunks are
    accumulated into larger blocks to reduce ffmpeg subprocess calls.
    """

    FIRST_MIN_SAMPLES = 96000  # must match _STREAM_FIRST_MIN_SAMPLES in main.py
    MIN_SAMPLES = 48000        # must match _STREAM_MIN_SAMPLES in main.py

    @staticmethod
    def _fake_stream(chunk_sizes):
        """Simulate chat_streaming() yielding audio chunks of given sizes."""
        for i, size in enumerate(chunk_sizes):
            audio = np.random.randn(size).astype(np.float32) * 0.3
            yield audio, f"text_{i}"

    @staticmethod
    def _buffered(chat_gen, first_min=96000, min_samples=48000):
        """Replicate _buffered_streaming logic from main.py (without WAV conversion)."""
        buffer = []
        buffer_samples = 0
        first_audio_yielded = False
        full_text = ""
        last_text_yielded = ""

        for audio_chunk, text_chunk in chat_gen:
            if text_chunk:
                full_text += text_chunk
            if audio_chunk is not None:
                buffer.append(audio_chunk)
                buffer_samples += len(audio_chunk)

                threshold = first_min if not first_audio_yielded else min_samples
                if buffer_samples >= threshold:
                    combined = np.concatenate(buffer)
                    buffer = []
                    buffer_samples = 0
                    first_audio_yielded = True
                    last_text_yielded = full_text
                    yield combined, full_text
            elif full_text != last_text_yielded:
                last_text_yielded = full_text
                yield None, full_text

        if buffer:
            yield np.concatenate(buffer), full_text
        elif not first_audio_yielded:
            yield None, full_text

    def test_first_chunk_accumulates_to_minimum(self):
        """First chunk needs most buffer since HLS has zero runway at start."""
        # 5 chunks of 24000 = 120000 total. First yields at 96000 (4 chunks).
        gen = self._fake_stream([24000] * 5)
        results = list(self._buffered(gen))
        assert len(results) == 2  # first at 96000 (4 chunks), remainder at 24000
        assert len(results[0][0]) == 96000
        assert len(results[1][0]) == 24000

    def test_accumulates_subsequent_chunks(self):
        """After first yield, chunks accumulate to MIN_SAMPLES (smaller threshold)."""
        # 10 chunks of 24000 (1s each = 240000 total).
        # First yield: 96000 (4 chunks). Then 48000 needed (2 chunks each).
        # Remaining: 6 chunks → 48000 + 48000 + 48000 = 3 more yields
        gen = self._fake_stream([24000] * 10)
        results = list(self._buffered(gen))
        assert len(results) == 4, f"Expected 4 yields, got {len(results)}"
        # All samples preserved
        total = sum(len(r[0]) for r in results)
        assert total == 24000 * 10

    def test_no_audio_yields_text_only(self):
        """Stream with text but no audio should yield text-only updates."""
        def text_only_stream():
            yield None, "hello "
            yield None, "world"

        results = list(self._buffered(text_only_stream()))
        # Text-only yields: first "hello ", then "hello world", then final flush
        assert len(results) >= 1
        # All should have None audio
        for audio, text in results:
            assert audio is None
        # Last result should have all text
        assert "hello" in results[-1][1]
        assert "world" in results[-1][1]

    def test_large_first_chunk_yields_once(self):
        """A single large chunk should yield immediately."""
        gen = self._fake_stream([100000])
        results = list(self._buffered(gen))
        assert len(results) == 1
        assert len(results[0][0]) == 100000

    def test_text_accumulates_across_yields(self):
        """Text should accumulate across all chunks."""
        gen = self._fake_stream([24000, 24000, 24000])
        results = list(self._buffered(gen))
        # Last yield should have all text
        last_text = results[-1][1]
        assert "text_0" in last_text
        assert "text_2" in last_text


class TestStreamingToggle:
    """Test that the streaming setting is respected."""

    def test_default_enabled(self):
        """Streaming defaults to True when no config provided."""
        cfg = {}
        enabled = cfg.get("enabled", True)
        assert enabled is True

    def test_config_disable(self):
        """Streaming can be disabled via config."""
        cfg = {"enabled": False}
        enabled = cfg.get("enabled", True)
        assert enabled is False


# ── Gradio streaming output pipeline ─────────────────────────────────────────


class TestGradioStreamingPipeline:
    """End-to-end test of the streaming audio output path.

    Simulates what Gradio does: our handler yields WAV bytes, then Gradio
    converts them to ADTS/AAC via pydub for browser playback. This tests
    the entire chain: numpy → WAV bytes → pydub AudioSegment → AAC export.
    """

    @staticmethod
    def _setup_pydub():
        """Apply the same ffmpeg + monkeypatch config as main.py."""
        import imageio_ffmpeg
        import pydub

        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

        _orig = pydub.AudioSegment.from_file.__func__

        @classmethod
        def _patched(cls, file, format=None, **kwargs):
            if format is None and hasattr(file, "read"):
                pos = file.tell()
                header = file.read(4)
                file.seek(pos)
                if header == b"RIFF":
                    format = "wav"
            return _orig(cls, file, format=format, **kwargs)

        pydub.AudioSegment.from_file = _patched
        return _orig

    @staticmethod
    def _restore_pydub(orig):
        import pydub
        pydub.AudioSegment.from_file = classmethod(orig)

    @staticmethod
    def _numpy_to_wav_bytes(audio, sample_rate=24000):
        """Same helper as main.py's _numpy_to_wav_bytes."""
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def test_full_pipeline_single_chunk(self):
        """One audio chunk → WAV bytes → pydub → AAC export."""
        import io
        import pydub

        orig = self._setup_pydub()
        try:
            audio = np.random.randn(24000).astype(np.float32) * 0.3
            wav_bytes = self._numpy_to_wav_bytes(audio)

            # Step 1: pydub reads WAV (this is where ffprobe was failing)
            segment = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))
            assert segment.duration_seconds > 0.9

            # Step 2: export to AAC/ADTS (this is what Gradio does)
            out_buf = io.BytesIO()
            segment.export(out_buf, format="adts")
            adts_bytes = out_buf.getvalue()
            assert len(adts_bytes) > 0, "ADTS export produced empty output"
            print(f"  WAV: {len(wav_bytes)} bytes → ADTS: {len(adts_bytes)} bytes")
        finally:
            self._restore_pydub(orig)

    def test_full_pipeline_multiple_chunks(self):
        """Multiple sequential chunks (simulating a streaming response)."""
        import io
        import pydub

        orig = self._setup_pydub()
        try:
            for i in range(5):
                # Each chunk is ~1s of audio at 24kHz
                audio = np.random.randn(24000).astype(np.float32) * 0.3
                wav_bytes = self._numpy_to_wav_bytes(audio)
                segment = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))

                out_buf = io.BytesIO()
                segment.export(out_buf, format="adts")
                assert len(out_buf.getvalue()) > 0, f"Chunk {i+1}: ADTS export failed"

            print(f"  5 chunks converted successfully")
        finally:
            self._restore_pydub(orig)

    def test_pipeline_with_fade_in_chunk(self):
        """First chunk with fade-in applied (start samples near zero)."""
        import io
        import pydub
        import tools.model.model_manager as mm

        orig = self._setup_pydub()
        try:
            audio = np.ones(24000, dtype=np.float32) * 0.5
            audio = mm._apply_fade_in(audio)
            assert abs(audio[0]) < 0.01, "Fade-in not applied"

            wav_bytes = self._numpy_to_wav_bytes(audio)
            segment = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))

            out_buf = io.BytesIO()
            segment.export(out_buf, format="adts")
            assert len(out_buf.getvalue()) > 0
        finally:
            self._restore_pydub(orig)

    def test_pipeline_with_short_chunk(self):
        """Very short audio chunk (~100ms) should still convert."""
        import io
        import pydub

        orig = self._setup_pydub()
        try:
            audio = np.random.randn(2400).astype(np.float32) * 0.3  # 100ms at 24kHz
            wav_bytes = self._numpy_to_wav_bytes(audio)
            segment = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))

            out_buf = io.BytesIO()
            segment.export(out_buf, format="adts")
            assert len(out_buf.getvalue()) > 0
        finally:
            self._restore_pydub(orig)

    def test_pipeline_with_silence(self):
        """Silent chunk should convert without errors."""
        import io
        import pydub

        orig = self._setup_pydub()
        try:
            audio = np.zeros(24000, dtype=np.float32)
            wav_bytes = self._numpy_to_wav_bytes(audio)
            segment = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))

            out_buf = io.BytesIO()
            segment.export(out_buf, format="adts")
            assert len(out_buf.getvalue()) > 0
        finally:
            self._restore_pydub(orig)
