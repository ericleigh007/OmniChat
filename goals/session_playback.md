# Goal: Session Playback

## Objective
Review a recorded RT conversation session, rebuild the visible transcript, sync playback overlays from manifest timing, and optionally export demo artifacts.

## Trigger
User selects a saved `session.json` from the Playback tab, uses `Open JSON`, or reopens a previously recorded session from `outputs/sessions/`.

## Inputs
- **Session manifest**: `outputs/sessions/<session-id>/session.json`
- **Optional session video**: `session.mp4`
- **Optional structured audio artifacts**: `turn_XXX_user.mp3`, `turn_XXX_model.mp3`
- **Playback state**: selected event index, playback position, export destination

## Process

### 1. Discover and load session metadata
- Scan the configured sessions root for JSON manifests with turn data
- Load the selected manifest into `SessionPlaybackManifest`
- Resolve optional video and per-turn audio artifact paths relative to the session directory

### 2. Rebuild transcript history
- Convert manifest turns into replay events with `build_replay_events(...)`
- Reconstruct visible chat history with `build_chat_history(...)`
- Render the transcript with the shared chat HTML renderer so playback mirrors live chat formatting

### 3. Drive synced playback state
- Use manifest timing offsets to compute the active turn and playback phase
- Update the overlay with turn number, modality, prompt tokens, response tokens, first-text timing, and completion timing
- Advance through replay events manually or by timed playback progression

### 4. Export demo artifacts
- Stitch recorded turn audio into a single exportable audio artifact when audio clips are present
- Export a demo-ready video artifact or open the recorded `session.mp4` directly when available

## Tools Used
- `tools/shared/session_playback.py` — discovery, manifest loading, replay timeline, overlay snapshots, exports
- `tools/shared/chat_render.py` — transcript HTML rendering shared with live chat
- `rt_app.py` — Playback tab controls, embedded video host, overlay updates, export actions

## Edge Cases
- Missing or invalid JSON → reject the selection as a non-session manifest
- Missing video artifact → keep transcript playback available and disable embedded/open-video actions
- Audio omitted by recording policy → keep replay timing and transcript available without stitched-audio export inputs
- Transcript omitted by recording policy → allow video-first playback with overlay stats even if visible text is sparse