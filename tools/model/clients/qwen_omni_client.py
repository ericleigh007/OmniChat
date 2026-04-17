"""Typed HTTP transport client for OpenAI-compatible Qwen Omni services."""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

import requests


@dataclass(frozen=True)
class QwenOmniClientConfig:
    base_url: str
    api_key: Optional[str] = None
    model_name: str = ""
    timeout_s: float = 120.0
    endpoints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QwenStreamEvent:
    text_delta: str = ""
    audio_delta: Optional[bytes] = None
    modality: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_chunk: Optional[dict[str, Any]] = None


class QwenOmniClient:
    """Remote transport wrapper for the current Qwen OpenAI-compatible path."""

    def __init__(self, config: QwenOmniClientConfig, session: Optional[requests.Session] = None) -> None:
        self._config = config
        self._session = session or requests.Session()

    @property
    def session(self) -> requests.Session:
        return self._session

    @property
    def config(self) -> QwenOmniClientConfig:
        return self._config

    def list_models(self) -> list[dict[str, Any]]:
        response = self._session.get(
            self._url("models"),
            headers=self._auth_headers(),
            timeout=self._config.timeout_s,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json().get("data", [])

    def resolve_model_name(self) -> str:
        if self._config.model_name:
            return self._config.model_name
        models = self.list_models()
        if models:
            return models[0].get("id", "")
        return ""

    def healthcheck(self) -> dict[str, Any]:
        endpoint = self._config.endpoints.get("health", "health")
        try:
            response = self._session.get(
                self._url("health"),
                headers=self._auth_headers(),
                timeout=min(self._config.timeout_s, 10.0),
            )
            return {
                "ok": response.status_code < 400,
                "status_code": response.status_code,
                "endpoint": endpoint,
            }
        except requests.RequestException as exc:
            return {
                "ok": False,
                "status_code": None,
                "endpoint": endpoint,
                "error": str(exc),
            }

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._session.post(
            self._url("chat_completions"),
            headers=self._json_headers(),
            json=payload,
            timeout=self._config.timeout_s,
            stream=False,
        )
        response.raise_for_status()
        return response.json()

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[QwenStreamEvent]:
        with self._session.post(
            self._url("chat_completions"),
            headers=self._json_headers(),
            json=payload,
            timeout=(min(self._config.timeout_s, 10.0), None),
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                yield self._parse_chunk(chunk)

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        data = {"model": self.resolve_model_name()}
        if prompt:
            data["prompt"] = prompt
        if language:
            data["language"] = language

        response = self._session.post(
            self._url("audio_transcriptions"),
            headers=self._auth_headers(),
            data=data,
            files={"file": (filename, audio_bytes, self._guess_mime_type(filename))},
            timeout=self._config.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def transcribe_audio_file(
        self,
        file_path: str,
        *,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        with open(file_path, "rb") as handle:
            return self.transcribe_audio_bytes(
                handle.read(),
                filename=os.path.basename(file_path) or "audio.wav",
                prompt=prompt,
                language=language,
            )

    def list_voices(self) -> list[str]:
        response = self._session.get(
            self._resolve_url(self._config.base_url.rstrip("/"), self._config.endpoints.get("audio_voices", "audio/voices")),
            headers=self._auth_headers(),
            timeout=self._config.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        voices = payload.get("voices", [])
        return [voice for voice in voices if isinstance(voice, str)]

    def synthesize_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        task_type: str = "CustomVoice",
        instructions: Optional[str] = None,
        response_format: str = "wav",
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: Optional[bool] = None,
    ) -> tuple[bytes, str]:
        payload = self._build_speech_payload(
            text,
            voice=voice,
            task_type=task_type,
            instructions=instructions,
            response_format=response_format,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            stream=False,
        )
        response = self._session.post(
            self._url("audio_speech"),
            headers=self._json_headers(),
            json=payload,
            timeout=self._config.timeout_s,
            stream=False,
        )
        response.raise_for_status()
        self._raise_for_json_error(response)
        return response.content, response.headers.get("content-type", "")

    def stream_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        task_type: str = "CustomVoice",
        instructions: Optional[str] = None,
        response_format: str = "pcm",
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: Optional[bool] = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        payload = self._build_speech_payload(
            text,
            voice=voice,
            task_type=task_type,
            instructions=instructions,
            response_format=response_format,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            stream=True,
        )
        with self._session.post(
            self._url("audio_speech"),
            headers=self._json_headers(),
            json=payload,
            timeout=(min(self._config.timeout_s, 10.0), None),
            stream=True,
        ) as response:
            response.raise_for_status()
            self._raise_for_json_error(response)
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk

    def get_transport_contract(self) -> dict[str, Any]:
        endpoints = dict(self._config.endpoints)
        return {
            "protocol": "http+sse",
            "base_url": self._config.base_url.rstrip("/"),
            "endpoints": endpoints,
            "stream_shape": {
                "text": "SSE delta content",
                "audio": "reserved for future delta bytes",
            },
            "realtime_ws_url": self.build_realtime_ws_url(),
        }

    def build_realtime_ws_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        if base_url.startswith("https://"):
            ws_base = "wss://" + base_url[len("https://"):]
        elif base_url.startswith("http://"):
            ws_base = "ws://" + base_url[len("http://"):]
        else:
            ws_base = base_url
        realtime_path = self._config.endpoints.get("realtime", "realtime")
        return self._resolve_url(ws_base, realtime_path, ws_scheme=True)

    def open_realtime_connection(self, *, extra_headers: Optional[list[str]] = None, subprotocols: Optional[list[str]] = None):
        import websocket

        headers = self._websocket_headers()
        if extra_headers:
            headers.extend(extra_headers)

        return websocket.create_connection(
            self.build_realtime_ws_url(),
            header=headers,
            timeout=self._config.timeout_s,
            subprotocols=subprotocols or ["realtime"],
        )

    def perform_realtime_handshake(self, *, session: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ws = None
        model_name = self.resolve_model_name()
        request = {
            "type": "session.update",
            "session": session or {
                "model": model_name,
                "modalities": ["text"],
                "instructions": "OmniChat transport probe",
            },
        }

        try:
            ws = self.open_realtime_connection(subprotocols=["realtime", "openai-realtime-v1"])
            ws.send(json.dumps(request))
            raw_event = ws.recv()
            event = json.loads(raw_event) if isinstance(raw_event, str) else raw_event
            event_type = event.get("type") if isinstance(event, dict) else None
            return {
                "ok": isinstance(event, dict) and event_type not in {"error", None},
                "url": self.build_realtime_ws_url(),
                "request": request,
                "event": event,
                "event_type": event_type,
            }
        except Exception as exc:
            return {
                "ok": False,
                "url": self.build_realtime_ws_url(),
                "request": request,
                "error": str(exc),
            }
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    def _parse_chunk(self, chunk: dict[str, Any]) -> QwenStreamEvent:
        choices = chunk.get("choices", [])
        if not choices:
            return QwenStreamEvent(modality=chunk.get("modality"), raw_chunk=chunk)

        modality = chunk.get("modality")
        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")
        text_delta = ""
        audio_delta = None

        audio_info = delta.get("audio")
        if isinstance(audio_info, dict):
            encoded = audio_info.get("data")
            if isinstance(encoded, str):
                audio_delta = self._decode_audio(encoded)

        if modality == "audio":
            if audio_delta is None:
                audio_delta = self._decode_audio(delta.get("content"))
        else:
            text_delta = self._extract_text(delta.get("content"))

        return QwenStreamEvent(
            text_delta=text_delta,
            audio_delta=audio_delta,
            modality=modality,
            finish_reason=finish_reason,
            raw_chunk=chunk,
        )

    def _decode_audio(self, content: Any) -> Optional[bytes]:
        if not isinstance(content, str) or not content:
            return None
        try:
            return base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError):
            return None

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    def _json_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self._auth_headers())
        return headers

    def _build_speech_payload(
        self,
        text: str,
        *,
        voice: Optional[str],
        task_type: str,
        instructions: Optional[str],
        response_format: str,
        language: Optional[str],
        ref_audio: Optional[str],
        ref_text: Optional[str],
        x_vector_only_mode: Optional[bool],
        stream: bool,
    ) -> dict[str, Any]:
        payload = {
            "model": self.resolve_model_name(),
            "input": text,
            "task_type": task_type,
            "response_format": response_format,
            "stream": stream,
        }
        if voice:
            payload["voice"] = voice
        if instructions:
            payload["instructions"] = instructions
        if language:
            payload["language"] = language
        if ref_audio:
            payload["ref_audio"] = ref_audio
        if ref_text:
            payload["ref_text"] = ref_text
        if x_vector_only_mode is not None:
            payload["x_vector_only_mode"] = x_vector_only_mode
        return payload

    def _raise_for_json_error(self, response: requests.Response) -> None:
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return
        try:
            payload = response.json()
        except ValueError:
            return
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or payload
            raise RuntimeError(str(message))
        raise RuntimeError(str(payload))

    def _auth_headers(self) -> dict[str, str]:
        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _websocket_headers(self) -> list[str]:
        headers = []
        if self._config.api_key:
            headers.append(f"Authorization: Bearer {self._config.api_key}")
        return headers

    def _url(self, endpoint_key: str) -> str:
        endpoint = self._config.endpoints.get(endpoint_key, endpoint_key)
        return self._resolve_url(self._config.base_url.rstrip("/"), endpoint)

    def _resolve_url(self, base_url: str, endpoint: str, *, ws_scheme: bool = False) -> str:
        if endpoint.startswith(("http://", "https://", "ws://", "wss://")):
            if ws_scheme and endpoint.startswith(("http://", "https://")):
                return self._swap_ws_scheme(endpoint)
            return endpoint
        if endpoint.startswith("/"):
            return f"{self._origin(base_url, ws_scheme=ws_scheme)}{endpoint}"
        return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _origin(self, url: str, *, ws_scheme: bool = False) -> str:
        parts = urlsplit(url)
        scheme = parts.scheme
        if ws_scheme:
            scheme = "wss" if scheme == "https" else "ws"
        return urlunsplit((scheme, parts.netloc, "", "", "")).rstrip("/")

    def _swap_ws_scheme(self, url: str) -> str:
        parts = urlsplit(url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))

    def _guess_mime_type(self, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".wav"):
            return "audio/wav"
        if lower.endswith(".mp3"):
            return "audio/mpeg"
        if lower.endswith(".flac"):
            return "audio/flac"
        if lower.endswith(".m4a"):
            return "audio/mp4"
        if lower.endswith(".ogg") or lower.endswith(".oga"):
            return "audio/ogg"
        return "application/octet-stream"
