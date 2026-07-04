from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - import guard for runtime setup
    raise SystemExit(
        "Missing dependency 'sounddevice'. Install with: uv pip install -r requirements.txt"
    ) from exc

MIN_AUDIO_SECONDS = 5.0
MAX_AUDIO_SECONDS = 20.0


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    target_length = int(round(len(audio) * target_rate / source_rate))
    source_positions = np.arange(target_length, dtype=np.float64) * source_rate / target_rate
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, len(audio) - 1)
    fraction = (source_positions - left).astype(np.float32)
    resampled = audio[left] * (1.0 - fraction) + audio[right] * fraction
    return resampled.astype(np.float32, copy=False)


def frequency_bin_range(sample_rate: int, fft_length: int, min_hz: float, max_hz: float) -> tuple[int, int]:
    bin_hz = sample_rate / fft_length
    start_bin = int(math.ceil(min_hz / bin_hz))
    stop_bin = int(math.floor(max_hz / bin_hz)) + 1
    return start_bin, stop_bin


def audio_to_spectrogram(audio: np.ndarray, metadata: dict) -> np.ndarray:
    fft_length = int(metadata["fft_length"])
    hop_length = int(metadata["hop_length"])
    sample_rate = int(metadata["sample_rate"])
    min_hz = float(metadata["spectrogram_min_freq_hz"])
    max_hz = float(metadata["spectrogram_max_freq_hz"])
    expected_bins = int(metadata["spectrogram_frequency_bins"])

    if len(audio) < fft_length:
        raise ValueError(f"Audio is too short for fft_length={fft_length}")

    start_bin, stop_bin = frequency_bin_range(sample_rate, fft_length, min_hz, max_hz)
    if stop_bin - start_bin != expected_bins:
        raise ValueError(f"Metadata expects {expected_bins} bins, but the computed range has {stop_bin - start_bin}")

    pad = fft_length // 2
    audio = np.pad(audio, (pad, pad), mode="reflect")
    window = np.hanning(fft_length + 1)[:-1].astype(np.float32)
    frames = 1 + (len(audio) - fft_length) // hop_length
    spectrogram = np.empty((frames, expected_bins), dtype=np.float32)

    for frame_index in range(frames):
        start = frame_index * hop_length
        frame = audio[start : start + fft_length] * window
        spectrum = np.fft.rfft(frame, n=fft_length)
        spectrogram[frame_index] = np.abs(spectrum[start_bin:stop_bin]).astype(np.float32)

    if metadata.get("normalization") == "log1p":
        spectrogram = np.log1p(spectrogram, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization: {metadata.get('normalization')}")

    return spectrogram[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)


def greedy_ctc_decode(log_probs: np.ndarray, chars: list[str], blank_index: int) -> str:
    best_path = log_probs[0].argmax(axis=-1)
    decoded: list[str] = []
    previous: int | None = None

    for index in best_path:
        index = int(index)
        if index == blank_index:
            previous = None
            continue
        if index != previous:
            decoded.append(chars[index])
        previous = index

    return "".join(decoded)


def decode_audio(session: ort.InferenceSession, metadata: dict, audio: np.ndarray, source_rate: int) -> str:
    target_rate = int(metadata["sample_rate"])
    audio = resample_linear(audio, source_rate, target_rate)
    duration_seconds = len(audio) / target_rate
    if duration_seconds < MIN_AUDIO_SECONDS or duration_seconds > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio duration must be between {MIN_AUDIO_SECONDS:.0f} and {MAX_AUDIO_SECONDS:.0f} seconds "
            f"after resampling to {target_rate} Hz. Got {duration_seconds:.2f} seconds."
        )
    spectrogram = audio_to_spectrogram(audio, metadata)
    outputs = session.run([metadata["onnx_output_name"]], {metadata["onnx_input_name"]: spectrogram})
    return greedy_ctc_decode(outputs[0], list(metadata["chars"]), int(metadata["blank_index"]))


def run_live(model_path: Path, metadata_path: Path, device: str | int | None, window_seconds: float, update_hz: float) -> None:
    metadata = load_metadata(metadata_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    # Stream capture rate can differ from model rate; decode path handles resampling.
    stream_rate = 48000
    chunk_size = int(stream_rate * 0.1)
    min_samples = int(stream_rate * MIN_AUDIO_SECONDS)
    max_samples = int(stream_rate * MAX_AUDIO_SECONDS)
    window_samples = int(stream_rate * max(MIN_AUDIO_SECONDS, min(MAX_AUDIO_SECONDS, window_seconds)))
    update_chunks = max(1, int(round(1.0 / (0.1 * update_hz))))

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    ring = np.zeros(max_samples, dtype=np.float32)
    write_pos = 0
    total_written = 0
    chunk_counter = 0
    last_text = ""

    def callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata.mean(axis=1, dtype=np.float32)
        audio_queue.put_nowait(mono)

    print("Listening... Press Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=stream_rate,
        blocksize=chunk_size,
        dtype="float32",
        channels=1,
        device=device,
        callback=callback,
    ):
        try:
            while True:
                chunk = audio_queue.get()
                n = len(chunk)
                if n >= max_samples:
                    print("slide!")
                    chunk = chunk[-max_samples:]
                    n = len(chunk)

                end = write_pos + n
                if end <= max_samples:
                    ring[write_pos:end] = chunk
                else:
                    split = max_samples - write_pos
                    ring[write_pos:] = chunk[:split]
                    ring[: end - max_samples] = chunk[split:]

                write_pos = (write_pos + n) % max_samples
                total_written += n
                chunk_counter += 1

                if chunk_counter % update_chunks != 0 or total_written < min_samples:
                    continue

                available = min(total_written, max_samples)
                use_samples = min(window_samples, available)
                start = (write_pos - use_samples) % max_samples
                if start < write_pos:
                    audio = ring[start:write_pos].copy()
                else:
                    audio = np.concatenate((ring[start:], ring[:write_pos]))

                text = decode_audio(session, metadata, audio, stream_rate)
                if text != last_text:
                    print(text)
                    last_text = text
        except KeyboardInterrupt:
            print("\nStopped.")


def parse_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Morse code from a live Linux microphone stream with the DeepCW ONNX model.")
    parser.add_argument("--model", type=Path, default=Path("../../model.onnx"), help="Path to model.onnx.")
    parser.add_argument("--metadata", type=Path, default=Path("../../model.onnx.json"), help="Path to model metadata JSON.")
    parser.add_argument("--device", type=str, default=None, help="Input device name or index for sounddevice.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=8.0,
        help="Seconds of recent audio used per decode (clamped to 5-20).",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=2.0,
        help="How often to run decoding (times per second).",
    )
    args = parser.parse_args()

    run_live(args.model, args.metadata, parse_device(args.device), args.window_seconds, args.update_hz)


if __name__ == "__main__":
    main()