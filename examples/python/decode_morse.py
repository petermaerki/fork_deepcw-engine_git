from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

MIN_AUDIO_SECONDS = 5.0
MAX_AUDIO_SECONDS = 20.0


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    """Read PCM WAV audio and return mono float32 samples in the range [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, audio.astype(np.float32, copy=False)


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with linear interpolation to keep the example dependency-light."""
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
    """Create a [1, 1, time, frequency] float32 spectrogram for the ONNX model."""
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
    """Collapse a CTC best path by removing blanks and repeated labels."""
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


def decode(model_path: Path, metadata_path: Path, wav_path: Path) -> str:
    metadata = load_metadata(metadata_path)
    target_rate = int(metadata["sample_rate"])
    source_rate, audio = read_wav_mono(wav_path)
    audio = resample_linear(audio, source_rate, target_rate)

    duration_seconds = len(audio) / target_rate
    if duration_seconds < MIN_AUDIO_SECONDS or duration_seconds > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio duration must be between {MIN_AUDIO_SECONDS:.0f} and {MAX_AUDIO_SECONDS:.0f} seconds "
            f"after resampling to {target_rate} Hz. Got {duration_seconds:.2f} seconds."
        )

    spectrogram = audio_to_spectrogram(audio, metadata)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = session.run([metadata["onnx_output_name"]], {metadata["onnx_input_name"]: spectrogram})
    return greedy_ctc_decode(outputs[0], list(metadata["chars"]), int(metadata["blank_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Morse code audio with the DeepCW ONNX model.")
    parser.add_argument("--model", type=Path, default=Path("../../model.onnx"), help="Path to model.onnx.")
    parser.add_argument("--metadata", type=Path, default=Path("../../model.onnx.json"), help="Path to model metadata JSON.")
    parser.add_argument("--wav", type=Path, required=True, help="Path to a 5-20 second PCM WAV file.")
    args = parser.parse_args()

    text = decode(args.model, args.metadata, args.wav)
    print(text)


if __name__ == "__main__":
    main()
