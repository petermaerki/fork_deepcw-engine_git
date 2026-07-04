cd expamples/python
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt 


ffmpeg -y -i examples/python/ka9q-recording-2026-07-02T14-37-30.webm -acodec pcm_s16le -ar 16000 -ac 1 examples/python/ka9q-recording-2026-07-02T14-37-30.wav && ffprobe -v error -show_entries stream=codec_name,sample_rate,channels -of default=noprint_wrappers=1 examples/python/ka9q-recording-2026-07-02T14-37-30.wav

