# spaceaudio

SpaceAudio translates a movie subtitle and produces a speech from it. It uses various Transformer libraries for translation and Mozilla TTS for speech generation.

## Demo
I've created audio subtitles in 16 different languages for Avatar: The Way of Water and put them on the [website](https://www.spaceaudio.xyz).

## Installation
You need python (3.8, 3.9, or 3.10) and [ffmpeg](https://ffmpeg.org).
Also I recommend you have [Anaconda](https://www.anaconda.com).

```bash
pip install -r requirements.txt
```

## Usage
```bash
main.py -n <name> -p <pathToSubtitle> -l <language> -o <output>
```
