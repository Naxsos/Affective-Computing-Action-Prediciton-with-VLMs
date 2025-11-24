# Affective-Computing-Action-Prediciton-with-VLMs
University Project for practical Course WS25/26

## Video Frame Extraction Tool

This repository includes a Python utility to extract frames from videos before a specific timestamp.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Extract n frames before a specific timestamp from a video:

```bash
python extract_frames.py <video_path> <timestamp_seconds> <num_frames> [--output <output_dir>]
```

#### Arguments:
- `video_path`: Path to your input video file
- `timestamp_seconds`: The timestamp in seconds (e.g., 10.5)
- `num_frames`: Number of frames to extract before the timestamp
- `--output` or `-o`: (Optional) Base output directory for frames (default: "output_frames")
- `--interval` or `-i`: (Optional) Extract every Nth frame going backwards (default: 1, consecutive frames)
- `--no-subfolder`: (Optional) Disable automatic timestamped subfolder creation (will overwrite existing files)

#### Examples:

Extract 10 consecutive frames before the 30-second mark:
```bash
python3 extract_frames.py my_video.mp4 30 10
```

Extract 20 frames, taking every 5th frame going backwards from 1:44 (104 seconds):
```bash
python3 extract_frames.py my_video.mp4 104 20 --interval 5
```

Extract frames with a custom output directory:
```bash
python3 extract_frames.py my_video.mp4 15.5 5 --output my_frames
```

Extract frames without auto-subfolder (will overwrite existing files):
```bash
python3 extract_frames.py my_video.mp4 30 10 --no-subfolder
```

### Output

#### Auto-Subfolder Organization (Default)
By default, each extraction creates a timestamped subfolder to prevent overwriting previous extractions:

**Folder naming format:** `{video_name}_t{timestamp}s_n{frames}_i{interval}_{datetime}/`

**Example:** `output_frames/P01_01_t104.0s_n20_i5_20251124_111814/`
- `P01_01` - Video name
- `t104.0s` - Target timestamp
- `n20` - Number of frames
- `i5` - Frame interval
- `20251124_111814` - Extraction date/time

#### Frame Files
The extracted frames are saved as JPEG images with filenames that include:
- Frame number (e.g., `frame_006138`)
- Timestamp in seconds (e.g., `time_102.403s`)

**Example:** `frame_006138_time_102.403s.jpg`

This organization allows you to run multiple extractions without losing previous results!

## Frame Analysis with OpenAI Vision API

Analyze extracted frames using OpenAI's Vision models (GPT-4o, GPT-4o-mini, GPT-4-turbo).

### Setup

1. Get your OpenAI API key from [platform.openai.com](https://platform.openai.com)
2. Set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Alternatively, you can pass it directly with the `--api-key` argument.

### Usage

```bash
python3 analyze_frames.py <image_directory> [options]
```

#### Arguments:
- `image_directory`: Directory containing the extracted frames
- `-p` or `--prompt`: (Optional) Your analysis prompt (default: "Describe what is happening in this sequence of images. What action is the person performing?")
- `-m` or `--model`: (Optional) Model to use: `gpt-4o` (default), `gpt-4o-mini`, `gpt-4-turbo`
- `-n` or `--max-images`: (Optional) Maximum number of images to analyze (default: all)
- `-k` or `--api-key`: (Optional) OpenAI API key (overrides environment variable)
- `-d` or `--detail`: (Optional) Image detail level: `low`, `high`, or `auto` (default)
- `-o` or `--output`: (Optional) Save response to a file

#### Examples:

Analyze all frames with default prompt:
```bash
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814
```

Analyze with custom prompt:
```bash
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814 --prompt "What action is the person performing?"
```

Analyze only the first 10 frames with a specific model:
```bash
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814 --prompt "Describe the emotion and behavior" --model gpt-4o-mini --max-images 10
```

Save the response to a file:
```bash
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814 --prompt "What is happening?" --output analysis_result.txt
```

Use high-detail mode for better accuracy:
```bash
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814 --prompt "Analyze facial expressions" --detail high
```

### Complete Workflow Example

```bash
# Step 1: Extract frames from video
python3 extract_frames.py P01_01.MP4 104 20 --interval 5

# Step 2: Analyze the extracted frames (uses default prompt)
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814 --output result.txt

# Or with custom prompt:
python3 analyze_frames.py output_frames/P01_01_t104.0s_n20_i5_20251124_111814 --prompt "What action is being performed?" --output result.txt
```
