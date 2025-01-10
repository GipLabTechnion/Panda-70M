import os, json, argparse, shutil
from datetime import datetime
import subprocess
from tqdm import tqdm


def get_video_files(root_dir):
    """Recursively find all video files in the given directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


def clear_directory(directory):
    """Remove all contents of a directory"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Splitting")
    parser.add_argument("--videos-root-dir", type=str, required=True,
                        help="Root directory containing video files and event_timecode.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split video clips")
    parser.add_argument("--clear-output", action='store_true',
                        help="Clear output directory before starting")
    args = parser.parse_args()

    # Check for event_timecode.json
    event_json_path = os.path.join(args.videos_root_dir, "event_timecode.json")
    if not os.path.exists(event_json_path):
        print(f"Error: event_timecode.json not found in {args.videos_root_dir}")
        exit(1)

    # Get list of video files
    video_paths = get_video_files(args.videos_root_dir)
    if not video_paths:
        print(f"No video files found in {args.videos_root_dir}")
        exit(1)

    # Load event timecode data
    with open(event_json_path) as f:
        video_timecodes = json.load(f)

    # Create output directory and clear if requested
    os.makedirs(args.output_dir, exist_ok=True)
    if args.clear_output:
        print(f"Clearing output directory: {args.output_dir}")
        clear_directory(args.output_dir)

    for video_path in tqdm(video_paths):
        video_basename = os.path.basename(video_path)
        video_name = os.path.splitext(video_basename)[0]

        if video_basename not in video_timecodes:
            print(f"Warning: Skipping {video_basename} - no timecode data found")
            continue

        timecodes = video_timecodes[video_basename]

        for i, timecode in enumerate(timecodes):
            start_time = datetime.strptime(timecode[0], '%H:%M:%S.%f')
            end_time = datetime.strptime(timecode[1], '%H:%M:%S.%f')
            video_duration = (end_time - start_time).total_seconds()

            output_path = os.path.join(args.output_dir, f"{video_name}.{i}.mp4")

            # Use ffmpeg to split the video
            os.system(f"ffmpeg -hide_banner -loglevel panic -ss {timecode[0]} -t {video_duration:.3f} -i \"{video_path}\" \"{output_path}\"")

    print("\nDone!")