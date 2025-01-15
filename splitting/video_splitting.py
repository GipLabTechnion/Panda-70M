import os, json, argparse, shutil, time
from datetime import datetime
import subprocess
from tqdm import tqdm


class FileLock:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_path = str(file_path) + ".lock"
        self.lock_file = None

    def __enter__(self):
        max_attempts = 60
        attempts = 0
        while attempts < max_attempts:
            try:
                self.lock_file = open(self.lock_path, 'x')
                return self
            except FileExistsError:
                time.sleep(1)
                attempts += 1
                if attempts >= max_attempts:
                    raise TimeoutError(f"Could not acquire lock for {self.file_path} after {max_attempts} seconds")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            self.lock_file.close()
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass


def safely_read_json(filepath):
    """Read JSON file with file locking"""
    if not os.path.exists(filepath):
        return {}

    with FileLock(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}


def safely_write_json(filepath, data):
    """Write JSON file with file locking"""
    with FileLock(filepath):
        with open(filepath, 'w') as f:
            json_str = json.dumps(data, indent=4)
            f.write(json_str)


def update_progress_json(progress_file, video_name, segment_idx=None, success=True):
    """Update progress JSON file with completed segments"""
    current_data = safely_read_json(progress_file)

    if video_name not in current_data:
        current_data[video_name] = {"completed_segments": [], "failed_segments": []}

    if segment_idx is not None:
        if success:
            if segment_idx not in current_data[video_name]["completed_segments"]:
                current_data[video_name]["completed_segments"].append(segment_idx)
        else:
            if segment_idx not in current_data[video_name]["failed_segments"]:
                current_data[video_name]["failed_segments"].append(segment_idx)

    safely_write_json(progress_file, current_data)


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


def split_video(video_path, timecodes, output_dir, progress_file, force_reprocess=False):
    """Split a video into segments based on timecodes"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    progress_data = safely_read_json(progress_file)
    video_progress = progress_data.get(video_name, {"completed_segments": [], "failed_segments": []})

    for i, timecode in enumerate(timecodes):
        # Skip if segment already processed successfully and not forcing reprocess
        if i in video_progress["completed_segments"] and not force_reprocess:
            print(f"Skipping {video_name}.{i}.mp4 - already processed")
            continue

        start_time = datetime.strptime(timecode[0], '%H:%M:%S.%f')
        end_time = datetime.strptime(timecode[1], '%H:%M:%S.%f')
        video_duration = (end_time - start_time).total_seconds()
        output_path = os.path.join(output_dir, f"{video_name}.{i}.mp4")

        try:
            # Use ffmpeg to split the video
            result = os.system(f'ffmpeg -hide_banner -loglevel panic -ss {timecode[0]} -t {video_duration:.3f} -i "{video_path}" "{output_path}"')

            if result == 0:
                update_progress_json(progress_file, video_name, i, True)
                print(f"Successfully extracted {video_name}.{i}.mp4")
            else:
                update_progress_json(progress_file, video_name, i, False)
                print(f"Failed to extract {video_name}.{i}.mp4")
        except Exception as e:
            update_progress_json(progress_file, video_name, i, False)
            print(f"Error extracting {video_name}.{i}.mp4: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Splitting")
    parser.add_argument("--videos-root-dir", type=str, required=True,
                        help="Root directory containing video files and event_timecode.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split video clips")
    parser.add_argument("--clear-output", action='store_true',
                        help="Clear output directory before starting")
    parser.add_argument("--force-reprocess", action='store_true',
                        help="Force reprocessing of already extracted segments")
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

    # Progress tracking file in videos root directory
    progress_file = os.path.join(args.videos_root_dir, "extraction_progress.json")
    progress_data = safely_read_json(progress_file)

    # Count total and completed segments
    total_segments = sum(len(timecodes) for timecodes in video_timecodes.values())
    completed_segments = sum(len(data["completed_segments"])
                             for data in progress_data.values())

    print(f"Found {total_segments} total segments")
    print(f"Previously completed {completed_segments} segments")

    # Process videos
    for video_path in tqdm(video_paths):
        video_basename = os.path.basename(video_path)

        if video_basename not in video_timecodes:
            print(f"Warning: Skipping {video_basename} - no timecode data found")
            continue

        timecodes = video_timecodes[video_basename]
        if not timecodes:  # Skip videos with no events
            print(f"Skipping {video_basename} - no events to extract")
            continue

        print(f"\nProcessing {video_basename} - {len(timecodes)} segments")
        split_video(video_path, timecodes, args.output_dir, progress_file, args.force_reprocess)

    print("\nDone!")