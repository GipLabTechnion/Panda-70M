from scenedetect import detect, ContentDetector
import cv2
import os, re, json, argparse, time, glob
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import numpy as np
import sys
from pathlib import Path


def process_video_path(args):
    """Process a single video path in parallel, checking resolution and existing results"""
    root, file, min_dimension, existing_results, force_reprocess, worker_id = args
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    # Check if it's a video file
    if not any(file.lower().endswith(ext) for ext in video_extensions):
        return None

    video_path = os.path.join(root, file)
    video_name = os.path.basename(video_path)

    # Check if already processed
    if video_name in existing_results and not force_reprocess:
        print(f"[Worker {worker_id}] Skipping {video_name} - already processed")
        return None

    # Check resolution if required
    if min_dimension is not None:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if min(width, height) < min_dimension:
            print(f"[Worker {worker_id}] Skipping {file} - resolution {width}x{height} below minimum dimension {min_dimension}")
            return None

    return video_path


def parallel_get_video_files(root_dir, min_dimension=None, existing_results=None, force_reprocess=False, num_workers=None):
    """Parallel implementation of video file discovery and filtering"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Get all potential video files
    all_files = []
    for root, _, files in os.walk(root_dir):
        for i, file in enumerate(files):
            worker_id = i % num_workers
            all_files.append((root, file, min_dimension, existing_results, force_reprocess, worker_id))

    # Process files in parallel
    with Pool(num_workers) as pool:
        video_paths = pool.map(process_video_path, all_files)

    # Filter out None values and return valid paths
    valid_paths = [path for path in video_paths if path is not None]
    return valid_paths


def cutscene_detection(video_path, cutscene_threshold=25, max_cutscene_len=10, min_scene_len=15):
    scene_list = detect(video_path, ContentDetector(threshold=cutscene_threshold, min_scene_len=min_scene_len), start_in_scene=True)
    end_frame_idx = [0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for scene in scene_list:
        new_end_frame_idx = scene[1].get_frames()
        while (new_end_frame_idx - end_frame_idx[-1]) > (max_cutscene_len + 2) * fps:
            end_frame_idx.append(end_frame_idx[-1] + int(max_cutscene_len * fps))
        end_frame_idx.append(new_end_frame_idx)

    cutscenes = []
    for i in range(len(end_frame_idx) - 1):
        cutscenes.append([end_frame_idx[i], end_frame_idx[i + 1]])

    return cutscenes


class FileLock:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_path = str(file_path) + ".lock"
        self.lock_file = None

    def __enter__(self):
        # Try to acquire lock
        max_attempts = 60  # Maximum number of attempts (60 seconds timeout)
        attempts = 0
        while attempts < max_attempts:
            try:
                # Try to create the lock file
                self.lock_file = open(self.lock_path, 'x')
                return self
            except FileExistsError:
                # Lock file exists, wait and retry
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

            def repl_func(match: re.Match):
                return " ".join(match.group().split())

            json_str = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, json_str)
            json_str = re.sub(r'\[\s+', '[', json_str)
            json_str = re.sub(r'],\s+\[', '], [', json_str)
            json_str = re.sub(r'\s+\]', ']', json_str)

            f.write(json_str)


def get_temp_file_path(videos_root_dir, process_id):
    """Generate a temporary file path for a specific process"""
    return os.path.join(videos_root_dir, f"cutscene_frame_idx_temp_{process_id}.json")


def merge_temp_files(videos_root_dir, output_json_file):
    """Merge all temporary JSON files into the main output file"""
    # Find all temporary files
    temp_files = glob.glob(os.path.join(videos_root_dir, "cutscene_frame_idx_temp_*.json"))

    if not temp_files:
        return {}

    # Load and merge data from all temp files
    merged_data = safely_read_json(output_json_file)

    for temp_file in temp_files:
        temp_data = safely_read_json(temp_file)
        merged_data.update(temp_data)
        try:
            os.remove(temp_file)
        except OSError as e:
            print(f"Error removing temporary file {temp_file}: {e}")

    # Write merged data to output file
    safely_write_json(output_json_file, merged_data)
    return merged_data


def recover_from_crash(videos_root_dir, output_json_file):
    """Check for and recover from a previous crashed execution"""
    print("Checking for temporary files from previous crashed execution...")
    merged_data = merge_temp_files(videos_root_dir, output_json_file)
    if merged_data:
        print(f"Recovered data for {len(merged_data)} videos from previous execution")
    return merged_data


def process_single_video(video_path, temp_file_path, process_id, progress_dict,
                         cutscene_threshold=25, max_cutscene_len=5, min_scene_len=15):
    """Process a single video and write results to a temporary file"""
    try:
        cutscenes_raw = cutscene_detection(video_path, cutscene_threshold, max_cutscene_len, min_scene_len)
        print(f"[Worker {process_id}] Processed {video_path}")

        # Update temporary JSON file with results
        video_name = os.path.basename(video_path)
        temp_data = safely_read_json(temp_file_path)
        temp_data[video_name] = cutscenes_raw
        safely_write_json(temp_file_path, temp_data)

        result = (video_name, cutscenes_raw)
    except Exception as e:
        print(f"[Worker {process_id}] Error processing {video_path}: {str(e)}")
        result = (os.path.basename(video_path), [])

    # Update progress
    progress_dict['processed'] += 1
    sys.stderr.write(f'\rProcessed {progress_dict["processed"]}/{progress_dict["total"]} videos')
    sys.stderr.flush()

    return result


def get_videos_to_process(args):
    """Main function to get videos that need processing"""
    # Get output JSON file path
    output_json_file = os.path.join(args.videos_root_dir, "cutscene_frame_idx.json")

    # Check for and recover from previous crash
    recovered_data = recover_from_crash(args.videos_root_dir, output_json_file)

    # Load existing results (including any recovered data)
    existing_results = safely_read_json(output_json_file)
    print(f"Found {len(existing_results)} previously processed videos")

    # Get and filter video files in parallel
    video_paths = parallel_get_video_files(
        args.videos_root_dir,
        args.min_dimension,
        existing_results,
        args.force_reprocess,
        args.num_workers
    )

    if not video_paths:
        print(f"No valid video files found in {args.videos_root_dir}")
        sys.exit(1)

    return video_paths, existing_results, output_json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cutscene Detection")
    parser.add_argument("--videos-root-dir", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: number of CPU cores - 1)")
    parser.add_argument("--cutscene-threshold", type=float, default=25,
                        help="Threshold for cutscene detection (default: 25)")
    parser.add_argument("--max-cutscene-len", type=int, default=5,
                        help="Maximum length of cutscenes in seconds (default: 5)")
    parser.add_argument("--min-scene-len", type=int, default=15,
                        help="Minimum length of scenes in frames (default: 15)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to process (default: process all)")
    parser.add_argument("--min-dimension", type=int, default=None,
                        help="Minimum width/height dimension required for processing (default: None)")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Force reprocessing of videos even if they exist in the output JSON")
    args = parser.parse_args()

    # Get and filter videos to process
    video_paths, existing_results, output_json_file = get_videos_to_process(args)

    # Limit number of videos if specified
    if args.max_videos is not None:
        video_paths = video_paths[:args.max_videos]

    total_videos = len(video_paths)
    print(f"Found {total_videos} new videos to process")

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() - 1)

    # Create shared progress dictionary
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['processed'] = 0
    progress_dict['total'] = total_videos

    # Process videos in parallel with temporary files
    with Pool(num_workers) as pool:
        results = []
        for i, video_path in enumerate(video_paths):
            temp_file_path = get_temp_file_path(args.videos_root_dir, i % num_workers)
            result = pool.apply_async(process_single_video,
                                      args=(video_path,
                                            temp_file_path,
                                            i % num_workers,
                                            progress_dict),
                                      kwds={'cutscene_threshold': args.cutscene_threshold,
                                            'max_cutscene_len': args.max_cutscene_len,
                                            'min_scene_len': args.min_scene_len})
            results.append(result)

        # Get all results
        results = [r.get() for r in results]

    # Merge all temporary files into the final output file
    print("\nMerging temporary files...")
    merge_temp_files(args.videos_root_dir, output_json_file)
    print("Done!")