from scenedetect import detect, ContentDetector
import cv2
import os, re, json, argparse
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import numpy as np
import sys


def get_video_resolution(video_path):
    """Get video resolution using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def get_video_files(root_dir, min_dimension=None):
    """Recursively find all video files in the given directory with optional resolution filtering"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                if min_dimension is not None:
                    width, height = get_video_resolution(video_path)
                    if min(width, height) < min_dimension:
                        print(f"Skipping {file} - resolution {width}x{height} below minimum dimension {min_dimension}")
                        continue
                video_files.append(video_path)

    return video_files


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


def process_single_video(video_path, process_id, progress_dict, cutscene_threshold=25, max_cutscene_len=5, min_scene_len=15):
    """Process a single video"""
    try:
        cutscenes_raw = cutscene_detection(video_path, cutscene_threshold, max_cutscene_len, min_scene_len)
        print(f"Worker {process_id}: Processed {video_path}")
        result = (os.path.basename(video_path), cutscenes_raw)
    except Exception as e:
        print(f"Worker {process_id}: Error processing {video_path}: {str(e)}")
        result = (os.path.basename(video_path), [])

    # Update progress
    progress_dict['processed'] += 1
    sys.stderr.write(f'\rProcessed {progress_dict["processed"]}/{progress_dict["total"]} videos')
    sys.stderr.flush()

    return result


def write_json_file(data, output_file):
    data = json.dumps(data, indent=4)

    def repl_func(match: re.Match):
        return " ".join(match.group().split())

    data = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, data)
    data = re.sub(r'\[\s+', '[', data)
    data = re.sub(r'],\s+\[', '], [', data)
    data = re.sub(r'\s+\]', ']', data)
    with open(output_file, "w") as f:
        f.write(data)


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
    args = parser.parse_args()

    # Get all video files from the root directory with resolution filtering
    video_paths = get_video_files(args.videos_root_dir, args.min_dimension)

    if not video_paths:
        print(f"No valid video files found in {args.videos_root_dir}")
        sys.exit(1)

    # Limit number of videos if specified
    if args.max_videos is not None:
        video_paths = video_paths[:args.max_videos]

    total_videos = len(video_paths)
    print(f"Found {total_videos} valid videos to process")

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() - 1)

    # Create shared progress dictionary
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict['processed'] = 0
    progress_dict['total'] = total_videos

    # Create process function with fixed parameters
    process_func = partial(process_single_video,
                           cutscene_threshold=args.cutscene_threshold,
                           max_cutscene_len=args.max_cutscene_len,
                           min_scene_len=args.min_scene_len)

    # Process videos in parallel
    with Pool(num_workers) as pool:
        results = []
        for i, video_path in enumerate(video_paths):
            result = pool.apply_async(process_func, (video_path, i % num_workers, progress_dict))
            results.append(result)

        # Get all results
        results = [r.get() for r in results]

    print("\n")  # New line after progress

    # Convert results to dictionary
    video_cutscenes = dict(results)

    # Save results in the videos root directory
    output_json_file = os.path.join(args.videos_root_dir, "cutscene_frame_idx.json")
    print(f"Writing results to {output_json_file}")
    write_json_file(video_cutscenes, output_json_file)
    print("Done!")