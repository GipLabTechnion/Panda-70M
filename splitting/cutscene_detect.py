from scenedetect import detect, ContentDetector
import cv2
import os, re, json, argparse
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import numpy as np
import sys


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
        result = (video_path.split("/")[-1], cutscenes_raw)
    except Exception as e:
        print(f"Worker {process_id}: Error processing {video_path}: {str(e)}")
        result = (video_path.split("/")[-1], [])

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
    parser.add_argument("--input-video-list", type=str, required=True,
                        help="Input file containing list of video paths")
    parser.add_argument("--output-video-list", type=str, required=True,
                        help="Output file to save processed video paths")
    parser.add_argument("--output-json-file", type=str, default="cutscene_frame_idx.json")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: number of CPU cores - 1)")
    parser.add_argument("--cutscene-threshold", type=float, default=10,
                        help="Threshold for cutscene detection (default: 25)")
    parser.add_argument("--max-cutscene-len", type=float, default=5,
                        help="Maximum length of cutscenes in seconds (default: 5)")
    parser.add_argument("--min-scene-len", type=float, default=15,
                        help="Minimum length of scenes in frames (default: 15)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum number of videos to process (default: process all)")
    args = parser.parse_args()

    # Read video paths
    with open(args.input_video_list, "r") as f:
        video_paths = f.read().splitlines()

    # Limit number of videos if specified
    if args.max_videos is not None:
        video_paths = video_paths[:args.max_videos]

    # Save the list of videos to be processed
    with open(args.output_video_list, "w") as f:
        f.write("\n".join(video_paths))
    print(f"Saved video list to process to {args.output_video_list}")

    total_videos = len(video_paths)
    print(f"Found {total_videos} videos to process")

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

    print(f"Writing results to {args.output_json_file}")
    write_json_file(video_cutscenes, args.output_json_file)
    print("Done!")