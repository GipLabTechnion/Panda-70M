import os, json, argparse, shutil, time
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import subprocess


def get_video_files(root_dir):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def update_progress(progress_dict, video_name, segment_idx=None, success=True):
    if video_name not in progress_dict:
        progress_dict[video_name] = {"completed": set(), "failed": set()}

    if segment_idx is not None:
        if success:
            progress_dict[video_name]["completed"].add(segment_idx)
        else:
            progress_dict[video_name]["failed"].add(segment_idx)


def split_video_segment(args):
    video_path, timecode, output_path, i = args
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    start_time = datetime.strptime(timecode[0], '%H:%M:%S.%f')
    end_time = datetime.strptime(timecode[1], '%H:%M:%S.%f')
    video_duration = (end_time - start_time).total_seconds()

    try:
        result = subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'panic',
            '-ss', timecode[0],
            '-t', f'{video_duration:.3f}',
            '-i', video_path,
            '-c', 'copy',
            output_path
        ], capture_output=True)

        success = result.returncode == 0
        return i, success, video_name
    except Exception as e:
        return i, False, video_name


def process_video(video_info, output_dir, progress_dict, force_reprocess=False):
    video_path, timecodes = video_info
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    video_progress = progress_dict.get(video_name, {"completed": set(), "failed": set()})

    segments_to_process = []
    for i, timecode in enumerate(timecodes):
        if i in video_progress["completed"] and not force_reprocess:
            continue

        output_path = os.path.join(output_dir, f"{video_name}.{i}.mp4")
        segments_to_process.append((video_path, timecode, output_path, i))

    return segments_to_process


def parallel_video_splitting(video_paths, video_timecodes, output_dir, progress_dict, num_workers=None, force_reprocess=False):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    all_segments = []
    for video_path in video_paths:
        video_basename = os.path.basename(video_path)
        if video_basename not in video_timecodes:
            continue

        timecodes = video_timecodes[video_basename]
        if not timecodes:
            continue

        segments = process_video((video_path, timecodes), output_dir, progress_dict, force_reprocess)
        all_segments.extend(segments)

    if not all_segments:
        return

    print(f"\nProcessing {len(all_segments)} segments using {num_workers} workers")
    with Pool(num_workers) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(split_video_segment, all_segments), total=len(all_segments)):
            segment_idx, success, video_name = result
            update_progress(progress_dict, video_name, segment_idx, success)
            results.append((video_name, segment_idx, success))

    success_count = sum(1 for _, _, success in results if success)
    print(f"\nCompleted: {success_count}/{len(all_segments)} segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Video Splitting")
    parser.add_argument("--videos-root-dir", type=str, required=True,
                        help="Root directory containing video files and event_timecode.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split video clips")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: CPU cores - 1)")
    parser.add_argument("--clear-output", action='store_true',
                        help="Clear output directory before starting")
    parser.add_argument("--force-reprocess", action='store_true',
                        help="Force reprocessing of already extracted segments")
    args = parser.parse_args()

    event_json_path = os.path.join(args.videos_root_dir, "event_timecode.json")
    if not os.path.exists(event_json_path):
        print(f"Error: event_timecode.json not found in {args.videos_root_dir}")
        exit(1)

    video_paths = get_video_files(args.videos_root_dir)
    if not video_paths:
        print(f"No video files found in {args.videos_root_dir}")
        exit(1)

    with open(event_json_path) as f:
        video_timecodes = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.clear_output:
        print(f"Clearing output directory: {args.output_dir}")
        clear_directory(args.output_dir)

    manager = Manager()
    progress_dict = manager.dict()

    parallel_video_splitting(
        video_paths,
        video_timecodes,
        args.output_dir,
        progress_dict,
        args.num_workers,
        args.force_reprocess
    )

    print("\nDone!")