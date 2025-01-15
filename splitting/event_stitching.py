import torch
from torchvision import transforms
from PIL import Image
import cv2
from multiprocessing import Pool, Manager
from collections import defaultdict
import os, sys, re, json, argparse, time
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

sys.path.append('ImageBind')
from models import imagebind_model
from models.imagebind_model import ModalityType


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


def update_json_with_result(output_file, video_name, events):
    """Update JSON file with new results"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            current_data = safely_read_json(output_file)
            current_data[video_name] = events
            safely_write_json(output_file, current_data)
            break
        except (TimeoutError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"Failed to update results for {video_name} after {max_retries} attempts: {str(e)}")
            time.sleep(1)


def get_video_files(root_dir):
    """Recursively find all video files in the given directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


def read_videoframe(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    res, frame = cap.read()
    if res:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
    return frame, res


def transfer_timecode(frameidx, fps):
    timecode = []
    for (start_idx, end_idx) in frameidx:
        s = str(timedelta(seconds=start_idx / fps, microseconds=1))[:-3]
        e = str(timedelta(seconds=end_idx / fps, microseconds=1))[:-3]
        timecode.append([s, e])
    return timecode


def extract_cutscene_feature(video_path, cutscenes, num_workers=8):
    features = torch.empty((0, 1024))
    res = []
    num_parallel_cutscene = 128
    for i in range(0, len(cutscenes), num_parallel_cutscene):
        cutscenes_sub = cutscenes[i:i + num_parallel_cutscene]
        frame_idx = []
        for cutscene in cutscenes_sub:
            start_frame_idx = int(0.95 * cutscene[0] + 0.05 * (cutscene[1] - 1))
            end_frame_idx = int(0.05 * cutscene[0] + 0.95 * (cutscene[1] - 1))
            frame_idx.extend([(video_path, start_frame_idx), (video_path, end_frame_idx)])

        with Pool(num_workers) as p:
            results = p.starmap(read_videoframe, frame_idx)
        frames = [image_transform(Image.fromarray(i[0])) for i in results]
        res.extend([i[1] for i in results])

        frames = torch.stack(frames, dim=0)
        with torch.no_grad():
            batch_features = model({ModalityType.VISION: frames.to(device)})[ModalityType.VISION]
        features = torch.vstack((features, batch_features.detach().cpu()))

    return features, res


def verify_cutscene(cutscenes, cutscene_feature, cutscene_status, transition_threshold=0.8):
    cutscenes_new = []
    cutscene_feature_new = []
    for i, cutscene in enumerate(cutscenes):
        start_frame_fet, end_frame_fet = cutscene_feature[2 * i], cutscene_feature[2 * i + 1]
        start_frame_res, end_frame_res = cutscene_status[2 * i], cutscene_status[2 * i + 1]
        diff = (start_frame_fet - end_frame_fet).pow(2).sum().sqrt()

        # Remove condition 1: start_frame or end_frame cannot be loaded
        if not (start_frame_res and end_frame_res):
            continue
        # Remove condition 2: cutscene include scene transition effect
        if diff > transition_threshold:
            continue

        cutscenes_new.append(cutscene)
        cutscene_feature_new.append([start_frame_fet, end_frame_fet])
    return cutscenes_new, cutscene_feature_new


def cutscene_stitching(cutscenes, cutscene_feature, eventcut_threshold=0.6):
    assert len(cutscenes) == len(cutscene_feature)
    num_cutscenes = len(cutscenes)

    # Add minimal fix for empty cutscenes
    if num_cutscenes == 0:
        return [], []

    events = []
    event_feature = []
    for i in range(num_cutscenes):
        # The first cutscene or the cutscene is discontinuous from the previous one => start a new event
        if i == 0 or cutscenes[i][0] != events[-1][-1]:
            events.append(cutscenes[i])
            event_feature.append(cutscene_feature[i])
            continue

        diff = (event_feature[-1][-1] - cutscene_feature[i][0]).pow(2).sum().sqrt()
        # The difference between the cutscene and the previous one is large => start a new event
        if diff > eventcut_threshold:
            events.append(cutscenes[i])
            event_feature.append(cutscene_feature[i])
        # Otherwise => extend the previous event
        else:
            events[-1].extend(cutscenes[i])
            event_feature[-1].extend(cutscene_feature[i])

    if len(events) > 0 and len(events[-1]) == 0:
        events.pop(-1)
        event_feature.pop(-1)

    return events, event_feature


def verify_event(events, event_feature, fps, min_event_len=1.5, max_event_len=60,
                 redundant_event_threshold=0.4, trim_begin_last_percent=0.1,
                 still_event_threshold=0.1, min_frames=None):
    assert len(events) == len(event_feature)
    num_events = len(events)

    events_final = []
    event_feature_final = torch.empty((0, 1024))

    min_event_len = min_event_len * fps
    max_event_len = max_event_len * fps

    for i in range(num_events):
        assert len(events[i]) == len(event_feature[i])
        # Remove condition 1: shorter than min_event_len
        if (events[i][-1] - events[i][0]) < min_event_len:
            continue

        # Remove condition 2: within-event difference is too small
        diff = (event_feature[i][0] - event_feature[i][-1]).pow(2).sum().sqrt()
        if diff < still_event_threshold:
            continue

        avg_feature = torch.stack(event_feature[i]).mean(axis=0)
        # Remove condition 3: too similar to the previous events
        diff = (event_feature_final - avg_feature).pow(2).sum(axis=1).sqrt()
        if torch.any(diff < redundant_event_threshold):
            continue

        # Trim the event if it is too long
        events[i][-1] = events[i][0] + min(int(max_event_len), (events[i][-1] - events[i][0]))

        # Calculate event length after trimming
        trim_len = int(trim_begin_last_percent * (events[i][-1] - events[i][0]))
        event_start = events[i][0] + trim_len
        event_end = events[i][-1] - trim_len

        # Check minimum frame requirement if specified
        if min_frames is not None and (event_end - event_start) < min_frames:
            continue

        events_final.append([event_start, event_end])
        event_feature_final = torch.vstack((event_feature_final, avg_feature))

    return events_final, event_feature_final


def process_video(video_path, cutscenes, min_frames=None, num_workers=8):
    """Process a single video and return its events"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        cutscene_raw_feature, cutscene_raw_status = extract_cutscene_feature(video_path, cutscenes, num_workers)
        cutscenes, cutscene_feature = verify_cutscene(cutscenes, cutscene_raw_feature, cutscene_raw_status, transition_threshold=0.45)
        events_raw, event_feature_raw = cutscene_stitching(cutscenes, cutscene_feature, eventcut_threshold=0.3)
        events, event_feature = verify_event(
            events_raw, event_feature_raw, fps,
            min_event_len=0.5, max_event_len=10,
            redundant_event_threshold=0.1,
            trim_begin_last_percent=0.1,
            still_event_threshold=0.2,
            min_frames=min_frames
        )

        return transfer_timecode(events, fps) if events else []

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stitching")
    parser.add_argument("--videos-root-dir", type=str, required=True,
                        help="Root directory containing video files and cutscene_frame_idx.json")
    parser.add_argument("--min-frames", type=int, default=None,
                        help="Minimum number of frames required for each event (default: None)")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Force reprocessing of videos even if they exist in the output JSON")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of worker processes for frame extraction (default: 8)")
    args = parser.parse_args()

    # Check for cutscene_frame_idx.json
    cutscene_json_path = os.path.join(args.videos_root_dir, "cutscene_frame_idx.json")
    if not os.path.exists(cutscene_json_path):
        print(f"Error: cutscene_frame_idx.json not found in {args.videos_root_dir}")
        sys.exit(1)

    # Get list of video files
    video_paths = get_video_files(args.videos_root_dir)
    if not video_paths:
        print(f"No video files found in {args.videos_root_dir}")
        sys.exit(1)

    # Load cutscene data
    with open(cutscene_json_path) as f:
        video_cutscenes = json.load(f)

    # Setup device and model
    device = "cuda"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Setup image transform
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # Output JSON file path
    output_json_file = os.path.join(args.videos_root_dir, "event_timecode.json")

    # Load existing results
    existing_results = safely_read_json(output_json_file)
    print(f"Found {len(existing_results)} previously processed videos")

    # Filter videos to process
    videos_to_process = []
    for video_path in video_paths:
        video_basename = os.path.basename(video_path)

        # Skip if video has no cutscene data
        if video_basename not in video_cutscenes:
            print(f"Skipping {video_basename} - no cutscene data found")
            continue

        # Skip if already processed and not force reprocessing
        if video_basename in existing_results and not args.force_reprocess:
            print(f"Skipping {video_basename} - already processed")
            continue

        videos_to_process.append(video_path)

    if not videos_to_process:
        print("No new videos to process")
        sys.exit(0)

    print(f"Processing {len(videos_to_process)} videos")

    # Process videos and update JSON continuously
    for video_path in tqdm(videos_to_process):
        video_basename = os.path.basename(video_path)
        cutscenes = video_cutscenes[video_basename]

        # Process the video
        events = process_video(video_path, cutscenes, args.min_frames, args.num_workers)

        # Update JSON with results (empty list for videos with no events)
        # None indicates processing error
        if events is not None:
            update_json_with_result(output_json_file, video_basename, events)
            status = "no events found" if len(events) == 0 else f"found {len(events)} events"
            print(f"Processed {video_basename} - {status}")
        else:
            print(f"Failed to process {video_basename}")

    print("\nDone!")