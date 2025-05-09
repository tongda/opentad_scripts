import json
from typing import Tuple, Union
from pathlib import Path
from decord import VideoReader
import cv2
import math
import random
from collections import defaultdict


def prune_start_end(start_frame, end_frame, fps, annotations):
    # if start_frame or end_frame is within an segment, move it to the start or end of the segment
    for anno in annotations:
        if anno["segment"][0] <= start_frame / fps  <= anno["segment"][1]:
            start_frame = int(anno["segment"][1] * fps)
        if anno["segment"][0] <= end_frame / fps <= anno["segment"][1]:
            end_frame = int(anno["segment"][0] * fps)
    return start_frame, end_frame


def split_video(video_path: str, annotations: list, clip_secs: Union[int, Tuple[int, int]], output_dir, overlap_secs: int = 0, train_val_ratio: float = 0.2):
    video = VideoReader(video_path)
    video_stem = Path(video_path).stem
    fps = video.get_avg_fps()
    
    frame_index = 0

    splitted_annotations = {
        "database": defaultdict(list)
    }
    min_clip_sec = clip_secs if isinstance(clip_secs, int) else clip_secs[0]
    while frame_index <= (len(video) - math.floor(min_clip_sec * video.get_avg_fps())):
    # for start_frame, end_frame in [[0,6900], [6150,11650], [10900,17175], [16425,23900], [23150,28375], [27625,35125], [34375,40175], [39425,46900]]:
        start_frame = frame_index
        if isinstance(clip_secs, int):
            end_frame = min(frame_index + math.ceil(clip_secs * video.get_avg_fps()), len(video))
        else:
            clip_len = random.randint(clip_secs[0], clip_secs[1])
            end_frame = min(start_frame + math.ceil(clip_len * video.get_avg_fps()), len(video))

        start_frame, end_frame = prune_start_end(start_frame, end_frame, fps, annotations)
        
        output_name = f"clip_{video_stem}_{start_frame}_{end_frame}"
        output_path = Path(output_dir)  / f"{output_name}.mp4"

        clip_annotation = {
            "duration": (end_frame - start_frame) / fps,
            "frame": end_frame - start_frame,
            "subset": "training" if random.random() > train_val_ratio else "testing",
            "annotations": []
        }
        
        for anno in annotations:
            if not (anno["segment"][1] * fps <= end_frame and anno["segment"][0] * fps >= start_frame):
                continue
            clip_anno = anno.copy()
            anno_start = anno["segment"][0] -  start_frame / fps
            anno_end = anno["segment"][1] - start_frame / fps
            clip_anno["segment"] = [anno_start, anno_end]
            clip_annotation["annotations"].append(clip_anno)

        if clip_annotation:
            splitted_annotations["database"][output_name] = clip_annotation
            clip = video.get_batch(list(range(start_frame, end_frame))).asnumpy()
            # write clip to video file in output_dir
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), video.get_avg_fps(), (clip[0].shape[1], clip[0].shape[0]))
            for frame in clip:
                writer.write(frame[:, :, ::-1])
            writer.release()

        frame_index = end_frame - math.floor(overlap_secs * video.get_avg_fps())
        print("Processed", output_name, "from", start_frame, "to", end_frame, "frame_index updated to", frame_index)
        print(clip_annotation)
        if end_frame >= len(video):
            break
    
    # write annotations to json file in output_dir
    with open(Path(output_dir) / f"annotations_{video_stem}.json", "w") as f:
        json.dump(splitted_annotations, f)

if __name__ == "__main__":
    with open("/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/annotations/b11_phone_backview_anno.json", "r") as f:
        database = json.load(f)["database"]

    for video_name in database.keys():
        annotations = database[video_name]["annotations"]
        split_video(
            f"/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/video/{video_name}.mp4",
            annotations, 
            [60, 80], 
            f"/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/clips/",
            overlap_secs=30,
            train_val_ratio=0.5,
        )
    # annotations = database["IMG_2316-2"]["annotations"]
    # split_video(
    #     "/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/video/IMG_2316-2.mp4",
    #     annotations, 
    #     [60, 80], 
    #     "/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/clips",
    #     overlap_secs=30,
    #     train_val_ratio=0.5,
    # )



        