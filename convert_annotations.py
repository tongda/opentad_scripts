import json
import sys
from pathlib import Path
import re
from datetime import datetime
from mmcv import VideoReader
import random

def parse_elan_txt(path: Path, test_split_ratio = 0.3):
    with open(path, "r") as f:
        lines = f.readlines()
    
    reader = VideoReader(str(path.parent.parent / "raw_data" / "video" / (path.stem + ".mp4")))

    segments = []
    for label, time_range in zip(lines[::2], lines[1::2]):
        label = label.strip()
        # time range example: 00:09:02.215 - 00:09:03.479, parse the format with regex into begin time and end time, and convert them into seconds
        begin_time, end_time = re.findall(r"\d{2}:\d{2}:\d{2}\.\d{3}", time_range)
        
        begin_time = datetime.strptime(begin_time, "%H:%M:%S.%f")
        end_time = datetime.strptime(end_time, "%H:%M:%S.%f")
        
        begin_time = begin_time.hour * 3600 + begin_time.minute * 60 + begin_time.second + begin_time.microsecond / 1e6
        end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6
            
        # if int(label.split("-")[0]) in [2, 3, 6, 7, 8, 9]:
        #     label = "2-组装零件"
        
        if label != "-":
            segment = {
                "label": label,
                "segment": [begin_time, end_time],
            }
            segments.append(segment)
    
    return {
        "duration": reader.frame_cnt / reader.fps,
        "frame": reader.frame_cnt,
        "subset": "training" if random.random() > test_split_ratio else "testing",
        "annotations": segments
    }

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("No arguments provided")
    
    raw_anno_dir = Path(sys.argv[1])

    database = {}
    for anno_file in raw_anno_dir.glob("*.txt"):
        if anno_file.stem == "category_idx":
            continue
        database[anno_file.stem] = parse_elan_txt(anno_file)
        
    # collect categories
    categories = list(set([segment["label"] for video in database.values() for segment in video["annotations"]]))
    print(categories)
    
    with open("data/b11_phone_motion2_backview/annotations/b11_phone_backview_anno.json", "w") as f:
        json.dump({"database": database}, f)
        
    with open("data/b11_phone_motion2_backview/annotations/category_idx.txt", "w") as f:
        for idx, category in enumerate(categories):
            f.write(f"{category}\n")