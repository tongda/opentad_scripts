import json
import os
import subprocess
from typing import Tuple, Union
from pathlib import Path
from decord import VideoReader
import math
import random
from collections import defaultdict
import concurrent.futures
import queue
import threading
import time


def prune_start_end(start_frame, end_frame, fps, annotations):
    # if start_frame or end_frame is within an segment, move it to the start or end of the segment
    for anno in annotations:
        if anno["segment"][0] <= start_frame / fps  <= anno["segment"][1]:
            start_frame = int(anno["segment"][1] * fps)
        if anno["segment"][0] <= end_frame / fps <= anno["segment"][1]:
            end_frame = int(anno["segment"][0] * fps)
    return start_frame, end_frame


def extract_clip_with_ffmpeg(video_path, output_path, start_time, duration):
    """使用FFmpeg提取视频片段"""
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖现有文件
        '-ss', str(start_time),  # 起始时间
        '-i', str(video_path),  # 输入文件
        '-t', str(duration),  # 持续时间
        '-c:v', 'h264_nvenc',  # 视频编码
        '-preset', 'slow',  # 编码速度和质量的权衡
        '-crf', '18',  # 质量设置，低值=高质量
        str(output_path)
    ]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(f"FFmpeg error: {process.stderr.decode()}")
        return False
    print(f"FFmpeg output: {process.stdout.decode()}")
    return True


class TaskExecutor:
    """任务执行器，使用线程池控制并发"""
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = queue.Queue()
        self.running = 0
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.results = []
        self.done = False
    
    def add_task(self, task_fn, *args, **kwargs):
        """添加任务到队列"""
        self.tasks.put((task_fn, args, kwargs))
        self._check_and_submit()
    
    def _check_and_submit(self):
        """检查并提交任务"""
        with self.lock:
            print(f"Running tasks: {self.running}, Queue size: {self.tasks.qsize()}")
            if self.running < self.max_workers and not self.tasks.empty():
                task_fn, args, kwargs = self.tasks.get()
                future = self.executor.submit(task_fn, *args, **kwargs)
                future.add_done_callback(self._task_done)
                self.running += 1
    
    def _task_done(self, future):
        """任务完成的回调"""
        try:
            result = future.result()
        except Exception as e:
            print(f"Task error: {e}")
            result = None
            
        with self.lock:
            self.running -= 1
            print(f"Task completed, running tasks: {self.running}")
            if result is not None:
                self.results.append(result)
                
        # Call _check_and_submit outside the lock to prevent deadlock
        self._check_and_submit()
        
        # Check completion status after releasing the lock
        with self.lock:
            # 如果设置为完成且没有运行中的任务和排队任务，停止执行器
            if self.done and self.running == 0 and self.tasks.empty():
                self.executor.shutdown(wait=False)
    
    def wait_completion(self):
        """等待所有任务完成"""
        self.done = True
        while self.running > 0 or not self.tasks.empty():
            time.sleep(0.1)
        return self.results


def process_video_clip(video_path, start_frame, end_frame, fps, output_path, clip_annotation):
    """处理单个视频片段的函数，用于线程池执行"""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    print(f"Processing clip from {start_time:.2f}s to {start_time + duration:.2f}s")
    success = extract_clip_with_ffmpeg(video_path, output_path, start_time, duration)
    return {
        "output_path": str(output_path), 
        "success": success, 
        "clip_annotation": clip_annotation
    }


def split_video(video_path: str, annotations: list, clip_secs: Union[int, Tuple[int, int]], 
               output_dir, overlap_secs: int = 0, train_val_ratio: float = 0.2,
               max_workers: int = 4):
    video = VideoReader(video_path)
    try:
        video_stem = Path(video_path).stem
        fps = video.get_avg_fps()
        
        frame_index = 0
        splitted_annotations = {
            "database": {}
        }
        
        # 创建任务执行器
        executor = TaskExecutor(max_workers=max_workers)
        
        min_clip_sec = clip_secs if isinstance(clip_secs, int) else clip_secs[0]
        clip_tasks = []
        
        while frame_index <= (len(video) - math.floor(min_clip_sec * video.get_avg_fps())):
            start_frame = frame_index
            if isinstance(clip_secs, int):
                end_frame = min(frame_index + math.ceil(clip_secs * video.get_avg_fps()), len(video))
            else:
                clip_len = random.randint(clip_secs[0], clip_secs[1])
                end_frame = min(start_frame + math.ceil(clip_len * video.get_avg_fps()), len(video))

            start_frame, end_frame = prune_start_end(start_frame, end_frame, fps, annotations)
            
            output_name = f"clip_{video_stem}_{start_frame}_{end_frame}"
            output_path = Path(output_dir) / f"{output_name}.mp4"
            
            # 创建输出目录（如果不存在）
            os.makedirs(Path(output_dir), exist_ok=True)

            clip_annotation = {
                "duration": (end_frame - start_frame) / fps,
                "frame": end_frame - start_frame,
                "subset": "training" if random.random() > train_val_ratio else "testing",
                "annotations": []
            }
            
            for anno in annotations:
                # 检查注释是否在当前片段范围内
                if not (anno["segment"][1] * fps <= end_frame and anno["segment"][0] * fps >= start_frame):
                    continue
                clip_anno = anno.copy()
                anno_start = anno["segment"][0] - start_frame / fps
                anno_end = anno["segment"][1] - start_frame / fps
                clip_anno["segment"] = [anno_start, anno_end]
                clip_annotation["annotations"].append(clip_anno)

            # 添加到任务执行器
            executor.add_task(
                process_video_clip, 
                video_path, 
                start_frame, 
                end_frame, 
                fps, 
                output_path, 
                clip_annotation
            )
            
            clip_tasks.append((output_name, clip_annotation))
            
            frame_index = end_frame - math.floor(overlap_secs * video.get_avg_fps())
            print(f"Queued task for {output_name}, from frame {start_frame} to {end_frame}")
            
            if end_frame >= len(video):
                break
        
        # 等待所有任务完成
        results = executor.wait_completion()
        
        # 处理结果和构建注释
        for output_name, clip_annotation in clip_tasks:
            if clip_annotation["annotations"]:  # 只添加有注释的片段
                splitted_annotations["database"][output_name] = clip_annotation
        
        # 写入注释到JSON文件
        with open(Path(output_dir) / f"annotations_{video_stem}.json", "w") as f:
            json.dump(splitted_annotations, f)
            
        print(f"Completed processing video {video_path}")
        return splitted_annotations
        
    finally:
        # 确保视频读取器被释放
        del video


if __name__ == "__main__":
    with open("/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/annotations/b11_phone_backview_anno.json", "r") as f:
        database = json.load(f)["database"]
    
    # 设置最大并行处理数
    max_workers = 4  # 可根据系统资源调整
    
    for video_name in database.keys():
        print(f"Processing video: {video_name}")
        annotations = database[video_name]["annotations"]
        split_video(
            f"/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/video/{video_name}.mp4",
            annotations, 
            [180, 240], 
            f"/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/clips2/",
            overlap_secs=30,
            train_val_ratio=0.5,
            max_workers=max_workers,
        )



