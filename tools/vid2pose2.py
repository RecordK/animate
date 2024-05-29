import sys
import os
from tqdm import tqdm
import torch  # GPU 사용 가능 여부를 확인하기 위해 추가
from pathlib import Path
import cv2
import numpy as np



# 현재 스크립트의 경로를 기준으로 프로젝트 루트 디렉토리 경로를 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(project_root)

from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil

# 사용할 CPU 코어 집합 (예: 0-9번 코어)
cpu_cores = list(range(10))

def set_affinity():
    pid = os.getpid()
    os.sched_setaffinity(pid, cpu_cores)

def process_video(vi):
    # 각 프로세스에서 PyTorch가 사용할 쓰레드 수를 제한
    torch.set_num_threads(len(cpu_cores))
    
    set_affinity()


    if not os.path.exists(vi):
        raise ValueError(f"Path: {vi} not exists")

    dir_path, video_name = (
        os.path.dirname(vi),
        os.path.splitext(os.path.basename(vi))[0],
    )
    out_path = os.path.join('/home/sign/kr/Moore-AnimateAnyone/j_data/kps_train', video_name + "_kps.mp4")

    detector = DWposeDetector()
    detector = detector.to("cuda:0")

    fps = get_fps(vi)
    frames = read_frames(vi)
    kps_results = []

    for i, frame_pil in tqdm(enumerate(frames), total=len(frames), desc=f"Processing {vi}", unit="frame"):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)
        kps_results.append(result)

    print(out_path)
    save_videos_from_pil(kps_results, out_path, fps=fps)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    
    videos = set(os.listdir(args.dir))
    v2=set()
    for i in os.listdir('/home/sign/kr/Moore-AnimateAnyone/j_data/kps_train'):
        v2.add(os.path.basename(i).split('_kps')[0]+'.mp4')
    v2=list(v2)
    
    video_paths = [os.path.join(os.getcwd(), 'j_data/train', vi) for vi in v2]

    # with Pool(processes=10) as pool:
    #     pool.map(process_video, video_paths)
    for video_path in video_paths:
        process_video(video_path)