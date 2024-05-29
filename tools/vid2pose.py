import os
import sys
from tqdm import tqdm
# 현재 스크립트의 경로를 기준으로 프로젝트 루트 디렉토리 경로를 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(project_root)

from src.dwpose import DWposeDetector
from pathlib import Path
from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np




# import sys
# sys.path.append("/home/sign/kr/Moore-AnimateAnyone/")  # 프로젝트 루트 디렉토리 경로로 변경해야 합니다.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--video_path", type=str)
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    # if not os.path.exists(args.video_path):
    #     raise ValueError(f"Path: {args.video_path} not exists")

    # dir_path, video_name = (
    #     os.path.dirname(args.video_path),
    #     os.path.splitext(os.path.basename(args.video_path))[0],
    # )
    # out_path = os.path.join(dir_path, video_name + "_kps.mp4")

    # detector = DWposeDetector()
    # detector = detector.to(f"cuda")

    # fps = get_fps(args.video_path)
    # frames = read_frames(args.video_path)
    # kps_results = []
    # for i, frame_pil in enumerate(frames):
    #     result, score = detector(frame_pil)
    #     score = np.mean(score, axis=-1)

    #     kps_results.append(result)
#11460a_a06
#11460a_a06_kps
    # print(out_path)
    # save_videos_from_pil(kps_results, out_path, fps=fps)
    videos=os.listdir(args.dir)
    for vi in videos:
        vi = os.path.join(os.path.join(os.getcwd(), 'j_data/train'),vi)
        if not os.path.exists(vi):
            raise ValueError(f"Path: {vi} not exists")

        dir_path, video_name = (
            os.path.dirname(vi),
            os.path.splitext(os.path.basename(vi))[0],
        )
        out_path = os.path.join(dir_path, video_name + "_kps.mp4")

        detector = DWposeDetector()
        detector = detector.to(f"cuda:0")

        fps = get_fps(vi)
        frames = read_frames(vi)
        kps_results = []
        # for i, frame_pil in enumerate(frames):
        for i, frame_pil in tqdm(enumerate(frames), total=len(frames), desc=f"Processing {vi}", unit="frame"):

            result, score = detector(frame_pil)
            score = np.mean(score, axis=-1)

            kps_results.append(result)

        print(out_path)
        save_videos_from_pil(kps_results, out_path, fps=fps)
