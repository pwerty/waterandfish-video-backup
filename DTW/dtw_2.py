# Updated dtw_system.py with automatic parameter estimation and convergence-based DBA

import os
import json
import numpy as np
from dtaidistance import dtw_ndim

# — 자동 파라미터 추정용 상수 —
ONSET_PERCENTILE = 90       # 움직임 시작점 감지 시 velocity 상위 퍼센타일
NORMALIZE_SCALE  = 0.1      # 관절별 Min–Max 최소 범위 = median(range) * SCALE
DBA_MAX_ITER     = 20       # DBA 최대 반복 횟수
DBA_TOL          = 1e-4     # DBA 템플릿 변화량 수렴 허용오차

def detect_onset(seq, percentile=ONSET_PERCENTILE):
    """
    프레임 간 최대 관절 속도로 움직임 시작점 감지 (Percentile 기반 자동 임계치)
    """
    diffs    = np.linalg.norm(seq[1:] - seq[:-1], axis=2)  # (T-1, N)
    max_vel  = diffs.max(axis=1)                           # (T-1,)
    vel_thresh = np.percentile(max_vel, percentile)
    idx = np.where(max_vel > vel_thresh)[0]
    return int(idx[0]) if idx.size else 0

def normalize_per_joint(seq, scale=NORMALIZE_SCALE):
    """
    관절별 Min–Max 정규화 (무의미한 작은 움직임 제외, 데이터 기반 최소 범위)
    """
    T, N, C = seq.shape
    out = np.zeros_like(seq)
    # 각 관절의 움직임 폭(range) 평균 계산
    ranges = np.array([np.ptp(seq[:, j, :], axis=0).mean() for j in range(N)])
    # median(range) * scale 을 최소 범위로 설정
    min_range = np.median(ranges) * scale

    for j in range(N):
        data = seq[:, j, :]                   # (T, 3)
        mn, mx = data.min(axis=0), data.max(axis=0)
        rng = mx - mn
        mask = rng > min_range
        out[:, j, :] = data
        if mask.any():
            out[:, j, mask] = (data[:, mask] - mn[mask]) / (rng[mask] + 1e-8)
    return out

def load_sequences(fp):
    """
    JSON 파일에서 'pose', 'left_hand', 'right_hand' 시퀀스를
    (T, N, 3) NumPy 배열로 반환
    """
    raw = json.load(open(fp, 'r', encoding='utf-8'))
    if isinstance(raw, str):
        raw = json.loads(raw)
    keys = ['pose', 'left_hand', 'right_hand']
    seqs = {}
    for k in keys:
        data = raw.get(k, None)
        if data is None:
            raise KeyError(f"키 '{k}'가 JSON에 없습니다: {fp}")
        seqs[k] = np.array(data, dtype=float)
    return seqs

def dba(sequences, init_template, max_iter=DBA_MAX_ITER, tol=DBA_TOL):
    """
    DTW Barycenter Averaging (수렴할 때까지 반복)
    tol 이하로 템플릿 변화량이 줄어들면 자동 멈춤
    """
    template = init_template.copy()
    for _ in range(max_iter):
        # 1) 프레임별 슬롯 준비
        slots = [[] for _ in range(len(template))]
        for seq in sequences:
            path = dtw_ndim.warping_path(template, seq)
            for i_ref, i_seq in path:
                slots[i_ref].append(seq[i_seq])
        # 2) 슬롯별 중앙값으로 새 템플릿 계산
        new_t = np.vstack([
            np.median(slot, axis=0) if slot else template[i_ref]
            for i_ref, slot in enumerate(slots)
        ])
        # 3) 수렴 검사
        if np.linalg.norm(new_t - template) < tol:
            break
        template = new_t
    return template

def main(data_dir="./dtw_data"):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if len(files) < 2:
        print("폴더에 최소 2개 이상의 JSON 파일이 필요합니다.")
        return

    keys = ['pose', 'left_hand', 'right_hand']
    seqs_by_key = {k: [] for k in keys}
    names = []

    # 1) 모든 파일 로드 + 전처리
    for fn in files:
        fp = os.path.join(data_dir, fn)
        seqs3d = load_sequences(fp)
        names.append(fn)

        for k in keys:
            arr3d = seqs3d[k]  # (T, N, 3)
            # 자동 시작점 감지 및 세그멘테이션
            start_idx = detect_onset(arr3d)
            segmented = arr3d[start_idx:]
            # 자동 정규화
            normalized = normalize_per_joint(segmented)
            T, N, _ = normalized.shape
            flat = normalized.reshape(T, N*3)  # (T, D_k)
            seqs_by_key[k].append(flat)

    # 2) 파일별 pivot DBA 수행 및 저장
    for idx, pivot_name in enumerate(names):
        merged = {}
        for k in keys:
            sequences = seqs_by_key[k]
            init = sequences[idx]
            template = dba(sequences, init)  # 자동 수렴
            T0, D = template.shape
            N = D // 3
            merged[k] = template.reshape(T0, N, 3).tolist()

        out_name = pivot_name.replace('.json', '_converted.json')
        out_fp = os.path.join(data_dir, out_name)
        with open(out_fp, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {out_fp}")

if __name__ == '__main__':
    main()

