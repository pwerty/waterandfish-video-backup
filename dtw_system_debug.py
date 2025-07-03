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


def median_smooth(template, window=2):
    """
    template: (T, D) 배열
    window: 앞뒤로 몇 프레임 범위를 median 에 사용할지
    """
    T, D = template.shape
    sm = np.zeros_like(template)
    for t in range(T):
        start, end = max(0, t-window), min(T, t+window+1)
        sm[t] = np.median(template[start:end], axis=0)
    return sm

def clamp_velocity(template, max_move=0.02):
    """
    template: (T, D) 배열
    max_move: 한 프레임당 최대 이동량 (거리 단위)
    """
    out = [template[0]]
    for p in template[1:]:
        prev = out[-1]
        delta = p - prev
        dist  = np.linalg.norm(delta)
        if dist > max_move:
            delta = delta * (max_move / dist)
        out.append(prev + delta)
    return np.vstack(out)

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

# … (기존 import, 자동 파라미터 정의, detect_onset, normalize_per_joint, dba 등) …

def summarize_range(seq, stage):
    # seq: (T, N, 3) 혹은 (T, D) 배열
    arr = seq.reshape(-1, 3) if seq.ndim == 3 else seq.reshape(-1, 3)
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    print(f"[{stage}] x∈[{mn[0]:.3f},{mx[0]:.3f}], "
          f"y∈[{mn[1]:.3f},{mx[1]:.3f}], "
          f"z∈[{mn[2]:.3f},{mx[2]:.3f}]")

def summarize_velocity(seq, stage):
    # seq.ndim == 3 이면 (T,N,3), ==2 이면 (T,D) 로 가정
    delta = seq[1:] - seq[:-1]
    if seq.ndim == 3:
        # (T-1, N, 3)
        jumps = np.linalg.norm(delta, axis=2).max(axis=1)
    else:
        # (T-1, D)
        jumps = np.linalg.norm(delta, axis=1)
    print(f"[{stage}] max jump={jumps.max():.4f}, 95th pct={np.percentile(jumps,95):.4f}")


def main(data_dir="./dtw_data"):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    for fn in files:
        fp = os.path.join(data_dir, fn)
        seqs3d = load_sequences(fp)
        for k in ['pose','left_hand','right_hand']:
            arr = seqs3d[k]  # (T, N, 3)
            
            # 1) 원본 범위 체크
            summarize_range(arr, f"{fn}/{k}/raw")
            
            # 2) Onset 절단
            idx = detect_onset(arr)
            seg = arr[idx:]
            summarize_range(seg, f"{fn}/{k}/segmented")
            summarize_velocity(seg, f"{fn}/{k}/segmented")
            
            # 3) 정규화
            normed = normalize_per_joint(seg)
            summarize_range(normed, f"{fn}/{k}/normalized")
            summarize_velocity(normed, f"{fn}/{k}/normalized")
            
            # 4) DBA (템플릿 생성, D-dimensional)
            flat_seqs = normed.reshape(len(normed), -1)
            template = dba([flat_seqs], flat_seqs)  # 테스트용으로 self-DBA 실행
            # reshape back to (T0, N, 3)
            T0, D = template.shape
            tpl3d = template.reshape(T0, D//3, 3)
            summarize_range(tpl3d, f"{fn}/{k}/dba")
            summarize_velocity(tpl3d, f"{fn}/{k}/dba")
            
            # 5) 후처리(스무딩+클램프) 검사
            sm = median_smooth(template, window=2).reshape(T0, D//3, 3)
            summarize_range(sm, f"{fn}/{k}/smoothed")
            summarize_velocity(sm, f"{fn}/{k}/smoothed")
            cl = clamp_velocity(sm, max_move=0.02)
            summarize_range(cl, f"{fn}/{k}/clamped")
            summarize_velocity(cl, f"{fn}/{k}/clamped")
            
        print("-"*40)


if __name__ == '__main__':
    main()

