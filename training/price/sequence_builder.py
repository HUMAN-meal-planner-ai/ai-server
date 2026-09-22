from __future__ import annotations

import numpy as np
import pandas as pd


def sequence_eligible_indices(frame: pd.DataFrame, sequence_length: int) -> np.ndarray:
    if sequence_length < 1:
        raise ValueError("시퀀스 길이는 1 이상이어야 합니다.")
    positions = frame.groupby("series_id", sort=False).cumcount()
    return frame.index[positions >= sequence_length - 1].to_numpy(dtype=int)


def build_sequences(
    transformed_features: np.ndarray,
    frame: pd.DataFrame,
    target_indices: np.ndarray,
    sequence_length: int,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """series 경계를 넘지 않는 관측치 기준 시퀀스를 생성한다."""
    if len(transformed_features) != len(frame):
        raise ValueError("변환된 feature와 원본 행 수가 일치해야 합니다.")

    sequences = []
    targets = []
    for target_index in target_indices:
        start_index = int(target_index) - sequence_length + 1
        if start_index < 0:
            raise ValueError("시퀀스 생성에 필요한 과거 관측치가 부족합니다.")
        series_ids = frame.loc[start_index:target_index, "series_id"]
        if len(series_ids) != sequence_length or series_ids.nunique() != 1:
            raise ValueError("시퀀스가 서로 다른 가격 시계열의 경계를 넘었습니다.")
        sequences.append(transformed_features[start_index : target_index + 1])
        targets.append(frame.at[target_index, target_column])

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)
