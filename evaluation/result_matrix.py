# evaluation/result_matrix.py

import re
import numpy as np


class ResultMatrix:
    """
    Stores R[i, j].

    R[i, j] = test accuracy on experience j
              after training experience i
    """

    def __init__(self, n_experiences: int):
        self.n_experiences = n_experiences
        self.matrix = np.full(
            shape=(n_experiences, n_experiences),
            fill_value=np.nan,
            dtype=float,
        )

    def update(self, train_exp_id: int, eval_results: dict):
        for key, value in eval_results.items():
            if "Top1_Acc_Exp" not in key:
                continue

            exp_id = self._extract_eval_exp_id(key)

            if exp_id is not None:
                self.matrix[train_exp_id, exp_id] = float(value)

    def _extract_eval_exp_id(self, key: str):
        match = re.search(r"Exp(\d+)", key)

        if match is None:
            return None

        return int(match.group(1))

    def get(self):
        return self.matrix.copy()

    def save(self, path: str):
        np.save(path, self.matrix)

    @staticmethod
    def load(path: str):
        return np.load(path)