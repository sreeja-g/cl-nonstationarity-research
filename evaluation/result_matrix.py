# evaluation/result_matrix.py

import re
import numpy as np


class ResultMatrix:
    """
    Stores result matrix R.

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
        """
        Updates row train_exp_id using Avalanche eval results.
        """

        if train_exp_id < 0 or train_exp_id >= self.n_experiences:
            raise ValueError("train_exp_id out of range")

        for key, value in eval_results.items():

            if "Top1_Acc_Exp" not in key:
                continue

            eval_exp_id = self._extract_eval_exp_id(key)

            if eval_exp_id is None:
                continue

            if eval_exp_id < 0 or eval_exp_id >= self.n_experiences:
                continue

            self.matrix[train_exp_id, eval_exp_id] = self._to_float(value)

    def _extract_eval_exp_id(self, key: str):
        """
        Extracts experience id from Avalanche metric key.

        Example:
            Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003
            -> 3
        """

        match = re.search(r"Exp(\d+)", key)

        if match is None:
            return None

        return int(match.group(1))

    def _to_float(self, value):
        """
        Converts metric value to float.
        Handles Python numbers, NumPy values, and Torch tensors.
        """

        if hasattr(value, "item"):
            return float(value.item())

        return float(value)

    def get(self):
        """
        Returns a copy of the result matrix.
        """

        return self.matrix.copy()

    def save(self, path: str):
        """
        Saves result matrix as .npy file.
        """

        np.save(path, self.matrix)

    @staticmethod
    def load(path: str):
        """
        Loads result matrix from .npy file.
        """

        return np.load(path)