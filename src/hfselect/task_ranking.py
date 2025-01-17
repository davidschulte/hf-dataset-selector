from typing import List, Optional, Union
import numpy as np
from collections.abc import Sequence


class InvalidTaskRankingError(Exception):
    default_message = "The task ranking is invalid."

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.default_message)


class TaskRanking(Sequence):

    def __init__(
            self,
            esm_configs: Union["ESMConfig", List["ESMConfig"]],
            scores: Union[float, List[float]],
            ranks: Optional[Union[int, List[int]]] = None
    ):

        self.esm_configs = esm_configs
        self.scores = scores
        self.ranks = ranks

        self._remove_faulty_scores()
        self.check_validity()
        self.sort()

    def __getitem__(self, index):

        if isinstance(index, int):
            return TaskRanking(
                self.esm_configs[index],
                self.scores[index],
                self.ranks[index]
            )

        elif isinstance(index, slice):
            return TaskRanking(self.esm_configs[index], self.scores[index], self.ranks[index])

        raise Exception(f"TaskRanking could not be indexed with a type {type(index)}.")

    def __len__(self):
        return len(self.esm_configs)

    def __repr__(self):
        return "\n".join(self._format_for_output())

    def __str__(self):
        return "\n".join(self[:10]._format_for_output(score_rounding=6)) + ("\n..." if len(self) > 10 else "")

    def _format_for_output(self, score_rounding: Optional[int] = None):
        output_lines = []
        for rank, esm_config, score in zip(self.ranks, self.esm_configs, self.scores):
            output_lines.append(f"{rank}.\t{esm_config.task_id}\t- "
                                f"Score: {round(score, score_rounding) if score_rounding else score}")

        return output_lines

    def sort(self):
        sorting_order = np.argsort(self.scores)[::-1]
        self.esm_configs = [self.esm_configs[idx] for idx in sorting_order]
        self.scores = [self.scores[idx] for idx in sorting_order]
        self.ranks = list(range(1, len(self)+1)) if self.ranks is None else [self.ranks[idx] for idx in sorting_order]

    def _remove_faulty_scores(self):
        faulty_indices = [score for score in self.scores if np.isnan(score)]
        self.esm_configs = self._remove_indices(self.esm_configs, faulty_indices)
        self.scores = self._remove_indices(self.scores, faulty_indices)
        self.ranks = self._remove_indices(self.ranks, faulty_indices)

    @staticmethod
    def _remove_indices(list_to_clean: Optional[list], indices: list[int]):
        if list_to_clean is None:
            return
        if len(indices) == 0:
            return list_to_clean

        return [elem for idx, elem in enumerate(list_to_clean) if idx not in indices]

    def check_validity(self):
        if len(self.esm_configs) != len(self.scores):
            raise InvalidTaskRankingError(
                f"Task ranking contains {len(self.esm_configs)} ESM configs but {len(self.scores)} scores."
            )

        if self.ranks is not None and len(self.esm_configs) != len(self.ranks):
            raise InvalidTaskRankingError(
                f"Task ranking contains {len(self.esm_configs)} ESM configs but {len(self.ranks)} provided ranks."
            )

        if len(self.esm_configs) == 0:
            raise InvalidTaskRankingError(
                f"Task ranking is empty."
            )
