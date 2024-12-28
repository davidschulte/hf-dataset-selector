from typing import List, Optional, Union
import numpy as np
from collections.abc import Sequence


class TaskRanking(Sequence):

    def __init__(
            self,
            esm_configs: Union["ESMConfig", List["ESMConfig"]],
            scores: Union[float, List[float]],
            ranks: Optional[Union[int, List[int]]] = None
    ):

        assert len(esm_configs) == len(scores)

        sorting_order = np.argsort(scores)[::-1]
        self.esm_configs = [esm_configs[idx] for idx in sorting_order]
        self.scores = [scores[idx] for idx in sorting_order]
        self.ranks = list(range(1, len(self)+1)) if ranks is None else [ranks[idx] for idx in sorting_order]

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
        return "\n".join(self[:10]._format_for_output(score_rounding=6)) + "\n..." if len(self) > 10 else ""

    def _format_for_output(self, score_rounding: Optional[int] = None):
        output_lines = []
        for rank, esm_config, score in zip(self.ranks, self.esm_configs, self.scores):
            output_lines.append(f"{rank}.\t{esm_config.task_id}\t-"
                                f"Score: {round(score, score_rounding) if score_rounding else score}")

        return output_lines
