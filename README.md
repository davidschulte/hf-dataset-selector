# Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning (EMNLP 2024)

This code contains the functionality to reproduce the paper.

## Most important scripts
- parse_dataset_info.py: Parses datasets from the Hugginface Hub
- main_transfer.py: Conducts intermediate task transfer learning for all source-target-pairs.
- main_train_esms.py: Trains ESMs for all sources
- main_esm_logme: Computes ESM-LogME scores for all source-target-pairs
- create_eval_tables.py: Combines several ranking metrics (ESM-LogME, LogME, TaskEmb, ...) to rankings
and evaluates them using ground truth target model performance

Other scripts starting with "main_" compute scores of remaining source selection methods.

We integrate the following existing source selection method implementations:
* [NCE, LEEP, LogME](https://github.com/thuml/LogME)
* [TaskEmb](https://github.com/tuvuumass/task-transferability)