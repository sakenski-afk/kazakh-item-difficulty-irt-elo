# Data

The study uses an anonymised export of the production MySQL database of Aqyl Battle, a Kazakh mobile quiz-based learning game operated by a private company. **The raw data are not distributed with this repository.** They include data of minors. Researchers may request access from the corresponding author; access is subject to the operator's approval and a data-use agreement. Item-level statistics without texts are openly available on Zenodo.

## Export format

`data/raw/` must contain `manifest.json` and one gzip-compressed CSV with a header row per table (`<table>.csv.gz`). The manifest lists, for every table, the row count, the exported columns and the columns excluded for privacy. Personally identifiable fields were removed before export: names, phone numbers, e-mail addresses, tokens and message texts.

The export in the paper was generated on 2026-09-22 with a limit of 100,000 rows per table (first rows by primary key). As a result, the tables cover different time windows (see `docs/data_dictionary.md` and `python src/audit_export.py`).

## Files required by the pipeline

| File | Used by | Key columns |
|---|---|---|
| `questions.csv.gz` | Part 1, Part 2 (text of cold-start items) | `id, questions` (text), `cat_id, lang, main_question_id, image, user_id, correct, fail, report, created_at` |
| `answers.csv.gz` | Part 1 option ablation | `id, question_id, text, select_count` |
| `correct_answers.csv.gz` | Part 1, Part 2 (answer key) | `question_id, answer_id` |
| `cats.csv.gz` | Part 1, Part 2 | `id, name, name_ru` |
| `group_questions.csv.gz` | Part 2 (tournament responses) | `user_id, question_id, answer_id, created_at` |
| `games.csv.gz` | Part 3 | `id, user_id, opponent_id, winner_id, status, created_at` |
| `matches.csv.gz` | Part 3 | `game_id, round, cat_id, user_*_question, opponent_*_question, created_at` |
| `manifest.json` | audit | – |

The remaining tables of the export are described in `docs/data_dictionary.md` but are not needed to reproduce the paper.

## Generated files

`data/items.csv.gz` (Part 1 dataset with splits) and `data/tournament_items.csv` (cold-start items) are produced by the pipeline. They contain question texts, so they are excluded from version control as well.
