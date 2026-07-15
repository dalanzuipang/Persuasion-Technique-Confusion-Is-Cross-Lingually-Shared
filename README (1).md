# Data

Article texts and gold span annotations are **not redistributed here** for
licence reasons. Obtain them from the original sources and place them as shown
below; notebook `01` asserts the resulting counts.

## Sources

1. **SemEval-2023 Task 3** — https://propaganda.math.unipd.it/semeval2023task3/
2. **CheckThat! 2024 Task 3** — https://gitlab.com/checkthat_lab/clef2024-checkthat-lab

The evaluation set is the intersection of the SemEval 2023 dev set and the
CheckThat! 2024 overlap subset (paper §III-D): the same articles, with
character-level spans taken from CheckThat!.

## Layout

```
data/
├── techniques_subtask3.txt              # 23 technique names, one per line (provided)
├── articles/
│   ├── en/article813452859.txt  ...     #  90 files
│   ├── pl/article25106.txt      ...     #  49 files
│   └── ru/article24100.txt      ...     #  48 files
└── gold/
    ├── en/labels-subtask-3-spans.txt    # TSV: article_id, technique, start, end
    ├── pl/labels-subtask-3-spans.txt
    └── ru/labels-subtask-3-spans.txt
```

Article filenames may carry an `article` prefix or not; `normalize_id` strips it.
Character offsets are `[start, end)`, half-open, character-level.

## ⚠️ Polish gold is duplicated upstream

The CheckThat! 2024 Polish dev span file ships with **every row duplicated**
(1,970 rows = 985 unique × 2). `load_spans` de-duplicates on
`(article_id, technique, start, end)`. **All Polish numbers in the paper use the
de-duplicated gold.** If you skip this, every PL result will differ.

## Expected counts (asserted by notebook 01)

| | Articles | Paragraphs | Gold spans |
|---|---:|---:|---:|
| EN | 90 | 3,127 | 1,801 |
| PL | 49 | 779 | 985 |
| RU | 48 | 503 | 739 |

Paragraphs follow the detectors' own segmentation (paper §III-C):
`content.splitlines()` → drop blank lines → drop lines longer than 1,000
characters. That filter discards 11.9% of PL and 8.7% of RU gold spans, which
are correspondingly unreachable by the detectors.

## Predictions

The nine prediction files in `../predictions/` are ours and **are** included.
Format: 4-column TSV, no header — `article_id`, `technique`, `start`, `end`.
Each is the output of a two-stage pipeline (paragraph assignment + span
localisation); see paper §III-D.
