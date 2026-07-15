# Predictions

Nine files, one per configuration: `{lang}_{method}.tsv` for
`lang ∈ {en, pl, ru}`, `method ∈ {sup_ft, prompt_a, iter_ens}`.

Format: 4-column TSV, **no header**:

```
article_id <TAB> technique <TAB> start <TAB> end
```

- `article_id` — pure digits (any `article` prefix is stripped by `normalize_id`)
- `technique` — one of the 23 names in `../data/techniques_subtask3.txt`
- `start`, `end` — character offsets, `[start, end)`, half-open

Each row is one predicted span from the two-stage pipeline: a paragraph-level
stage assigns techniques to paragraphs, and a localisation stage narrows each
assigned technique to a character span, inheriting the paragraph label.

Notebook `01` checks all nine are present and parseable.
