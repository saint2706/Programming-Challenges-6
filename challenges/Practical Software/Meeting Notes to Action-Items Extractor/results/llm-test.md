# LLM refinement pass on TEST

Model: Qwen3-1.7B-Q8_0.gguf (Qwen/Qwen3-1.7B-GGUF, Apache-2.0), served by llama-server on
127.0.0.1:8080 with `--jinja --reasoning-budget 0 --reasoning-format none`. Run once per setting on the
100-item TEST split with spaCy on. The rules-only row is the committed TEST result in `report.md`.

| run                                              | pred | tp | fp | fn | P     | R     | F1    | owner acc | due acc |
| ------------------------------------------------ | ---- | -- | -- | -- | ----- | ----- | ----- | --------- | ------- |
| rules only (committed)                           | 98   | 96 | 2  | 4  | 0.980 | 0.960 | 0.970 | 0.958     | 1.000   |
| LLM, default `--min-confidence 0.4` (final code) | 98   | 96 | 2  | 4  | 0.980 | 0.960 | 0.970 | 0.760     | 0.854   |
| LLM, opt-in `--min-confidence 0.3`               | 140  | 98 | 42 | 2  | 0.700 | 0.980 | 0.817 | 0.745     | 0.847   |
| LLM, before security fixes (confidence 0.55)     | 138  | 99 | 39 | 1  | 0.717 | 0.990 | 0.832 | 0.747     | 0.818   |

The pre-fix row is from the earlier run, before the grounding check, the 0.35 confidence and the
threshold check on added items. The default row is the final code. At the default threshold the model
adds no items, so the only effect is owner and due fills on items the rules already found. Those fills
lower owner accuracy from 0.958 to 0.760 and due accuracy from 1.000 to 0.854.

Conclusion: in this setup the LLM pass does not improve any measured metric. F1 is unchanged at the
default threshold and falls at the opt-in threshold, mostly from added false positives. Owner and due
accuracy fall in both LLM runs. The rules-only pipeline stays the default.
