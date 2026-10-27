# Evaluation report

Gold file SHA-256: `7bdfd66667bcbf2b5250540d7c18b786bf36c21ee24217f31c1874f48fad30cd`. LLM pass: no. spaCy NER: on.

## DEV split

Backend: spacy:en_core_web_sm

| scope     | docs | gold | pred | tp | fp | fn | P     | R     | F1    | owner acc | due acc |
| --------- | ---- | ---- | ---- | -- | -- | -- | ----- | ----- | ----- | --------- | ------- |
| overall   | 24   | 90   | 88   | 87 | 1  | 3  | 0.989 | 0.967 | 0.978 | 0.943     | 1.000   |
| asr       | 3    | 13   | 12   | 12 | 0  | 1  | 1.000 | 0.923 | 0.960 | 0.667     | 1.000   |
| checklist | 3    | 11   | 11   | 11 | 0  | 0  | 1.000 | 1.000 | 1.000 | 0.909     | 1.000   |
| notes     | 3    | 10   | 10   | 10 | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |
| speaker   | 3    | 17   | 17   | 17 | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |
| srt       | 3    | 10   | 9    | 9  | 0  | 1  | 1.000 | 0.900 | 0.947 | 1.000     | 1.000   |
| teams     | 3    | 9    | 9    | 8  | 1  | 1  | 0.889 | 0.889 | 0.889 | 1.000     | 1.000   |
| vtt       | 3    | 9    | 9    | 9  | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |
| zoom      | 3    | 11   | 11   | 11 | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |

### Errors by document

- `dev-teams-2`
  - spurious: We reopen the old incident
- `dev-teams-3`
  - missed: Own the vendor shortlist
- `dev-srt-1`
  - missed: Review the security questionnaire
- `dev-asr-1`
  - missed: Test the export flow

## TEST split

Backend: spacy:en_core_web_sm

| scope     | docs | gold | pred | tp | fp | fn | P     | R     | F1    | owner acc | due acc |
| --------- | ---- | ---- | ---- | -- | -- | -- | ----- | ----- | ----- | --------- | ------- |
| overall   | 24   | 100  | 98   | 96 | 2  | 4  | 0.980 | 0.960 | 0.970 | 0.958     | 1.000   |
| asr       | 3    | 14   | 14   | 14 | 0  | 0  | 1.000 | 1.000 | 1.000 | 0.714     | 1.000   |
| checklist | 3    | 13   | 13   | 13 | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |
| notes     | 3    | 11   | 11   | 11 | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |
| speaker   | 3    | 15   | 15   | 14 | 1  | 1  | 0.933 | 0.933 | 0.933 | 1.000     | 1.000   |
| srt       | 3    | 10   | 10   | 10 | 0  | 0  | 1.000 | 1.000 | 1.000 | 1.000     | 1.000   |
| teams     | 3    | 14   | 13   | 13 | 0  | 1  | 1.000 | 0.929 | 0.963 | 1.000     | 1.000   |
| vtt       | 3    | 11   | 11   | 10 | 1  | 1  | 0.909 | 0.909 | 0.909 | 1.000     | 1.000   |
| zoom      | 3    | 12   | 11   | 11 | 0  | 1  | 1.000 | 0.917 | 0.957 | 1.000     | 1.000   |

### Errors by document

- `test-speaker-1`
  - spurious: Get going with the quarterly numbers
- `test-speaker-2`
  - missed: Look at the terms
- `test-zoom-1`
  - missed: Look at the release checklist
- `test-teams-1`
  - missed: Check if the reserved instances expire soon
- `test-vtt-2`
  - missed: Rewrite the headline
  - spurious: Rewrite it
