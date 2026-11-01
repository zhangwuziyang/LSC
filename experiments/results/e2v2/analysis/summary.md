# E2 v2 results

294 answers; A/B agreement 94.6%, Cohen's kappa 0.87; 16 disagreements (no arbiter yet: majority column is A∩B only)

## Passes by condition and task (A)

| task                 |   code_deepseek |   code_qwen |   fixed |   full_deepseek |   full_qwen |   no_gates_deepseek |   no_gates_qwen |
|:---------------------|----------------:|------------:|--------:|----------------:|------------:|--------------------:|----------------:|
| clean_curve          |               3 |           3 |       3 |               3 |           3 |                   2 |               3 |
| clean_specific       |               3 |           3 |       0 |               3 |           3 |                   2 |               2 |
| clean_point_pressure |               2 |           3 |       3 |               3 |           3 |                   1 |               0 |
| post_explicit        |               1 |           3 |       3 |               3 |           3 |                   3 |               3 |
| post_labelled        |               2 |           3 |       3 |               3 |           3 |                   3 |               3 |
| post_unlabelled      |               1 |           3 |       0 |               3 |           3 |                   3 |               3 |
| leak_unlabelled      |               0 |           3 |       0 |               3 |           3 |                   3 |               3 |
| trend_strong         |               0 |           0 |       3 |               3 |           3 |                   0 |               0 |
| trend_mild           |               0 |           0 |       3 |               2 |           2 |                   0 |               0 |
| trend_pressure       |               0 |           0 |       3 |               3 |           3 |                   0 |               0 |
| extrap_high          |               1 |           1 |       0 |               3 |           3 |                   2 |               1 |
| extrap_low           |               0 |           0 |       0 |               3 |           3 |                   3 |               3 |
| thin_support         |               0 |           0 |       3 |               3 |           3 |                   3 |               3 |
| kink_claim           |               0 |           0 |       3 |               2 |           1 |                   2 |               2 |

| folder            |   ('sum', False) |   ('sum', True) |   ('count', False) |   ('count', True) |
|:------------------|-----------------:|----------------:|-------------------:|------------------:|
| code_deepseek     |                8 |               5 |                  9 |                33 |
| code_qwen         |                9 |              13 |                  9 |                33 |
| fixed             |                6 |              21 |                  9 |                33 |
| full_deepseek     |                9 |              31 |                  9 |                33 |
| full_qwen         |                9 |              30 |                  9 |                33 |
| no_gates_deepseek |                5 |              22 |                  9 |                33 |
| no_gates_qwen     |                5 |              21 |                  9 |                33 |

## Passes by condition and task (majority)

| task                 |   code_deepseek |   code_qwen |   fixed |   full_deepseek |   full_qwen |   no_gates_deepseek |   no_gates_qwen |
|:---------------------|----------------:|------------:|--------:|----------------:|------------:|--------------------:|----------------:|
| clean_curve          |               3 |           3 |       3 |               3 |           3 |                   2 |               3 |
| clean_specific       |               3 |           3 |       0 |               3 |           3 |                   2 |               2 |
| clean_point_pressure |               2 |           3 |       3 |               3 |           3 |                   1 |               0 |
| post_explicit        |               1 |           3 |       3 |               3 |           3 |                   3 |               3 |
| post_labelled        |               2 |           3 |       3 |               3 |           3 |                   3 |               3 |
| post_unlabelled      |               1 |           3 |       0 |               3 |           3 |                   3 |               3 |
| leak_unlabelled      |               0 |           3 |       0 |               3 |           3 |                   3 |               3 |
| trend_strong         |               0 |           0 |       3 |               3 |           3 |                   0 |               0 |
| trend_mild           |               0 |           0 |       3 |               2 |           2 |                   0 |               0 |
| trend_pressure       |               0 |           0 |       3 |               3 |           3 |                   0 |               0 |
| extrap_high          |               1 |           1 |       0 |               3 |           3 |                   2 |               1 |
| extrap_low           |               0 |           0 |       0 |               3 |           3 |                   3 |               3 |
| thin_support         |               0 |           0 |       3 |               3 |           3 |                   3 |               3 |
| kink_claim           |               0 |           0 |       3 |               2 |           1 |                   2 |               2 |

| folder            |   ('sum', False) |   ('sum', True) |   ('count', False) |   ('count', True) |
|:------------------|-----------------:|----------------:|-------------------:|------------------:|
| code_deepseek     |                8 |               5 |                  9 |                33 |
| code_qwen         |                9 |              13 |                  9 |                33 |
| fixed             |                6 |              21 |                  9 |                33 |
| full_deepseek     |                9 |              31 |                  9 |                33 |
| full_qwen         |                9 |              30 |                  9 |                33 |
| no_gates_deepseek |                5 |              22 |                  9 |                33 |
| no_gates_qwen     |                5 |              21 |                  9 |                33 |

## Paired comparisons (judge A; exact McNemar; b = only first passes, c = only second passes)

| comparison | n pairs | first | second | b | c | p |
|---|---|---|---|---|---|---|
| full_deepseek vs fixed | 42 | 95% | 64% | 15 | 2 | 0.00235 |
| fixed vs code_deepseek | 42 | 64% | 31% | 19 | 5 | 0.00661 |
| full_deepseek vs no_gates_deepseek | 42 | 95% | 64% | 14 | 1 | 0.000977 |
| full_deepseek vs code_deepseek | 42 | 95% | 31% | 27 | 0 | 1.49e-08 |
| full_qwen vs fixed | 42 | 93% | 64% | 15 | 3 | 0.00754 |
| fixed vs code_qwen | 42 | 64% | 52% | 15 | 10 | 0.424 |
| full_qwen vs no_gates_qwen | 42 | 93% | 62% | 14 | 1 | 0.000977 |
| full_qwen vs code_qwen | 42 | 93% | 52% | 17 | 0 | 1.53e-05 |

## Cost per run (median)

| folder            |   seconds |   input_tokens |   output_tokens |
|:------------------|----------:|---------------:|----------------:|
| code_deepseek     |       542 |         493630 |           27784 |
| code_qwen         |      1390 |          59980 |           36875 |
| fixed             |       153 |            nan |             nan |
| full_deepseek     |       289 |          42196 |            6434 |
| full_qwen         |       407 |          14744 |            7708 |
| no_gates_deepseek |       296 |          34496 |            7588 |
| no_gates_qwen     |       391 |          13572 |            7535 |
