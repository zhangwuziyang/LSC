# E2c: provider comparison

## Share of tasks passed

|                             |   clean |   extrapolation |   post_treatment |   thin_support |   trend_violation |   overall |
|:----------------------------|--------:|----------------:|-----------------:|---------------:|------------------:|----------:|
| ('deepseek', 'full')        |       1 |               1 |                1 |              1 |                 1 |       1   |
| ('deepseek', 'no_gates')    |       1 |               1 |                0 |              0 |                 0 |       0.4 |
| ('deepseek', 'no_semantic') |       1 |               1 |                1 |              1 |                 1 |       1   |
| ('qwen', 'full')            |       1 |               1 |                1 |              1 |                 1 |       1   |
| ('qwen', 'no_gates')        |       1 |               0 |                0 |              0 |                 0 |       0.2 |
| ('qwen', 'no_semantic')     |       1 |               1 |                1 |              1 |                 1 |       1   |

## Individual checks

|                             |   finished_with_report |   no_post_treatment_covariate |   all_numbers_verified |   explains_exclusion |   reports_caveat |   declines_extrapolation |
|:----------------------------|-----------------------:|------------------------------:|-----------------------:|---------------------:|-----------------:|-------------------------:|
| ('deepseek', 'full')        |                      1 |                           1   |                    1   |                    1 |                1 |                        1 |
| ('deepseek', 'no_gates')    |                      1 |                           0.8 |                    0.8 |                    1 |                0 |                        1 |
| ('deepseek', 'no_semantic') |                      1 |                           1   |                    1   |                    1 |                1 |                        1 |
| ('qwen', 'full')            |                      1 |                           1   |                    1   |                    1 |                1 |                        1 |
| ('qwen', 'no_gates')        |                      1 |                           1   |                    0.2 |                    1 |                1 |                        1 |
| ('qwen', 'no_semantic')     |                      1 |                           1   |                    1   |                    1 |                1 |                        1 |

## Cost per run

|                             |   turns |   tool_calls |   input_tokens |   output_tokens |   minutes |
|:----------------------------|--------:|-------------:|---------------:|----------------:|----------:|
| ('deepseek', 'full')        |     9   |          9.2 |        40225.4 |          5857.8 |       5.9 |
| ('deepseek', 'no_gates')    |     7.2 |          6.8 |        24835.4 |          4623.2 |       5.7 |
| ('deepseek', 'no_semantic') |    10   |         22.4 |        47545.8 |          8017   |       6.2 |
| ('qwen', 'full')            |     9   |         10   |        15325.4 |          8729.8 |       7.3 |
| ('qwen', 'no_gates')        |     8   |          9.4 |        13272   |          7450   |       6.8 |
| ('qwen', 'no_semantic')     |    10.6 |         16.4 |        15959.4 |          8951.8 |       7.5 |
