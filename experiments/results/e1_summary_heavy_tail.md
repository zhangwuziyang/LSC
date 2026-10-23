# E1 summary (n=20000, reps=20)

Target: theta(10, d) on d in [6, 16]. coverage = mean pointwise 95% CI coverage; band_covers = share of reps whose 95% uniform band covers the whole true curve; kink_err = |estimated kink - 10| in days; kink_covers = share of reps whose 95% kink CI contains 10.

|                                    |   rmse |   abs_bias |   coverage |   band_covers |   ci_width |   kink_err |   kink_covers |   kink_ci_width |   seconds |
|:-----------------------------------|-------:|-----------:|-----------:|--------------:|-----------:|-----------:|--------------:|----------------:|----------:|
| ('heavy_tail', 'E0+M0 (original)') |  0.163 |      0.123 |      0.138 |             0 |      0.054 |      0.3   |          0.05 |           0.425 |     1.434 |
| ('heavy_tail', 'E0+M1')            |  0.287 |      0.227 |      0     |             0 |      0.053 |      0.225 |          0.7  |           0.394 |     2.825 |
| ('heavy_tail', 'E1+M0')            |  0.081 |      0.072 |      0.97  |             1 |      0.42  |      1.638 |          1    |           6.502 |    44.16  |
| ('heavy_tail', 'E1+M1 (proposed)') |  0.064 |      0.049 |      0.965 |             1 |      0.319 |      2.175 |          1    |           8.419 |    96.6   |
