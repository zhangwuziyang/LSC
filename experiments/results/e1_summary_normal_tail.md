# E1 summary (n=20000, reps=20)

Target: theta(10, d) on d in [6, 16]. coverage = mean pointwise 95% CI coverage; band_covers = share of reps whose 95% uniform band covers the whole true curve; kink_err = |estimated kink - 10| in days; kink_covers = share of reps whose 95% kink CI contains 10.

|                                     |   rmse |   abs_bias |   coverage |   band_covers |   ci_width |   kink_err |   kink_covers |   kink_ci_width |   seconds |
|:------------------------------------|-------:|-----------:|-----------:|--------------:|-----------:|-----------:|--------------:|----------------:|----------:|
| ('normal_tail', 'E0+M0 (original)') |  0.175 |      0.132 |      0.123 |             0 |      0.057 |      0.3   |          0    |           0.4   |     1.442 |
| ('normal_tail', 'E0+M1')            |  0.297 |      0.23  |      0.002 |             0 |      0.055 |      0.225 |          0.85 |           0.45  |     2.622 |
| ('normal_tail', 'E1+M0')            |  0.085 |      0.072 |      0.952 |             1 |      0.44  |      1.2   |          0.9  |           4.327 |    46.527 |
| ('normal_tail', 'E1+M1 (proposed)') |  0.057 |      0.046 |      0.982 |             1 |      0.298 |      1.3   |          0.95 |           6.728 |    89.981 |
