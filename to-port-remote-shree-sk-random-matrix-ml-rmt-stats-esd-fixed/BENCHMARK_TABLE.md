# Final benchmark — OLD vs NEW vs FIXED

Repo's own `benchmark.py` ensembles and seeds (`default_rng(1000+seed)`), 6 seeds.
The OLD and NEW columns reproduce the delivered `benchmark_results.csv` exactly
(NEW median 0.0319 / max 0.6034 at 3 seeds), which validates the harness.

## Aggregate (Sigma^2 and Delta_3 rows, n=56)

| scoring | median | mean | max |
|---|---|---|---|
| OLD vs asymptote | 0.0421 | 2.3348 | 61.6636 |
| NEW vs asymptote (as delivered) | 0.0274 | 0.0613 | 0.6270 |
| NEW vs exact law | 0.0123 | 0.0459 | 0.6270 |
| **FIXED vs exact law** | 0.0102 | 0.0140 | 0.0477 |

## Per case

| case | NEW mean/max | FIXED mean/max | coordinate | stripped |
|---|---|---|---|---|
| GOE eigenvalues (N=1500) | 0.0365 / 0.1033 | **0.0178 / 0.0329** | identity | 4 |
| Poisson levels (N=6000) | 0.0106 / 0.0480 | **0.0079 / 0.0295** | identity | 0 |
| Wishart singular values, nu domain (3584x1024) | 0.0356 / 0.1113 | **0.0145 / 0.0381** | identity | 0 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | 0.1862 / 0.6270 | **0.0139 / 0.0310** | sqrt | 0 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | 0.0287 / 0.0988 | **0.0165 / 0.0477** | identity | 5 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | 0.0339 / 0.1157 | **0.0135 / 0.0319** | identity | 0 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | 0.0973 / 0.2524 | **0.0137 / 0.0300** | sqrt | 10 |

## Every row

| case | stat | L | exact | OLD | NEW | FIXED | NEW err | FIXED err |
|---|---|---|---|---|---|---|---|---|
| GOE eigenvalues (N=1500) | Sigma^2 | 5 | 0.7684 | 0.7981 | 0.7896 | 0.7838 | 0.0279 | 0.0201 |
| GOE eigenvalues (N=1500) | Sigma^2 | 10 | 0.9087 | 0.9384 | 0.9419 | 0.9372 | 0.0366 | 0.0314 |
| GOE eigenvalues (N=1500) | Sigma^2 | 20 | 1.0491 | 1.1410 | 1.0931 | 1.0836 | 0.0420 | 0.0329 |
| GOE eigenvalues (N=1500) | Sigma^2 | 50 | 1.2348 | 1.1474 | 1.2157 | 1.2069 | 0.0155 | 0.0226 |
| GOE eigenvalues (N=1500) | Delta_3 | 5 | 0.1738 | 0.1719 | 0.1722 | 0.1723 | 0.1033 | 0.0085 |
| GOE eigenvalues (N=1500) | Delta_3 | 10 | 0.2357 | 0.2333 | 0.2332 | 0.2329 | 0.0305 | 0.0120 |
| GOE eigenvalues (N=1500) | Delta_3 | 20 | 0.3014 | 0.3020 | 0.3021 | 0.3020 | 0.0185 | 0.0020 |
| GOE eigenvalues (N=1500) | Delta_3 | 50 | 0.3914 | 0.3969 | 0.3965 | 0.3965 | 0.0181 | 0.0129 |
| Poisson levels (N=6000) | Sigma^2 | 5 | 5.0000 | 5.0266 | 5.0009 | 4.9937 | 0.0002 | 0.0013 |
| Poisson levels (N=6000) | Sigma^2 | 10 | 10.0000 | 10.1223 | 10.0099 | 10.0391 | 0.0010 | 0.0039 |
| Poisson levels (N=6000) | Sigma^2 | 20 | 20.0000 | 19.8574 | 19.7756 | 19.8961 | 0.0112 | 0.0052 |
| Poisson levels (N=6000) | Sigma^2 | 50 | 50.0000 | 47.2048 | 47.5995 | 48.5243 | 0.0480 | 0.0295 |
| Poisson levels (N=6000) | Delta_3 | 5 | 0.3333 | 0.3341 | 0.3331 | 0.3308 | 0.0007 | 0.0075 |
| Poisson levels (N=6000) | Delta_3 | 10 | 0.6667 | 0.6682 | 0.6660 | 0.6601 | 0.0011 | 0.0098 |
| Poisson levels (N=6000) | Delta_3 | 20 | 1.3333 | 1.3304 | 1.3269 | 1.3363 | 0.0049 | 0.0022 |
| Poisson levels (N=6000) | Delta_3 | 50 | 3.3333 | 3.3920 | 3.3920 | 3.3454 | 0.0176 | 0.0036 |
| Wishart singular values, nu domain (3584x1024) | Sigma^2 | 5 | 0.7684 | 0.7766 | 0.7771 | 0.7765 | 0.0117 | 0.0106 |
| Wishart singular values, nu domain (3584x1024) | Sigma^2 | 10 | 0.9087 | 0.9431 | 0.9331 | 0.9336 | 0.0269 | 0.0274 |
| Wishart singular values, nu domain (3584x1024) | Sigma^2 | 20 | 1.0491 | 1.1173 | 1.0752 | 1.0759 | 0.0248 | 0.0255 |
| Wishart singular values, nu domain (3584x1024) | Sigma^2 | 50 | 1.2348 | 1.0643 | 1.1874 | 1.1878 | 0.0384 | 0.0381 |
| Wishart singular values, nu domain (3584x1024) | Delta_3 | 5 | 0.1738 | 0.1734 | 0.1735 | 0.1736 | 0.1113 | 0.0014 |
| Wishart singular values, nu domain (3584x1024) | Delta_3 | 10 | 0.2357 | 0.2353 | 0.2355 | 0.2361 | 0.0403 | 0.0017 |
| Wishart singular values, nu domain (3584x1024) | Delta_3 | 20 | 0.3014 | 0.3020 | 0.3024 | 0.3019 | 0.0196 | 0.0017 |
| Wishart singular values, nu domain (3584x1024) | Delta_3 | 50 | 0.3914 | 0.3955 | 0.3940 | 0.3950 | 0.0119 | 0.0093 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Sigma^2 | 5 | 0.7684 | 0.8382 | 0.8337 | 0.7716 | 0.0853 | 0.0041 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Sigma^2 | 10 | 0.9087 | 1.1521 | 1.0929 | 0.9314 | 0.2028 | 0.0250 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Sigma^2 | 20 | 1.0491 | 1.4299 | 1.4567 | 1.0810 | 0.3886 | 0.0304 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Sigma^2 | 50 | 1.2348 | 2.1677 | 2.0090 | 1.1966 | 0.6270 | 0.0310 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Delta_3 | 5 | 0.1738 | 0.1727 | 0.1727 | 0.1729 | 0.1060 | 0.0053 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Delta_3 | 10 | 0.2357 | 0.2339 | 0.2340 | 0.2345 | 0.0336 | 0.0052 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Delta_3 | 20 | 0.3014 | 0.3001 | 0.3006 | 0.3009 | 0.0136 | 0.0016 |
| Wishart eigenvalue domain lambda=nu^2/N (3584x1024) | Delta_3 | 50 | 0.3914 | 0.4025 | 0.4020 | 0.3947 | 0.0323 | 0.0084 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Sigma^2 | 5 | 0.7684 | 1.0028 | 0.7620 | 0.7576 | 0.0080 | 0.0140 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Sigma^2 | 10 | 0.9087 | 1.6762 | 0.9305 | 0.9289 | 0.0240 | 0.0222 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Sigma^2 | 20 | 1.0491 | 2.6907 | 1.0414 | 1.0367 | 0.0074 | 0.0118 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Sigma^2 | 50 | 1.2348 | 3.8346 | 1.1788 | 1.1759 | 0.0453 | 0.0477 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Delta_3 | 5 | 0.1738 | 0.1693 | 0.1715 | 0.1707 | 0.0988 | 0.0182 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Delta_3 | 10 | 0.2357 | 0.2312 | 0.2322 | 0.2325 | 0.0258 | 0.0135 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Delta_3 | 20 | 0.3014 | 0.3051 | 0.3005 | 0.3016 | 0.0132 | 0.0005 |
| Wishart + 20 heavy spikes, nu domain (3584x1024) | Delta_3 | 50 | 0.3914 | 0.4187 | 0.3922 | 0.3929 | 0.0072 | 0.0039 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Sigma^2 | 5 | 0.7684 | 0.7585 | 0.7592 | 0.7586 | 0.0117 | 0.0127 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Sigma^2 | 10 | 0.9087 | 0.9246 | 0.9154 | 0.9156 | 0.0074 | 0.0076 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Sigma^2 | 20 | 1.0491 | 1.0637 | 1.0718 | 1.0725 | 0.0217 | 0.0223 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Sigma^2 | 50 | 1.2348 | 1.2780 | 1.2055 | 1.2066 | 0.0237 | 0.0228 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Delta_3 | 5 | 0.1738 | 0.1742 | 0.1742 | 0.1743 | 0.1157 | 0.0026 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Delta_3 | 10 | 0.2357 | 0.2362 | 0.2361 | 0.2362 | 0.0431 | 0.0021 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Delta_3 | 20 | 0.3014 | 0.2997 | 0.2997 | 0.2996 | 0.0106 | 0.0059 |
| SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape] | Delta_3 | 50 | 0.3914 | 0.4053 | 0.4040 | 0.4039 | 0.0374 | 0.0319 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Sigma^2 | 5 | 0.7684 | 5.4013 | 0.7400 | 0.7563 | 0.0367 | 0.0158 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Sigma^2 | 10 | 0.9087 | 14.2496 | 0.8460 | 0.9173 | 0.0689 | 0.0095 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Sigma^2 | 20 | 1.0491 | 35.9771 | 0.8584 | 1.0726 | 0.1818 | 0.0224 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Sigma^2 | 50 | 1.2348 | 77.3760 | 0.9231 | 1.2080 | 0.2524 | 0.0217 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Delta_3 | 5 | 0.1738 | 0.2070 | 0.1754 | 0.1751 | 0.1236 | 0.0073 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Delta_3 | 10 | 0.2357 | 0.3311 | 0.2349 | 0.2356 | 0.0377 | 0.0004 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Delta_3 | 20 | 0.3014 | 0.8075 | 0.2976 | 0.3007 | 0.0034 | 0.0024 |
| SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024) | Delta_3 | 50 | 0.3914 | 2.1976 | 0.3605 | 0.4031 | 0.0743 | 0.0300 |
