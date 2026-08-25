| family | class | pairs | median rel | IQR of rel | median abs-rel | sign-disagree | rho vs progress | runs w/ rho>0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `curvature/gg` | quadratic | 215 | -0.00927% | -0.202% … 0.157% | 0.181% | 0.0% | +0.29* | 7/7 |
| `curvature/vhv_update` | quadratic | 215 | -0.0155% | -0.191% … 0.253% | 0.208% | 0.0% | -0.17 | 0/7 |
| `curvature/eta_star_rho` | quadratic | 215 | 0.0334% | -0.122% … 0.309% | 0.211% | 0.0% | -0.26* | 0/7 |
| `curvature/dhd` | quadratic | 215 | -0.00613% | -0.196% … 0.256% | 0.219% | 0.0% | -0.26* | 0/7 |
| `curvature/vhv_random` | quadratic | 215 | -0.104% | -0.498% … 0.329% | 0.435% | 0.0% | -0.16 | 1/7 |
| `curvature/Hg_norm` | quadratic | 215 | -0.0345% | -0.609% … 0.442% | 0.524% | 0.0% | +0.11 | 7/7 |
| `curvature/eta_star` | quadratic | 186 | 0.0188% | -0.66% … 0.587% | 0.605% | 0.0% | -0.03 | 4/7 |
| `curvature/vhv_gradient` | quadratic | 215 | 0.0514% | -0.568% … 1.14% | 0.673% | 0.0% | -0.20* | 0/7 |
| `curvature/gHg` | quadratic | 215 | 0.112% | -0.65% … 1.43% | 0.935% | 0.0% | -0.13 | 2/7 |
| `update/direction_norm` | update | 215 | 0% | 0% … 0% | 0% | 0.0% | n/a | — |
| `update/loss_before` | update | 215 | 0.00118% | -0.0191% … 0.029% | 0.0242% | 0.0% | +0.40* | 7/7 |
| `update/loss_after` | update | 215 | 0.00568% | -0.0203% … 0.0317% | 0.0281% | 0.0% | +0.34* | 7/7 |
| `update/p1` | update | 215 | 0.0033% | -0.347% … 0.352% | 0.351% | 0.5% | +0.46* | 7/7 |
| `update/p2` | update | 215 | -0.0151% | -0.483% … 0.444% | 0.461% | 0.5% | +0.34* | 7/7 |
| `update/residual_p1` | update | 215 | 0.396% | -2.22% … 3.73% | 2.77% | 2.8% | +0.36* | 7/7 |
| `update/actual` | update | 215 | 0.299% | -2.74% … 2.86% | 2.8% | 3.3% | +0.63* | 7/7 |
| `update/normalized_residual` | update | 215 | 2.15% | -52.3% … 1.75 | 77.9% | 20.5% | +0.40* | 7/7 |
| `update/residual_p2` | update | 215 | 2.94% | -52.4% … 1.83 | 83.7% | 20.5% | +0.42* | 7/7 |
| `curvature/c_fd_gradient` | error | 215 | -1.02% | -5.07% … 2.54% | 3.94% | 5.6% | -0.66* | 0/7 |
| `curvature/c_fd_random` | error | 215 | -1.56 | -1,243 … 498 | 688 | 53.0% | +0.37* | 7/7 |
| `curvature/c_fd_update` | error | 215 | -3.46 | -2,195 … 1,397 | 1,854 | 49.8% | +0.42* | 7/7 |
| `curvature/e_sym_gradient` | error | 215 | 6,834 | 2,013 … 23,620 | 6,834 | 0.0% | +0.03 | 4/7 |
| `curvature/e_lin_gradient` | error | 215 | 10,589 | 8,894 … 11,880 | 10,589 | 0.0% | +0.34* | 7/7 |
| `curvature/e_sym_random` | error | 215 | 11,859 | 4,957 … 36,931 | 11,859 | 0.0% | -0.14 | 3/7 |
| `curvature/e_sym_update` | error | 215 | 13,137 | 4,347 … 51,127 | 13,137 | 0.0% | -0.54* | 0/7 |
| `curvature/e_lin_update` | error | 215 | 14,464 | 12,917 … 17,460 | 14,464 | 0.0% | -0.80* | 0/7 |
| `curvature/e_lin_random` | error | 215 | 14,474 | 13,254 … 15,462 | 14,474 | 0.0% | -0.67* | 0/7 |
| `curvature/e_curv_gradient` | error | **0** | — | — | — | — | — | — |
| `curvature/e_curv_random` | error | **0** | — | — | — | — | — | — |
| `curvature/e_curv_update` | error | **0** | — | — | — | — | — | — |
| `curvature/e_fd_gradient` | error | **0** | — | — | — | — | — | — |
| `curvature/e_fd_random` | error | **0** | — | — | — | — | — | — |
| `curvature/e_fd_update` | error | **0** | — | — | — | — | — | — |
| `curvature/fd_cos_gradient` | error | **0** | — | — | — | — | — | — |
| `curvature/fd_cos_random` | error | **0** | — | — | — | — | — | — |
| `curvature/fd_cos_update` | error | **0** | — | — | — | — | — | — |
| `curvature/curv_snr_update` | snr | 215 | -94.5% | -98% … -89.5% | 94.5% | 2.3% | -0.67* | 0/7 |
| `curvature/curv_snr_random` | snr | 215 | -94.5% | -97.8% … -91.1% | 94.5% | 2.8% | -0.67* | 0/7 |
| `curvature/fd_snr_update` | snr | 215 | -99.4% | -99.8% … -99.2% | 99.4% | 0.0% | -0.88* | 0/7 |
| `curvature/curv_snr_gradient` | snr | 215 | -99.8% | -99.9% … -99.3% | 99.8% | 0.0% | -0.91* | 0/7 |
| `curvature/fd_snr_random` | snr | 215 | -99.9% | -99.9% … -99.9% | 99.9% | 0.0% | -0.55* | 0/7 |
| `curvature/fd_snr_gradient` | snr | 215 | -100% | -100% … -100% | 100% | 0.0% | -0.62* | 0/7 |
| `curvature/fd_eps_update` | floor | 215 | 0% | -66.7% … 0% | 0% | 0.0% | +0.39* | 7/7 |
| `curvature/fd_eps_random` | floor | 215 | -66.7% | -90% … 0% | 66.7% | 0.0% | +0.48* | 7/7 |
| `curvature/curv_eps_update` | floor | 215 | 0% | -70% … 2.33 | 90% | 0.0% | -0.15 | 1/7 |
| `curvature/curv_eps_random` | floor | 215 | 0% | -66.7% … 9 | 97% | 0.0% | -0.02 | 4/7 |
| `curvature/curv_eps_gradient` | floor | 215 | 10.6 | 4.77 … 21.9 | 10.6 | 0.0% | +0.91* | 7/7 |
| `curvature/fd_eps_gradient` | floor | 215 | 29 | 9 … 29 | 29 | 0.0% | +0.77* | 7/7 |
| `curvature/curv_floor_gradient` | floor | 215 | 487 | 128 … 1,967 | 487 | 0.0% | -0.90* | 0/7 |
| `curvature/fd_floor_gradient` | floor | 215 | 2,203 | 2,185 … 6,557 | 2,203 | 0.0% | -0.57* | 0/7 |
| `curvature/curv_floor_random` | floor | 215 | 65,520 | 655 … 589,859 | 65,520 | 0.0% | +0.07 | 4/7 |
| `curvature/arith_eps` | floor | 215 | 65,535 | 65,535 … 65,535 | 65,535 | 0.0% | n/a | — |
| `curvature/eta_star_rho_threshold` | floor | 215 | 65,535 | 65,535 … 65,535 | 65,535 | 0.0% | n/a | — |
| `curvature/curv_floor_update` | floor | 215 | 65,535 | 5,893 … 728,189 | 65,535 | 0.0% | +0.22* | 6/7 |
| `curvature/fd_floor_update` | floor | 215 | 66,144 | 65,530 … 196,904 | 66,144 | 0.0% | +0.37* | 7/7 |
| `curvature/fd_floor_random` | floor | 215 | 195,918 | 65,531 … 655,885 | 195,918 | 0.0% | +0.48* | 7/7 |
| `curvature/fd_conclusive_gradient` | flag | 215 | -1 | -1 … -1 | 1 | 100.0% | n/a | — |
| `curvature/fd_conclusive_random` | flag | 215 | -1 | -1 … -1 | 1 | 18.6% | n/a | — |
| `curvature/fd_conclusive_update` | flag | 215 | -1 | -1 … -1 | 1 | 0.9% | n/a | — |
| `curvature/verdict_code_gradient` | flag | 215 | 1 | 1 … 1 | 1 | 86.5% | +0.28 | — |
| `curvature/verdict_code_random` | flag | 215 | 1 | 1 … 1 | 1 | 0.0% | -0.11 | 0/7 |
| `curvature/verdict_code_update` | flag | 215 | 1 | 1 … 1 | 1 | 0.0% | +0.00 | 1/7 |
