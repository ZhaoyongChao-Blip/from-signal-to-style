# LightGBM 训练结果


## y_raw

- label: y_raw
- n_train: 3940768
- n_valid: 801991
- n_test: 1483914
- best_iter: 25
- auc_train: 0.6604
- pos_rate_train: 0.4851
- ic_train: 0.111
- icir_train: 1.0485
- n_days_train: 1215
- auc_valid: 0.5463
- pos_rate_valid: 0.4614
- ic_valid: 0.0841
- icir_valid: 0.6324
- n_days_valid: 242
- auc_test: 0.5368
- pos_rate_test: 0.494
- ic_test: 0.0996
- icir_test: 0.692
- n_days_test: 448

逐年 walk-forward:

 walk_year  n_test  auc_test  ic_test  icir_test
      2021  804589    0.5237   0.0573     0.7420
      2022  801538    0.5424   0.0780     1.2454
      2023  801991    0.5563   0.0835     0.6936
      2024  801832    0.5290   0.0823     0.6722
      2025  682082    0.5550   0.0827     0.6733

## y_exc

- label: y_exc
- n_train: 3940768
- n_valid: 801991
- n_test: 1483914
- best_iter: 235
- auc_train: 0.6029
- pos_rate_train: 0.4347
- ic_train: 0.1901
- icir_train: 1.7721
- n_days_train: 1215
- auc_valid: 0.5283
- pos_rate_valid: 0.4506
- ic_valid: 0.0679
- icir_valid: 0.5592
- n_days_valid: 242
- auc_test: 0.5462
- pos_rate_test: 0.4313
- ic_test: 0.1074
- icir_test: 0.6481
- n_days_test: 448

逐年 walk-forward:

 walk_year  n_test  auc_test  ic_test  icir_test
      2021  804589    0.5282   0.0697     0.8704
      2022  801538    0.5294   0.0778     0.8156
      2023  801991    0.5283   0.0646     0.5684
      2024  801832    0.5270   0.0720     0.3408
      2025  682082    0.5392   0.1127     0.9524

## y_exc_nzt

- label: y_exc_nzt
- n_train: 3751482
- n_valid: 776778
- n_test: 1394162
- best_iter: 239
- auc_train: 0.6133
- pos_rate_train: 0.4111
- ic_train: 0.2254
- icir_train: 1.9985
- n_days_train: 1215
- auc_valid: 0.5424
- pos_rate_valid: 0.4354
- ic_valid: 0.1069
- icir_valid: 0.8218
- n_days_valid: 242
- auc_test: 0.562
- pos_rate_test: 0.4036
- ic_test: 0.1538
- icir_test: 0.9205
- n_days_test: 448

逐年 walk-forward:

 walk_year  n_test  auc_test  ic_test  icir_test
      2021  756049    0.5407   0.1104     1.2506
      2022  755911    0.5468   0.1283     1.2509
      2023  776778    0.5425   0.1050     0.8561
      2024  749821    0.5464   0.1188     0.5737
      2025  644341    0.5525   0.1680     1.2573

## 特征重要性（gain, y_exc Top20）

- ma_cross_20_60: 279,309
- ma_cross_5_20: 259,895
- turn_ma_5: 61,005
- amt_ma_20: 43,921
- vol_60: 35,961
- turn_ma_20: 35,552
- log_turn: 35,364
- ret_120: 34,286
- boll_width: 33,927
- vol_ma_ratio_20: 32,467
- corr_pv_20: 28,645
- hl_range_20: 27,337
- ma_dev_60: 25,829
- log_amt: 25,426
- ma_dev_5: 24,962
- vol_20: 23,805
- kdj_k: 23,750
- kdj_d: 23,742
- turn_ratio: 23,260
- ret_1: 22,089