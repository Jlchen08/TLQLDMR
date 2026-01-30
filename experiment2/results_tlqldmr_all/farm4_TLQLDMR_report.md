# TL-QLDMR 深度寻优报告 - Wind farm site 4 (Nominal capacity-66MW)

## 结论摘要
- 经过 20 次随机搜索（seed=42）获得当前最优配置。
- 测试集指标：RMSE=12.4835，MAE=10.0541，MAPE=17115048.66%，R²=0.5924。
- 峰值点绝对误差=14.5662，谷值点绝对误差=46.5416。

## 最优配置
- C_S: 0.05
- C_T: 100.0
- feature_set: simple
- kernel_gamma: 0.003
- lambda1: 0.0001
- lambda2: 10.0
- nystrom_batch_size: 256
- nystrom_epochs: 24
- nystrom_lr: 0.003
- nystrom_n_components: 500
- target_train_ratio: 0.4
- window_size: 24

## 峰谷拟合观察
- 峰谷区域误差已量化于上方指标，建议结合峰谷局部放大图进行视觉核验。
- 若需进一步提升峰谷拟合，可在更小的滑窗、较高的核复杂度或更大 Nystrom 维度上继续搜索。
