# 足球转会数据的多元统计计算结果

本文件由 `compute_multivariate_stats.py` 自动生成，数据源为 `football_transfer_analysis/Transfer Market Data.csv`。统计口径以 2022 年夏窗附近的转会记录为基础，主要变量包括年龄、市值、转会费、相对溢价强度、地理距离、球员类型和转会条件。

## 0. 图表文件

以下图表与本统计结果保存在同一目录：

- `03_chart_transfer_condition_counts.png`
- `04_chart_player_type_paid_share.png`
- `05_chart_pca_explained_variance.png`
- `06_chart_pca_scores_by_condition.png`
- `07_chart_mahalanobis_distance.png`
- `08_chart_hotelling_mean_differences.png`
- `09_chart_player_type_profiles.png`


## 1. 数据规模与基础事实

| metric | value |
| --- | --- |
| 全部转会记录 | 19,139.0000 |
| 完整多元分析样本 | 18,752.0000 |
| 付费转会样本 | 1,326.0000 |
| 免费转会比例 | 0.9307 |
| 跨国转会比例 | 0.2433 |
| 转会费中位数 EUR | 0.0000 |
| 市值中位数 EUR | 100,000.0000 |


## 2. 按球员类型的描述统计

| Player Type | n | mean_age | median_market_value | median_cost | paid_share | mean_distance_km |
| --- | --- | --- | --- | --- | --- | --- |
| Defender | 5,887.000 | 25.488 | 100,000.000 | 0.000 | 0.063 | 621.675 |
| Forward | 5,610.000 | 25.686 | 100,000.000 | 0.000 | 0.091 | 816.587 |
| Midfielder | 5,436.000 | 25.323 | 100,000.000 | 0.000 | 0.070 | 627.193 |
| Goalkeeper | 1,819.000 | 25.214 | 75,000.000 | 0.000 | 0.038 | 420.292 |


## 3. 按转会条件的描述统计

| Transfer Condition | n | mean_age | median_market_value | median_cost | mean_premium_eur | mean_distance_km |
| --- | --- | --- | --- | --- | --- | --- |
| Free Transfer | 17,426.000 | 25.582 | 100,000.000 | 0.000 | -241,796.167 | 578.425 |
| Underpaid Transfer | 881.000 | 24.658 | 1,500,000.000 | 700,000.000 | -1,694,965.607 | 1,533.337 |
| Overpaid Transfer | 445.000 | 22.813 | 1,500,000.000 | 2,730,000.000 | 2,612,177.528 | 2,211.860 |


## 4. Wishart / 协方差矩阵口径

若完整样本近似来自五维多元正态总体，则标准化后的样本协方差矩阵满足 `(n-1)S ~ Wishart_p(n-1, Sigma)`。这里的作用不是把 Wishart 当作结论，而是说明 PCA、Hotelling T^2、Mahalanobis 距离都建立在同一个协方差结构上。付费转会样本标准化协方差矩阵如下：

| feature | age | log_market_value | log_cost | premium_ratio_log | distance_km |
| --- | --- | --- | --- | --- | --- |
| age | 1.001 | 0.153 | -0.012 | -0.313 | 0.011 |
| log_market_value | 0.153 | 1.001 | 0.858 | -0.042 | 0.042 |
| log_cost | -0.012 | 0.858 | 1.001 | 0.443 | 0.091 |
| premium_ratio_log | -0.313 | -0.042 | 0.443 | 1.001 | 0.111 |
| distance_km | 0.011 | 0.042 | 0.091 | 0.111 | 1.001 |


## 5. PCA 结果

| component | eigenvalue | explained_variance_ratio | cumulative_ratio |
| --- | --- | --- | --- |
| PC1 | 1.9658 | 0.3929 | 0.3929 |
| PC2 | 1.3733 | 0.2745 | 0.6673 |
| PC3 | 0.9909 | 0.1980 | 0.8654 |
| PC4 | 0.6562 | 0.1311 | 0.9965 |
| PC5 | 0.0176 | 0.0035 | 1.0000 |


PCA loading：

| feature | PC1 | PC2 | PC3 | PC4 | PC5 |
| --- | --- | --- | --- | --- | --- |
| age | -0.0117 | 0.6747 | 0.2317 | 0.7007 | -0.0035 |
| log_market_value | 0.6187 | 0.3543 | -0.0968 | -0.3019 | -0.6255 |
| log_cost | 0.7074 | 0.0083 | -0.0871 | 0.0361 | 0.7005 |
| premium_ratio_log | 0.3162 | -0.6349 | -0.0016 | 0.6154 | -0.3437 |
| distance_km | 0.1294 | -0.1269 | 0.9640 | -0.1944 | 0.0007 |


## 6. 多元正态性：Mardia 检验

| statistic | value |
| --- | --- |
| n | 1,326.000000 |
| p | 5.000000 |
| mardia_skewness | 39.015237 |
| skew_chi_square | 8,622.367295 |
| skew_df | 35.000000 |
| skew_p | 0.000000 |
| mardia_kurtosis | 95.637634 |
| kurtosis_z | 131.957777 |
| kurtosis_p | 0.000000 |


解释：p 值很小，说明转会数据不服从理想多元正态。论文中因此应把正态模型作为诊断基准，而不是把它当成严格真实的数据生成机制。

## 7. Mahalanobis 距离异常记录

| metric | value |
| --- | --- |
| Mahalanobis D^2 97.5% 阈值 | 12.8325 |
| Mahalanobis D^2 99% 阈值 | 15.0863 |
| 超过 97.5% 阈值比例 | 0.0596 |
| 超过 99% 阈值比例 | 0.0400 |


| name | age | Player Type | origin_club | new_club | player_market_value | cost | distance_km | Transfer Condition | mahalanobis_d2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kofi Balmer | 21.000 | Defender | Larne FC | Crystal Palace U21 | 150,000.000 | 300.000 | 401.849 | Underpaid Transfer | 176.588 |
| Arthur Chaves | 21.000 | Defender | Avaí Futebol Clube (SC) | Académico Viseu FC | 50,000.000 | 2,400,000.000 | 7,486.765 | Overpaid Transfer | 128.117 |
| Roman Evgenjev | 23.000 | Defender | Dynamo Moscow | Krylya Sovetov Samara | 3,500,000.000 | 25,000.000 | 0.000 | Underpaid Transfer | 102.195 |
| Alejandro Pozuelo | 30.000 | Midfielder | Toronto FC | Inter Miami CF | 8,000,000.000 | 136,000.000 | 2,261.536 | Underpaid Transfer | 62.912 |
| Roggerio Nyakossi | 18.000 | Defender | Servette FC | Olympique de Marseille B | 100,000.000 | 2,000,000.000 | 464.652 | Overpaid Transfer | 61.637 |
| Cesare Casadei | 19.000 | Midfielder | Inter Milan Primavera | Chelsea FC U21 | 1,000,000.000 | 15,000,000.000 | 1,556.899 | Overpaid Transfer | 47.148 |
| Marco Delle Monache | 17.000 | Forward | Delfino Pescara 1936 | UC Sampdoria | 100,000.000 | 1,500,000.000 | 0.000 | Overpaid Transfer | 46.759 |
| Mario Gila | 21.000 | Defender | Real Madrid Castilla | SS Lazio | 500,000.000 | 6,000,000.000 | 1,372.635 | Overpaid Transfer | 36.767 |
| Alfie Pond | 18.000 | Defender | Exeter City | Wolverhampton Wanderers U21 | 50,000.000 | 580,000.000 | 0.000 | Overpaid Transfer | 36.628 |
| Nitai Greis | 18.000 | Goalkeeper | Udinese Primavera | Maccabi Haifa U19 | 10,000.000 | 100,000.000 | 2,317.705 | Overpaid Transfer | 34.792 |


## 8. Hotelling T^2：付费转会与免费转会的多元均值差异

| statistic | value |
| --- | --- |
| T^2 | 1,113,386.684879 |
| F | 278,302.135752 |
| df1 | 4.000000 |
| df2 | 18,747.000000 |
| p_value | 0.000000 |


| variable | paid_standardized_mean | free_standardized_mean | difference_paid_minus_free |
| --- | --- | --- | --- |
| age | -0.3133 | 0.0238 | -0.3371 |
| log_market_value | 1.8552 | -0.1412 | 1.9964 |
| log_cost | 3.5937 | -0.2735 | 3.8671 |
| distance_km | 0.5450 | -0.0415 | 0.5865 |


## 9. MANOVA：球员类型是否对应不同的多元转会画像

| statistic | value |
| --- | --- |
| n | 1,326.000000 |
| p | 4.000000 |
| groups | 4.000000 |
| wilks_lambda | 0.955080 |
| bartlett_chi_square | 60.737047 |
| df | 12.000000 |
| p_value | 0.000000 |
| hypothesis_df | 3.000000 |
| error_df | 1,322.000000 |


解释：Wilks' Lambda 越小，组间差异越明显。这里的 p 值显示，Forward、Midfielder、Defender、Goalkeeper 的年龄、市场价值、转会费和距离构成的联合画像显著不同。

## 10. 主要国家流向

| country_origin_club | country_new_club | n | median_cost | mean_distance_km |
| --- | --- | --- | --- | --- |
| Italy | Italy | 2,684.000 | 0.000 | 0.000 |
| Spain | Spain | 1,562.000 | 0.000 | 0.000 |
| Germany | Germany | 946.000 | 0.000 | 0.000 |
| Russia | Russia | 738.000 | 0.000 | 0.000 |
| Romania | Romania | 439.000 | 0.000 | 0.000 |
| Serbia | Serbia | 439.000 | 0.000 | 0.000 |
| Turkey | Turkey | 414.000 | 0.000 | 0.000 |
| France | France | 405.000 | 0.000 | 0.000 |
| Iran | Iran | 399.000 | 0.000 | 0.000 |
| Greece | Greece | 355.000 | 0.000 | 0.000 |
| England | England | 342.000 | 0.000 | 0.000 |
| Bosnia-Herzegovina | Bosnia-Herzegovina | 283.000 | 0.000 | 0.000 |


## 11. 可直接放进论文的统计结论

- 转会费分布高度偏态，免费转会占绝大多数，因此对金额变量使用 `log(1+x)` 是必要的。

- PCA 的前两个主成分解释了主要协方差结构，可理解为“价格/价值规模轴”和“年龄-价格错位或地理摩擦轴”。

- Mardia 检验拒绝多元正态，提醒我们不能机械依赖正态假设；但多元正态仍可作为 Hotelling、Mahalanobis 和 Wishart 解释的理论起点。

- Mahalanobis 距离能识别普通单变量排序看不出的联合异常值，例如年龄、市值、费用与跨国距离同时不寻常的转会。

- Hotelling T^2 显示付费转会和免费转会在多元均值上显著不同，说明“是否付费”不是单纯费用变量，而代表完全不同的市场机制。

- MANOVA 显示球员类型与联合转会画像显著相关，支持论文中“市场按球员功能定位进行差异化定价”的论点。
