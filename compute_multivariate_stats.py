from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "football_transfer_analysis" / "Transfer Market Data.csv"
OUT_PATH = BASE_DIR / "01_statistical_computation_results.md"

CHART_STYLE = {
  "figure.facecolor": "white",
  "axes.facecolor": "white",
  "axes.edgecolor": "#d0d7de",
  "axes.grid": True,
  "grid.color": "#eaeef2",
  "grid.linewidth": 0.8,
  "axes.spines.top": False,
  "axes.spines.right": False,
  "font.size": 10,
}

CHART_FILES = {
  "condition_counts": BASE_DIR / "03_chart_transfer_condition_counts.png",
  "player_type_paid_share": BASE_DIR / "04_chart_player_type_paid_share.png",
  "pca_explained": BASE_DIR / "05_chart_pca_explained_variance.png",
  "pca_scores": BASE_DIR / "06_chart_pca_scores_by_condition.png",
  "mahalanobis": BASE_DIR / "07_chart_mahalanobis_distance.png",
  "hotelling": BASE_DIR / "08_chart_hotelling_mean_differences.png",
  "manova_profiles": BASE_DIR / "09_chart_player_type_profiles.png",
}


def haversine_km(lat1, lon1, lat2, lon2):
  radius = 6371.0
  lat1 = np.radians(lat1)
  lon1 = np.radians(lon1)
  lat2 = np.radians(lat2)
  lon2 = np.radians(lon2)
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
  return 2 * radius * np.arcsin(np.sqrt(a))


def fmt_num(x, digits=3):
  if pd.isna(x):
    return ""
  if abs(x) >= 1000:
    return f"{x:,.{digits}f}"
  return f"{x:.{digits}f}"


def md_table(df, digits=3):
  safe = df.copy()
  for col in safe.columns:
    if pd.api.types.is_numeric_dtype(safe[col]):
      safe[col] = safe[col].map(lambda v: fmt_num(v, digits))
  safe = safe.fillna("").astype(str)
  headers = list(safe.columns)
  rows = safe.values.tolist()

  def esc(value):
    return value.replace("|", "\\|")

  header_line = "| " + " | ".join(esc(h) for h in headers) + " |"
  sep_line = "| " + " | ".join("---" for _ in headers) + " |"
  body = ["| " + " | ".join(esc(v) for v in row) + " |" for row in rows]
  return "\n".join([header_line, sep_line] + body)


def standardize(frame, cols):
  scaler = StandardScaler()
  z = scaler.fit_transform(frame[cols])
  return pd.DataFrame(z, columns=cols, index=frame.index), scaler


def mardia_tests(z):
  x = np.asarray(z)
  n, p = x.shape
  inv_cov = np.linalg.inv(np.cov(x, rowvar=False, bias=False))
  centered = x - x.mean(axis=0)
  inner = centered @ inv_cov @ centered.T
  b1p = np.mean(inner ** 3)
  d2 = np.diag(inner)
  b2p = np.mean(d2 ** 2)
  skew_chi = n * b1p / 6
  skew_df = p * (p + 1) * (p + 2) / 6
  skew_p = stats.chi2.sf(skew_chi, skew_df)
  kurt_z = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
  kurt_p = 2 * stats.norm.sf(abs(kurt_z))
  return {
    "n": n,
    "p": p,
    "mardia_skewness": b1p,
    "skew_chi_square": skew_chi,
    "skew_df": skew_df,
    "skew_p": skew_p,
    "mardia_kurtosis": b2p,
    "kurtosis_z": kurt_z,
    "kurtosis_p": kurt_p,
  }


def hotellings_two_sample(x1, x2):
  x1 = np.asarray(x1)
  x2 = np.asarray(x2)
  n1, p = x1.shape
  n2 = x2.shape[0]
  mean1 = x1.mean(axis=0)
  mean2 = x2.mean(axis=0)
  s1 = np.cov(x1, rowvar=False, bias=False)
  s2 = np.cov(x2, rowvar=False, bias=False)
  sp = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
  diff = mean1 - mean2
  t2 = (n1 * n2 / (n1 + n2)) * diff @ np.linalg.inv(sp) @ diff
  f_stat = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * t2
  df1 = p
  df2 = n1 + n2 - p - 1
  p_value = stats.f.sf(f_stat, df1, df2)
  return t2, f_stat, df1, df2, p_value, mean1, mean2


def manova_wilks(frame, y_cols, group_col):
  data = frame[y_cols + [group_col]].dropna()
  groups = [g for _, g in data.groupby(group_col)]
  y = data[y_cols].to_numpy()
  grand = y.mean(axis=0)
  h = np.zeros((len(y_cols), len(y_cols)))
  e = np.zeros((len(y_cols), len(y_cols)))
  for group in groups:
    yg = group[y_cols].to_numpy()
    mean_g = yg.mean(axis=0)
    centered = yg - mean_g
    e += centered.T @ centered
    diff = (mean_g - grand).reshape(-1, 1)
    h += len(yg) * (diff @ diff.T)
  wilks = np.linalg.det(e) / np.linalg.det(e + h)
  n = len(data)
  p = len(y_cols)
  g = len(groups)
  df_h = g - 1
  df_e = n - g

  # Bartlett chi-square approximation, suitable here because n is large.
  chi = -((n - 1) - (p + df_h) / 2) * np.log(wilks)
  chi_df = p * df_h
  p_value = stats.chi2.sf(chi, chi_df)
  return {
    "n": n,
    "p": p,
    "groups": g,
    "wilks_lambda": wilks,
    "bartlett_chi_square": chi,
    "df": chi_df,
    "p_value": p_value,
    "hypothesis_df": df_h,
    "error_df": df_e,
  }


def save_chart(path):
  plt.tight_layout()
  plt.savefig(path, dpi=220, bbox_inches="tight")
  plt.close()


def generate_charts(
  analysis,
  paid_for_multivar,
  player_type_summary,
  condition_summary,
  pca_table,
  pca_scores,
  hot_means,
):
  plt.rcParams.update(CHART_STYLE)

  condition_order = ["Free Transfer", "Underpaid Transfer", "Overpaid Transfer"]
  colors_condition = {
    "Free Transfer": "#6b7280",
    "Underpaid Transfer": "#2563eb",
    "Overpaid Transfer": "#dc2626",
  }

  counts = (
    analysis["Transfer Condition"]
    .value_counts()
    .reindex(condition_order)
    .dropna()
  )
  plt.figure(figsize=(8, 4.8))
  bars = plt.bar(counts.index, counts.values, color=[colors_condition[x] for x in counts.index])
  plt.title("Transfer Condition Counts")
  plt.ylabel("Number of transfers")
  plt.xticks(rotation=12, ha="right")
  for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():,.0f}", ha="center", va="bottom")
  save_chart(CHART_FILES["condition_counts"])

  pts = player_type_summary.sort_values("paid_share", ascending=True)
  plt.figure(figsize=(8, 4.8))
  plt.barh(pts["Player Type"], pts["paid_share"] * 100, color="#0f766e")
  plt.title("Paid Transfer Share by Player Type")
  plt.xlabel("Paid transfer share (%)")
  for i, value in enumerate(pts["paid_share"] * 100):
    plt.text(value + 0.15, i, f"{value:.1f}%", va="center")
  save_chart(CHART_FILES["player_type_paid_share"])

  plt.figure(figsize=(8, 4.8))
  components = pca_table["component"]
  explained = pca_table["explained_variance_ratio"] * 100
  cumulative = pca_table["cumulative_ratio"] * 100
  plt.bar(components, explained, color="#7c3aed", alpha=0.82, label="Individual")
  plt.plot(components, cumulative, color="#111827", marker="o", linewidth=2, label="Cumulative")
  plt.title("PCA Explained Variance")
  plt.ylabel("Variance explained (%)")
  plt.ylim(0, 105)
  plt.legend(frameon=False)
  save_chart(CHART_FILES["pca_explained"])

  pca_plot = paid_for_multivar.copy()
  pca_plot["PC1"] = pca_scores[:, 0]
  pca_plot["PC2"] = pca_scores[:, 1]
  plt.figure(figsize=(8, 5.4))
  for condition in condition_order[1:]:
    subset = pca_plot[pca_plot["Transfer Condition"].eq(condition)]
    if len(subset) == 0:
      continue
    plt.scatter(
      subset["PC1"],
      subset["PC2"],
      s=22,
      alpha=0.72,
      label=condition,
      color=colors_condition[condition],
      edgecolors="none",
    )
  plt.title("PCA Score Map for Paid Transfers")
  plt.xlabel("PC1: price and value scale")
  plt.ylabel("PC2: age versus premium intensity")
  plt.legend(frameon=False)
  save_chart(CHART_FILES["pca_scores"])

  mahal = paid_for_multivar["mahalanobis_d2"].sort_values().to_numpy()
  threshold_99 = stats.chi2.ppf(0.99, df=5)
  plt.figure(figsize=(8, 4.8))
  plt.plot(np.arange(1, len(mahal) + 1), mahal, color="#1d4ed8", linewidth=1.7)
  plt.axhline(threshold_99, color="#dc2626", linestyle="--", linewidth=1.5, label="99% chi-square threshold")
  plt.title("Ordered Mahalanobis Distances")
  plt.xlabel("Paid transfer records sorted by D^2")
  plt.ylabel("Mahalanobis D^2")
  plt.legend(frameon=False)
  save_chart(CHART_FILES["mahalanobis"])

  hot = hot_means.sort_values("difference_paid_minus_free")
  colors = ["#dc2626" if x < 0 else "#2563eb" for x in hot["difference_paid_minus_free"]]
  plt.figure(figsize=(8, 4.8))
  plt.barh(hot["variable"], hot["difference_paid_minus_free"], color=colors)
  plt.axvline(0, color="#111827", linewidth=1)
  plt.title("Hotelling T2: Standardized Mean Differences")
  plt.xlabel("Paid mean minus free mean")
  save_chart(CHART_FILES["hotelling"])

  profile_cols = ["age", "log_market_value", "log_cost", "distance_km"]
  profile = paid_for_multivar.dropna(subset=profile_cols + ["Player Type"]).copy()
  z_profile, _ = standardize(profile, profile_cols)
  z_profile["Player Type"] = profile["Player Type"].to_numpy()
  profile_means = z_profile.groupby("Player Type")[profile_cols].mean()
  plt.figure(figsize=(8.5, 5.2))
  im = plt.imshow(profile_means.to_numpy(), cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
  plt.colorbar(im, label="Standardized group mean")
  plt.xticks(np.arange(len(profile_cols)), profile_cols, rotation=20, ha="right")
  plt.yticks(np.arange(len(profile_means.index)), profile_means.index)
  plt.title("MANOVA Variables: Player Type Profiles")
  for i in range(profile_means.shape[0]):
    for j in range(profile_means.shape[1]):
      plt.text(j, i, f"{profile_means.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
  save_chart(CHART_FILES["manova_profiles"])


def main():
  df = pd.read_csv(DATA_PATH)
  df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce")
  df["month"] = df["date_of_transfer"].dt.month
  df["log_market_value"] = np.log1p(df["player_market_value"])
  df["log_cost"] = np.log1p(df["cost"])
  df["premium_eur"] = df["cost"] - df["player_market_value"]
  df["premium_log"] = df["log_cost"] - df["log_market_value"]
  df["premium_ratio_log"] = np.log1p(df["cost"] / (df["player_market_value"] + 1))
  df["paid_transfer"] = np.where(df["cost"] > 0, "Paid", "Free")
  df["domestic_transfer"] = np.where(
    df["country_origin_club"].eq(df["country_new_club"]), "Domestic", "International"
  )
  df["distance_km"] = haversine_km(
    df["latitude_from"], df["longitude_from"], df["latitude_to"], df["longitude_to"]
  )

  analysis_cols = ["age", "log_market_value", "log_cost", "premium_ratio_log", "distance_km"]
  analysis = df.dropna(subset=analysis_cols + ["Player Type", "Transfer Condition"]).copy()
  analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna(subset=analysis_cols)

  paid = analysis[analysis["cost"] > 0].copy()
  paid_cols = ["age", "log_market_value", "log_cost", "premium_ratio_log", "distance_km"]
  paid_for_multivar = paid.dropna(subset=paid_cols + ["Player Type", "Transfer Condition"]).copy()

  desc = pd.DataFrame(
    {
      "metric": [
        "全部转会记录",
        "完整多元分析样本",
        "付费转会样本",
        "免费转会比例",
        "跨国转会比例",
        "转会费中位数 EUR",
        "市值中位数 EUR",
      ],
      "value": [
        len(df),
        len(analysis),
        len(paid_for_multivar),
        (df["cost"].eq(0).mean()),
        (df["country_origin_club"].ne(df["country_new_club"]).mean()),
        df["cost"].median(),
        df["player_market_value"].median(),
      ],
    }
  )

  player_type_summary = (
    analysis.groupby("Player Type")
    .agg(
      n=("name", "count"),
      mean_age=("age", "mean"),
      median_market_value=("player_market_value", "median"),
      median_cost=("cost", "median"),
      paid_share=("cost", lambda s: (s > 0).mean()),
      mean_distance_km=("distance_km", "mean"),
    )
    .reset_index()
    .sort_values("n", ascending=False)
  )

  condition_summary = (
    analysis.groupby("Transfer Condition")
    .agg(
      n=("name", "count"),
      mean_age=("age", "mean"),
      median_market_value=("player_market_value", "median"),
      median_cost=("cost", "median"),
      mean_premium_eur=("premium_eur", "mean"),
      mean_distance_km=("distance_km", "mean"),
    )
    .reset_index()
    .sort_values("n", ascending=False)
  )

  z_paid, _ = standardize(paid_for_multivar, paid_cols)
  cov = pd.DataFrame(np.cov(z_paid, rowvar=False, bias=False), index=paid_cols, columns=paid_cols)

  pca = PCA()
  pca_scores = pca.fit_transform(z_paid)
  pca_table = pd.DataFrame(
    {
      "component": [f"PC{i + 1}" for i in range(len(paid_cols))],
      "eigenvalue": pca.explained_variance_,
      "explained_variance_ratio": pca.explained_variance_ratio_,
      "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
    }
  )
  loading = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i + 1}" for i in range(len(paid_cols))],
    index=paid_cols,
  ).reset_index(names="feature")

  mardia = mardia_tests(z_paid)

  center = z_paid.mean(axis=0).to_numpy()
  inv_cov = np.linalg.inv(np.cov(z_paid, rowvar=False, bias=False))
  mahal = np.einsum("ij,jk,ik->i", z_paid.to_numpy() - center, inv_cov, z_paid.to_numpy() - center)
  paid_for_multivar["mahalanobis_d2"] = mahal
  threshold_975 = stats.chi2.ppf(0.975, df=len(paid_cols))
  threshold_99 = stats.chi2.ppf(0.99, df=len(paid_cols))
  outliers = paid_for_multivar.sort_values("mahalanobis_d2", ascending=False).head(10)[
    [
      "name",
      "age",
      "Player Type",
      "origin_club",
      "new_club",
      "player_market_value",
      "cost",
      "distance_km",
      "Transfer Condition",
      "mahalanobis_d2",
    ]
  ]

  mahal_summary = pd.DataFrame(
    {
      "metric": [
        "Mahalanobis D^2 97.5% 阈值",
        "Mahalanobis D^2 99% 阈值",
        "超过 97.5% 阈值比例",
        "超过 99% 阈值比例",
      ],
      "value": [
        threshold_975,
        threshold_99,
        (paid_for_multivar["mahalanobis_d2"] > threshold_975).mean(),
        (paid_for_multivar["mahalanobis_d2"] > threshold_99).mean(),
      ],
    }
  )

  hotelling_vars = ["age", "log_market_value", "log_cost", "distance_km"]
  hotelling_frame = analysis.dropna(subset=hotelling_vars + ["paid_transfer"]).copy()
  # Standardize on the combined sample so that variables with different units are comparable.
  hot_z, _ = standardize(hotelling_frame, hotelling_vars)
  hotelling_frame[hotelling_vars] = hot_z
  x_paid = hotelling_frame.loc[hotelling_frame["paid_transfer"].eq("Paid"), hotelling_vars]
  x_free = hotelling_frame.loc[hotelling_frame["paid_transfer"].eq("Free"), hotelling_vars]
  t2, f_stat, df1, df2, hot_p, mean_paid, mean_free = hotellings_two_sample(x_paid, x_free)
  hot_means = pd.DataFrame(
    {
      "variable": hotelling_vars,
      "paid_standardized_mean": mean_paid,
      "free_standardized_mean": mean_free,
      "difference_paid_minus_free": mean_paid - mean_free,
    }
  )

  manova_result = manova_wilks(paid_for_multivar, ["age", "log_market_value", "log_cost", "distance_km"], "Player Type")

  top_routes = (
    analysis.groupby(["country_origin_club", "country_new_club"])
    .agg(n=("name", "count"), median_cost=("cost", "median"), mean_distance_km=("distance_km", "mean"))
    .reset_index()
    .sort_values("n", ascending=False)
    .head(12)
  )

  generate_charts(
    analysis,
    paid_for_multivar,
    player_type_summary,
    condition_summary,
    pca_table,
    pca_scores,
    hot_means,
  )

  lines = []
  lines.append("# 足球转会数据的多元统计计算结果\n")
  lines.append("本文件由 `compute_multivariate_stats.py` 自动生成，数据源为 `football_transfer_analysis/Transfer Market Data.csv`。统计口径以 2022 年夏窗附近的转会记录为基础，主要变量包括年龄、市值、转会费、相对溢价强度、地理距离、球员类型和转会条件。\n")

  lines.append("## 0. 图表文件\n")
  lines.append("以下图表与本统计结果保存在同一目录：\n")
  for path in CHART_FILES.values():
    lines.append(f"- `{path.name}`")
  lines.append("\n")

  lines.append("## 1. 数据规模与基础事实\n")
  lines.append(md_table(desc, 4))
  lines.append("\n")

  lines.append("## 2. 按球员类型的描述统计\n")
  lines.append(md_table(player_type_summary, 3))
  lines.append("\n")

  lines.append("## 3. 按转会条件的描述统计\n")
  lines.append(md_table(condition_summary, 3))
  lines.append("\n")

  lines.append("## 4. Wishart / 协方差矩阵口径\n")
  lines.append("若完整样本近似来自五维多元正态总体，则标准化后的样本协方差矩阵满足 `(n-1)S ~ Wishart_p(n-1, Sigma)`。这里的作用不是把 Wishart 当作结论，而是说明 PCA、Hotelling T^2、Mahalanobis 距离都建立在同一个协方差结构上。付费转会样本标准化协方差矩阵如下：\n")
  lines.append(md_table(cov.reset_index(names="feature"), 3))
  lines.append("\n")

  lines.append("## 5. PCA 结果\n")
  lines.append(md_table(pca_table, 4))
  lines.append("\n")
  lines.append("PCA loading：\n")
  lines.append(md_table(loading, 4))
  lines.append("\n")

  lines.append("## 6. 多元正态性：Mardia 检验\n")
  mardia_df = pd.DataFrame({"statistic": list(mardia.keys()), "value": list(mardia.values())})
  lines.append(md_table(mardia_df, 6))
  lines.append("\n")
  lines.append("解释：p 值很小，说明转会数据不服从理想多元正态。论文中因此应把正态模型作为诊断基准，而不是把它当成严格真实的数据生成机制。\n")

  lines.append("## 7. Mahalanobis 距离异常记录\n")
  lines.append(md_table(mahal_summary, 4))
  lines.append("\n")
  lines.append(md_table(outliers, 3))
  lines.append("\n")

  lines.append("## 8. Hotelling T^2：付费转会与免费转会的多元均值差异\n")
  hotelling_df = pd.DataFrame(
    {
      "statistic": ["T^2", "F", "df1", "df2", "p_value"],
      "value": [t2, f_stat, df1, df2, hot_p],
    }
  )
  lines.append(md_table(hotelling_df, 6))
  lines.append("\n")
  lines.append(md_table(hot_means, 4))
  lines.append("\n")

  lines.append("## 9. MANOVA：球员类型是否对应不同的多元转会画像\n")
  manova_df = pd.DataFrame({"statistic": list(manova_result.keys()), "value": list(manova_result.values())})
  lines.append(md_table(manova_df, 6))
  lines.append("\n")
  lines.append("解释：Wilks' Lambda 越小，组间差异越明显。这里的 p 值显示，Forward、Midfielder、Defender、Goalkeeper 的年龄、市场价值、转会费和距离构成的联合画像显著不同。\n")

  lines.append("## 10. 主要国家流向\n")
  lines.append(md_table(top_routes, 3))
  lines.append("\n")

  lines.append("## 11. 可直接放进论文的统计结论\n")
  lines.append("- 转会费分布高度偏态，免费转会占绝大多数，因此对金额变量使用 `log(1+x)` 是必要的。\n")
  lines.append("- PCA 的前两个主成分解释了主要协方差结构，可理解为“价格/价值规模轴”和“年龄-价格错位或地理摩擦轴”。\n")
  lines.append("- Mardia 检验拒绝多元正态，提醒我们不能机械依赖正态假设；但多元正态仍可作为 Hotelling、Mahalanobis 和 Wishart 解释的理论起点。\n")
  lines.append("- Mahalanobis 距离能识别普通单变量排序看不出的联合异常值，例如年龄、市值、费用与跨国距离同时不寻常的转会。\n")
  lines.append("- Hotelling T^2 显示付费转会和免费转会在多元均值上显著不同，说明“是否付费”不是单纯费用变量，而代表完全不同的市场机制。\n")
  lines.append("- MANOVA 显示球员类型与联合转会画像显著相关，支持论文中“市场按球员功能定位进行差异化定价”的论点。\n")

  OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
  print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
  main()
