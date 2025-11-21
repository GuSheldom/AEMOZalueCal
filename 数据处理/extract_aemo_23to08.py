import os
import glob
import pandas as pd
from datetime import datetime, time

AEMO_DIR = "/Users/guxiuchen/Desktop/数据处理/aemo"
OUTPUT_CSV = "/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_extract.csv"

# 配置：时间区间
START_TS = pd.Timestamp("2023-12-31 23:00:00")
END_TS = pd.Timestamp("2025-11-01 08:00:00")

# 需要的列
TIME_COL = "SETTLEMENTDATE"
PRICE_COL = "RRP"


def read_last_12_of_dec_2023() -> pd.DataFrame:
    dec_path = os.path.join(AEMO_DIR, "PRICE_AND_DEMAND_202312_SA1.csv")
    df = pd.read_csv(dec_path)
    # 只取最后13行
    return df.tail(13)


def read_month_files_after_dec_2023() -> pd.DataFrame:
    # 取 2024-01 到 2025-07（目录中已有的全部文件）
    pattern = os.path.join(AEMO_DIR, "PRICE_AND_DEMAND_*.csv")
    files = sorted(glob.glob(pattern))
    # 过滤掉 202312，自然保留其后续月份
    files = [p for p in files if not p.endswith("PRICE_AND_DEMAND_202312_SA1.csv")]
    dfs = []
    for p in files:
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def parse_datetime(col: pd.Series) -> pd.Series:
    # AEMO 格式示例：YYYY/MM/DD HH:MM:SS
    return pd.to_datetime(col, format="%Y/%m/%d %H:%M:%S", errors="coerce")


def label_window_and_mode(ts: pd.Timestamp) -> tuple[bool, str | None]:
    # 仅保留 每天23:00-次日08:00
    # charge: [23:00, 05:00)
    # discharge: [05:00, 08:00)
    tod = ts.time()
    in_window = (tod >= time(23, 0)) or (tod < time(8, 0))
    if not in_window:
        return False, None
    if (tod >= time(23, 0)) or (tod < time(5, 0)):
        return True, "charge"
    else:
        return True, "discharge"


def main() -> None:
    parts = []
    # 从202312最后12行开始
    parts.append(read_last_12_of_dec_2023())
    # 合并后续所有月份
    rest = read_month_files_after_dec_2023()
    if not rest.empty:
        parts.append(rest)
    merged = pd.concat(parts, ignore_index=True)

    # 解析时间
    merged[TIME_COL] = parse_datetime(merged[TIME_COL])
    merged = merged.dropna(subset=[TIME_COL])

    # 限定时间区间
    merged = merged[(merged[TIME_COL] >= START_TS) & (merged[TIME_COL] <= END_TS)]

    # 打上标签，并只保留窗口内
    keep_mask = []
    labels = []
    for ts in merged[TIME_COL]:
        keep, lab = label_window_and_mode(ts)
        keep_mask.append(keep)
        labels.append(lab)
    labels_kept = [lab for lab, k in zip(labels, keep_mask) if k]
    merged = merged[keep_mask].copy()
    merged["MODE"] = labels_kept

    # 只保留所需列
    out = merged[[TIME_COL, PRICE_COL, "MODE"]].copy()
    out = out.rename(columns={TIME_COL: "time", PRICE_COL: "rrp", "MODE": "label"})

    # 排序并输出
    out = out.sort_values("time").reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"已输出: {OUTPUT_CSV}, 行数: {len(out)}")


if __name__ == "__main__":
    main() 