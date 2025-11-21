import os
import pandas as pd
from datetime import datetime, time, timedelta
from typing import List, Tuple, Dict, Optional
import time as pytime
import argparse

# Inputs/Outputs (defaults; may be overridden when --month is set)
EXTRACT_CSV = "/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_extract.csv"
OUTPUT_XLSX = "/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_with_opt.xlsx"
PROGRESS_CSV = "/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_progress.csv"

# Power and time parameters
SLOT_MINUTES = 5
CHARGE_POWER_KW = 670.0
DISCHARGE_POWER_KW = 2400.0
REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT = SLOT_MINUTES * CHARGE_POWER_KW / DISCHARGE_POWER_KW  # 5*670/2400
ENERGY_PER_CHARGE_SLOT_FULL_KWH = CHARGE_POWER_KW * SLOT_MINUTES / 60.0  # 670*5/60
ENERGY_PER_DISCHARGE_MIN_KWH = DISCHARGE_POWER_KW / 60.0  # 40 kWh per min


def assign_cycle_date(ts: pd.Timestamp) -> pd.Timestamp:
    # 修正的周期归属逻辑：
    # 一个周期从某天23:00开始到次日08:00结束
    # 23:00-23:59 和次日 00:00-07:55 都归属到开始那天（23:00所在的天）
    if ts.time() >= time(23, 0):  # 23:00-23:59，属于当天开始的周期
        return ts.normalize()
    elif ts.time() < time(8, 0):  # 00:00-07:55，属于前一天开始的周期
        return (ts - pd.Timedelta(days=1)).normalize()
    else:
        # 08:00-22:59 不应该出现在我们的数据中（因为只提取了23:00-08:00）
        # 但为了安全起见，归属到当天
        return ts.normalize()


def solve_opt_for_cycle(times: List[pd.Timestamp], rrps: List[float], labels: List[str], fixed_z: Optional[float] = None) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """给定一个周期的 5 分钟行，求最优 z 与分钟分配 x_{i,j}。
    返回：z_used, x_dict，其中 x_dict[(i,j)]=分钟。
    i 索引充电行，j 索引放电行（均为周期内相对索引）。

    若 fixed_z 不为 None，则仅用该 z 解一次 LP（极速模式）。
    """
    try:
        import pulp  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 PuLP：在虚拟环境中 pip install pulp")

    # 构建充/放电索引映射
    charge_idx = [k for k, lab in enumerate(labels) if lab == "charge"]
    discharge_idx = [k for k, lab in enumerate(labels) if lab == "discharge"]

    if not charge_idx or not discharge_idx:
        return fixed_z or 0.0, {}

    # 价格向量
    pc = {i: rrps[i] for i in charge_idx}
    pdv = {j: rrps[j] for j in discharge_idx}

    def solve_for_z(z_val: float) -> Tuple[float, Dict[Tuple[int, int], float]]:
        # 构建 LP
        prob = pulp.LpProblem("max_profit", pulp.LpMaximize)
        x_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for i in charge_idx:
            for j in discharge_idx:
                x_vars[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, cat=pulp.LpContinuous)
        # 单位分钟收益：若 pd-pc>z 则 (pd-pc)*40，否则 0
        unit_profit = {(i, j): ((pdv[j] - pc[i]) * ENERGY_PER_DISCHARGE_MIN_KWH if (pdv[j] - pc[i]) > z_val else 0.0)
                        for i in charge_idx for j in discharge_idx}
        prob += pulp.lpSum(unit_profit[(i, j)] * x_vars[(i, j)] for i in charge_idx for j in discharge_idx)
        # 行约束
        for i in charge_idx:
            prob += pulp.lpSum(x_vars[(i, j)] for j in discharge_idx) <= REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT
        # 列约束
        for j in discharge_idx:
            prob += pulp.lpSum(x_vars[(i, j)] for i in charge_idx) <= SLOT_MINUTES
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        x_val: Dict[Tuple[int, int], float] = {}
        for i in charge_idx:
            for j in discharge_idx:
                val = float(x_vars[(i, j)].value() or 0.0)
                if val > 1e-9:
                    x_val[(i, j)] = val
        return z_val, x_val

    # 极速：只解一次 z=fixed_z（默认 0.0）
    if fixed_z is not None:
        return solve_for_z(fixed_z)

    # 常规：枚举候选 z
    # 候选 z 集合：取 diff=pd−pc 的正值集合，并加入 0
    diffs = []
    for i in charge_idx:
        for j in discharge_idx:
            d = pdv[j] - pc[i]
            if d > 0:
                diffs.append(d)
    if not diffs:
        return 0.0, {}
    cand_z = sorted(set([0.0] + [max(0.0, d - 1e-9) for d in diffs]))

    best_profit = -1.0
    best_z = 0.0
    best_x: Dict[Tuple[int, int], float] = {}

    for z in cand_z:
        # 解 z
        z_used, x_val = solve_for_z(z)
        # 计算收益（基于 x_val）
        profit = 0.0
        for (i, j), val in x_val.items():
            profit += (pdv[j] - pc[i]) * ENERGY_PER_DISCHARGE_MIN_KWH * val
        if profit > best_profit + 1e-6:
            best_profit = profit
            best_z = z_used
            best_x = x_val

    return best_z, best_x


def build_excel(month: Optional[str] = None, fast_z0: bool = False) -> None:
    df = pd.read_csv(EXTRACT_CSV)
    df["time"] = pd.to_datetime(df["time"])

    # 分配周期日期
    df["cycle_date"] = df["time"].apply(assign_cycle_date)

    # 如果指定 month（YYYY-MM），则仅处理该月的周期
    out_xlsx = OUTPUT_XLSX
    progress_csv = PROGRESS_CSV
    if month:
        # 过滤周期所在月份
        period = pd.Period(month, freq="M")
        df = df[df["cycle_date"].dt.to_period("M") == period]
        # 更新输出文件名
        out_xlsx = f"/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_with_opt_{month}.xlsx"
        progress_csv = f"/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_progress_{month}.csv"
        if fast_z0:
            out_xlsx = f"/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_with_opt_{month}_z0Fast.xlsx"
            progress_csv = f"/Users/guxiuchen/Desktop/数据处理/AEMO_23to08_progress_{month}_z0Fast.csv"

    total_cycles = int(df["cycle_date"].nunique())
    if total_cycles == 0:
        print("未找到该月份的周期数据。")
        return

    done = 0
    start_ts = pytime.time()

    out_rows = []

    for cycle_date, g in df.groupby("cycle_date", sort=True):
        g_sorted = g.sort_values("time").reset_index(drop=True)
        times = g_sorted["time"].tolist()
        rrps = g_sorted["rrp"].astype(float).tolist()
        labels = g_sorted["label"].tolist()

        if fast_z0:
            z_used, x_dict = solve_opt_for_cycle(times, rrps, labels, fixed_z=0.0)
        else:
            z_used, x_dict = solve_opt_for_cycle(times, rrps, labels)

        # 汇总到行级：计算每个充电行/放电行的分钟总量
        charge_minutes: Dict[int, float] = {}
        discharge_minutes: Dict[int, float] = {}
        for (i, j), v in x_dict.items():
            charge_minutes[i] = charge_minutes.get(i, 0.0) + v
            discharge_minutes[j] = discharge_minutes.get(j, 0.0) + v

        # 行级能量（第5列）：充电为正，放电为负
        energies: List[float] = []
        pnls: List[float] = []
        for idx, row in g_sorted.iterrows():
            lab = row["label"]
            price = float(row["rrp"])
            if lab == "charge":
                minutes = charge_minutes.get(idx, 0.0)
                energy = (minutes / REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT) * ENERGY_PER_CHARGE_SLOT_FULL_KWH
                pnl = - price * energy
            else:
                minutes = discharge_minutes.get(idx, 0.0)
                energy = - minutes * ENERGY_PER_DISCHARGE_MIN_KWH
                pnl = - price * energy  # energy为负，-price*energy为正（收益）
            energies.append(energy)
            pnls.append(pnl)

        # 累积电量（第6列）：避免 <0
        cum = 0.0
        cum_list: List[float] = []
        for e in energies:
            cum = max(0.0, cum + e)
            cum_list.append(cum)

        # 周期总收益（第8列）
        cycle_total = sum(pnls)

        for k in range(len(g_sorted)):
            out_rows.append({
                "time": g_sorted.loc[k, "time"],
                "rrp": g_sorted.loc[k, "rrp"],
                "label": g_sorted.loc[k, "label"],
                "z": z_used,
                "qty_kwh": energies[k],
                "cum_kwh": cum_list[k],
                "pnl": pnls[k],
                "cycle_total_pnl": cycle_total
            })

        # 进度与中间结果
        done += 1
        elapsed = pytime.time() - start_ts
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total_cycles - done) / rate if rate > 0 else float("inf")
        mode_tag = "z0Fast" if fast_z0 else "optZscan"
        print(f"[Progress {mode_tag}] {done}/{total_cycles} cycles | z={z_used:.4f} | cycle_pnl={cycle_total:.2f} | elapsed={elapsed/60:.1f}m | eta={remaining/60:.1f}m", flush=True)

        try:
            prog_df = pd.DataFrame(out_rows)
            prog_df = prog_df.rename(columns={
                "time": "时间",
                "rrp": "电价(RRP)",
                "label": "阶段",
                "z": "z值",
                "qty_kwh": "电量(kWh)",
                "cum_kwh": "累计电量(kWh)",
                "pnl": "成本/收益",
                "cycle_total_pnl": "周期总收益",
            })
            prog_df.to_csv(progress_csv, index=False)
        except Exception:
            pass

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.rename(columns={
        "time": "时间",
        "rrp": "电价(RRP)",
        "label": "阶段",
        "z": "z值",
        "qty_kwh": "电量(kWh)",
        "cum_kwh": "累计电量(kWh)",
        "pnl": "成本/收益",
        "cycle_total_pnl": "周期总收益",
    })
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="23to08_opt")
    print(f"已输出Excel: {out_xlsx}, 行数: {len(out_df)}")


def main():
    parser = argparse.ArgumentParser(description="Build monthly or full Excel with optimal z per cycle")
    parser.add_argument("--month", type=str, default=None, help="仅处理某个自然月（格式 YYYY-MM，例如 2024-01）")
    parser.add_argument("--fast-z0", action="store_true", help="每周期固定 z=0，只解一次LP（显著加速）")
    parser.add_argument("--batch-all-months", action="store_true", help="对数据覆盖到的所有月份批量导出")
    args = parser.parse_args()

    if args.batch_all_months:
        df0 = pd.read_csv(EXTRACT_CSV)
        df0["time"] = pd.to_datetime(df0["time"])
        df0["cycle_date"] = df0["time"].apply(assign_cycle_date)
        months = sorted(str(p) for p in df0["cycle_date"].dt.to_period("M").unique())
        print("将批量处理月份:", ", ".join(months))
        for m in months:
            build_excel(month=m, fast_z0=args.fast_z0)
        return

    build_excel(month=args.month, fast_z0=args.fast_z0)


if __name__ == "__main__":
    main() 