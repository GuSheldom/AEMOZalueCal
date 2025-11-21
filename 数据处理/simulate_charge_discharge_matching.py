"""
电价套利匹配仿真（充电/放电价格匹配 + z 安全边际 + 功率不对等处理）

核心思想：
- 充电时段（23:00–05:00）与放电时段（05:00–08:00）的每5分钟价格序列作为输入；
- z 仅作为“门槛”用于可行性判定：只有当 放电价 > 充电价 + z 才允许这对进行分配；
- 功率不对等：充电功率 670kW，放电功率 2400kW。每个“充电5分钟段”的能量，按放电功率仅需 5*670/2400 ≈ 1.3958 分钟即可释放；
- 收益计算不包含 z：按每个分配片段的（放电价 − 充电价）× 能量累加（负值按0计），z 只影响是否允许匹配，不进入收益公式；
- 分配策略：
  * greedy（默认）：可跨多个放电段逐段拼补，直到满足所需放电分钟或候选耗尽；
  * opt：使用线性规划在所有充/放电组合上同时优化，得到全局最优的分钟分配与收益（需要 PuLP）。

输出包含总收益、每个充电段的分配细节（拆分片段）、以及各放电段剩余可用分钟数。
"""
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# -----------------------------
# Domain model and parameters
# -----------------------------

SLOT_MINUTES: int = 5  # 时间槽宽度（分钟）
CHARGE_POWER_KW: float = 670.0  # 充电功率
DISCHARGE_POWER_KW: float = 2400.0  # 放电功率

# 每个5分钟充电段对应需要的放电时间（分钟），保证能量守恒：
# 充电能量 = 670kW * (5/60)h，需要用 2400kW 放电，则所需分钟 = 5*670/2400
REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT: float = (
    SLOT_MINUTES * CHARGE_POWER_KW / DISCHARGE_POWER_KW
)

# 每个放电时间段的最大可用放电时间（分钟）（即单个5分钟放电槽最多可用5分钟）
DISCHARGE_SLOT_CAPACITY_MIN: float = float(SLOT_MINUTES)

# 每分钟放电对应能量（kWh）：2400 kW * (1/60) h
ENERGY_KWH_PER_DISCHARGE_MIN: float = DISCHARGE_POWER_KW / 60.0


@dataclass
class TimeSlot:
    """统一描述一个5分钟时间槽及其价格（用于充电或放电序列）。

    属性：
    - index：该槽在序列中的索引（0-based）；
    - price：该时间槽对应电价（单位同输入价格）。
    """
    index: int
    price: float


@dataclass
class AllocationPiece:
    """单个“分配片段”，记录某充电段分配到某个放电段的分钟数与该片段的单位能量价差。

    字段：
    - discharge_index：放电时间槽索引（指向具体哪个放电5分钟段）；
    - allocated_minutes：从该放电槽实际分配到的“放电分钟”（可能是小数，如 0.3 分钟）；
    - spread：本片段的单位能量价差（毛利），定义为：
      spread = max(放电价 − 充电价, 0)。
      注意：z 仅用于“是否允许匹配”的门槛判定，不进入收益；负 spread 视为 0（不计亏损）。
    """
    discharge_index: int
    allocated_minutes: float
    # 对应分配部分的单位能量价差（不含 z）：放电价 - 充电价
    spread: float


@dataclass
class ChargeAllocation:
    """描述某个“充电5分钟段”的整体分配与收益。

    字段：
    - charge_index：充电时间槽索引；
    - charge_price：该充电槽的电价；
    - required_minutes：为释放该充电槽能量所需的放电分钟（= 5*670/2400）；
    - pieces：分配到的若干“放电片段”（可跨多个放电槽，取决于分配模式）；
    - unmet_minutes：未匹配到的放电分钟（候选不足时为正）。

    性质：
    - allocated_minutes：已匹配到的放电分钟之和；
    - profit：将每个片段的 spread 乘以对应能量并累加（spread≤0 已在片段处截断为0）。
    """
    charge_index: int
    charge_price: float
    required_minutes: float
    pieces: List[AllocationPiece] = field(default_factory=list)
    unmet_minutes: float = 0.0

    @property
    def allocated_minutes(self) -> float:
        return sum(p.allocated_minutes for p in self.pieces)

    @property
    def profit(self) -> float:
        # 收益按能量比例计算。
        # 基准能量：每个充电槽的总能量 = CHARGE_POWER_KW * (SLOT_MINUTES/60) kWh。
        # 片段能量比例：allocated_minutes / REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT。
        # 每片段收益：spread * 片段能量；负 spread 已在片段创建时截断为0。
        # 单位换算：RRP是AUD/MWh，能量是kWh，需要除以1000转换为MWh
        energy_kwh_per_full: float = CHARGE_POWER_KW * SLOT_MINUTES / 60.0
        if REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT <= 0:
            return 0.0
        total = 0.0
        for p in self.pieces:
            energy_ratio = p.allocated_minutes / REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT
            energy_kwh = energy_kwh_per_full * energy_ratio
            energy_mwh = energy_kwh / 1000.0  # 转换为MWh
            total += p.spread * energy_mwh  # AUD/MWh × MWh = AUD
        # 防御性：若总和出现负值（理论上不会），按0计。
        return max(total, 0.0)


def generate_random_prices(num_slots: int, seed: Optional[int] = None,
                           mean: float = 0.0, stddev: float = 50.0,
                           allow_negatives: bool = True) -> List[float]:
    """生成高斯分布的随机价格序列，用于演示。

    参数：
    - num_slots：生成的5分钟槽数量；
    - seed：随机种子（可复现）；
    - mean/stddev：均值/标准差；
    - allow_negatives：是否允许负电价（充电允许，放电通常不允许）。
    """
    if seed is not None:
        random.seed(seed)
    prices: List[float] = []
    for _ in range(num_slots):
        val = random.gauss(mean, stddev)
        if not allow_negatives:
            val = max(val, 0.0)
        prices.append(val)
    return prices


def build_timeslots(prices: List[float]) -> List[TimeSlot]:
    """将价格序列包装为 `TimeSlot` 列表（附带索引）。"""
    return [TimeSlot(index=i, price=p) for i, p in enumerate(prices)]


def simulate_matching_greedy(
    charge_prices: List[float],
    discharge_prices: List[float],
    z: float,
    prefer: str = "min",
) -> Tuple[float, List[ChargeAllocation], Dict[int, float]]:
    """贪心分配（multi）：逐候选依次扣减，直到满足或耗尽。

    z 仅用于过滤候选（放电价 > 充电价 + z）。收益用（放电价 − 充电价）× 能量（负值截0）。
    """
    assert prefer in ("min", "max")

    charge_slots: List[TimeSlot] = build_timeslots(charge_prices)
    discharge_slots: List[TimeSlot] = build_timeslots(discharge_prices)

    discharge_remaining: Dict[int, float] = {s.index: DISCHARGE_SLOT_CAPACITY_MIN for s in discharge_slots}

    sorted_charge_slots: List[TimeSlot] = sorted(charge_slots, key=lambda s: s.price)
    allocations: List[ChargeAllocation] = []

    for ch in sorted_charge_slots:
        required = REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT
        alloc = ChargeAllocation(charge_index=ch.index, charge_price=ch.price, required_minutes=required)

        candidates: List[TimeSlot] = [d for d in discharge_slots if d.price > ch.price + z]
        reverse = (prefer == "max")
        candidates.sort(key=lambda s: s.price, reverse=reverse)

        if not candidates:
            alloc.unmet_minutes = required
            allocations.append(alloc)
            continue

        needed = required
        for d in candidates:
            if needed <= 0:
                break
            rem = discharge_remaining.get(d.index, 0.0)
            if rem <= 1e-9:
                continue
            take = min(rem, needed)
            if take <= 0:
                continue
            # 收益按（放电价 − 充电价）计算；z 只用于阈值，不进入收益
            spread = d.price - ch.price
            alloc.pieces.append(AllocationPiece(discharge_index=d.index, allocated_minutes=take, spread=max(spread, 0.0)))
            discharge_remaining[d.index] = rem - take
            needed -= take

        alloc.unmet_minutes = max(0.0, needed)
        allocations.append(alloc)

    total_profit = sum(a.profit for a in allocations)
    allocations.sort(key=lambda a: a.charge_index)
    return total_profit, allocations, discharge_remaining


def simulate_matching_opt(
    charge_prices: List[float],
    discharge_prices: List[float],
    z: float,
) -> Tuple[float, List[ChargeAllocation], Dict[int, float]]:
    """全局最优分配：使用线性规划在所有充/放电组合上同时优化。

    目标：最大化 Σ_{i,j} unit_profit_per_min(i,j) * x_{i,j}
    约束：
      - 对每个充电段 i：Σ_j x_{i,j} ≤ required_i（=1.3958）
      - 对每个放电段 j：Σ_i x_{i,j} ≤ capacity_j（=5）
      - x_{i,j} ≥ 0
    其中 unit_profit_per_min(i,j) = 1_{(p_d[j] > p_c[i] + z)} × max(p_d[j] − p_c[i], 0) × ENERGY_KWH_PER_DISCHARGE_MIN
    即：z 仅作为门槛，收益不包含 z。
    """
    try:
        import pulp  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 PuLP 才能使用 --strategy opt，全局最优求解。请先 pip install pulp")

    charge_slots: List[TimeSlot] = build_timeslots(charge_prices)
    discharge_slots: List[TimeSlot] = build_timeslots(discharge_prices)

    num_c = len(charge_slots)
    num_d = len(discharge_slots)

    # 计算每对(i,j)的每分钟收益（已截负为0），并用 z 作为门槛判断是否允许
    profit_per_min: Dict[Tuple[int, int], float] = {}
    for i, ch in enumerate(charge_slots):
        for j, dj in enumerate(discharge_slots):
            if dj.price > ch.price + z:
                # 收益不含 z：按（放电价 − 充电价）
                # 单位换算：RRP是AUD/MWh，能量是kWh，需要除以1000转换为MWh
                energy_mwh_per_min = ENERGY_KWH_PER_DISCHARGE_MIN / 1000.0
                unit = max(dj.price - ch.price, 0.0) * energy_mwh_per_min
            else:
                unit = 0.0
            profit_per_min[(i, j)] = unit

    # 构建 LP
    prob = pulp.LpProblem("max_profit", pulp.LpMaximize)
    x_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}

    for i in range(num_c):
        for j in range(num_d):
            x_vars[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, cat=pulp.LpContinuous)

    # 目标函数
    prob += pulp.lpSum(profit_per_min[(i, j)] * x_vars[(i, j)] for i in range(num_c) for j in range(num_d))

    # 行约束：每个充电段的总分配分钟 ≤ 1.3958
    for i in range(num_c):
        prob += pulp.lpSum(x_vars[(i, j)] for j in range(num_d)) <= REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT

    # 列约束：每个放电段的总分配分钟 ≤ 5
    for j in range(num_d):
        prob += pulp.lpSum(x_vars[(i, j)] for i in range(num_c)) <= DISCHARGE_SLOT_CAPACITY_MIN

    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved", "Undefined", "Infeasible", "Unbounded"):
        raise RuntimeError(f"PuLP 求解异常：{pulp.LpStatus[prob.status]}")

    # 还原分配结果
    allocations: List[ChargeAllocation] = []
    for i, ch in enumerate(charge_slots):
        alloc = ChargeAllocation(charge_index=i, charge_price=ch.price, required_minutes=REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT)
        total_alloc = 0.0
        for j, dj in enumerate(discharge_slots):
            val = float(x_vars[(i, j)].value() or 0.0)
            if val > 1e-9:
                # 还原片段的单位能量价差（不含 z）
                spread = max(dj.price - ch.price, 0.0)
                alloc.pieces.append(AllocationPiece(discharge_index=j, allocated_minutes=val, spread=spread))
                total_alloc += val
        alloc.unmet_minutes = max(0.0, REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT - total_alloc)
        allocations.append(alloc)

    # 放电剩余分钟
    discharge_remaining: Dict[int, float] = {}
    for j in range(num_d):
        used = sum(float(x_vars[(i, j)].value() or 0.0) for i in range(num_c))
        discharge_remaining[j] = max(0.0, DISCHARGE_SLOT_CAPACITY_MIN - used)

    total_profit = sum(a.profit for a in allocations)
    allocations.sort(key=lambda a: a.charge_index)
    return total_profit, allocations, discharge_remaining


def simulate_matching(
    charge_prices: List[float],
    discharge_prices: List[float],
    z: float,
    prefer: str = "min",
    strategy: str = "greedy",
) -> Tuple[float, List[ChargeAllocation], Dict[int, float]]:
    """
    统一入口：
    - strategy = "greedy"：贪心（multi）；
    - strategy = "opt"：全局最优（PuLP 线性规划）。
    """
    assert strategy in ("greedy", "opt")
    if strategy == "opt":
        return simulate_matching_opt(charge_prices, discharge_prices, z)
    return simulate_matching_greedy(charge_prices, discharge_prices, z, prefer)


def main() -> None:
    """命令行入口：生成随机价格、执行匹配，并输出小结与样例分配详情。

    常用参数：
    - --z：安全边际 z（越大越保守，成交更难）；
    - --seed：随机种子（复现实验）；
    - --prefer：候选排序（min 从低到高；max 从高到低，用于探索理论最优）；
    - --strategy：greedy（默认）或 opt（全局最优，需要 PuLP）。
    - --charge-mean/std、--dis-mean/std：充/放电价格的随机分布参数。
    """
    parser = argparse.ArgumentParser(description="Simulate charge/discharge matching with z-threshold and rate mismatch handling.")
    parser.add_argument("--z", type=float, default=0.0, help="阈值 z，使得只有当放电价 > 充电价 + z 时才匹配")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--prefer", type=str, default="min", choices=["min", "max"], help="候选放电价选择策略：min按放电价从小到大，max用于实验最优收益（仅对greedy有效）")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "opt"], help="分配策略：greedy贪心；opt全局最优（PuLP）")
    parser.add_argument("--charge-mean", type=float, default=0.0)
    parser.add_argument("--charge-std", type=float, default=50.0)
    parser.add_argument("--dis-mean", type=float, default=80.0)
    parser.add_argument("--dis-std", type=float, default=60.0)
    args = parser.parse_args()

    # 23:00-05:00 共6小时 -> 72个5分钟段
    num_charge_slots = 6 * 60 // SLOT_MINUTES
    # 05:00-08:00 共3小时 -> 36个5分钟段
    num_discharge_slots = 3 * 60 // SLOT_MINUTES

    # 生成演示用的充/放电价格序列
    charge_prices = generate_random_prices(
        num_slots=num_charge_slots,
        seed=args.seed,
        mean=args.charge_mean,
        stddev=args.charge_std,
        allow_negatives=True,
    )
    # 放电均值略高，常产生正向套利机会；通常不允许放电负价
    discharge_prices = generate_random_prices(
        num_slots=num_discharge_slots,
        seed=args.seed + 1 if args.seed is not None else None,
        mean=args.dis_mean,
        stddev=args.dis_std,
        allow_negatives=False,
    )

    total_profit, allocations, discharge_remaining = simulate_matching(
        charge_prices=charge_prices,
        discharge_prices=discharge_prices,
        z=args.z,
        prefer=args.prefer,
        strategy=args.strategy,
    )

    # 控制台输出小结与若干条样例
    print("参数：")
    print(f"  z = {args.z}")
    print(f"  策略 = {args.strategy}")
    print(f"  充电功率 = {CHARGE_POWER_KW} kW, 放电功率 = {DISCHARGE_POWER_KW} kW, 槽宽 = {SLOT_MINUTES} 分钟")
    print(f"  每个充电段需要的放电分钟 = {REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT:.4f} 分钟")
    print()

    served = sum(1 for a in allocations if a.unmet_minutes <= 1e-9)
    partially_served = sum(1 for a in allocations if 1e-9 < a.unmet_minutes < REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT)
    unserved = sum(1 for a in allocations if a.unmet_minutes >= REQUIRED_DISCHARGE_MIN_PER_CHARGE_SLOT - 1e-9)

    print("结果汇总：")
    print(f"  总收益: {total_profit:.2f}")
    print(f"  充电段数: {len(allocations)}, 完全匹配: {served}, 部分匹配: {partially_served}, 未匹配: {unserved}")
    print()

    # 展示前若干条详情，避免过长
    show_n = 10
    print(f"前{show_n}个充电段的分配详情（按原始索引）:")
    for a in allocations[:show_n]:
        print(f"  充电段#{a.charge_index:02d} 价={a.charge_price:7.2f} 需求分={a.required_minutes:.3f} -> 分配{a.allocated_minutes:.3f} 分, 未满足 {a.unmet_minutes:.3f} 分, 收益={a.profit:.2f}")
        if a.pieces:
            for p in a.pieces:
                print(f"    放电段#{p.discharge_index:02d} 分配{p.allocated_minutes:.3f} 分, 价差={p.spread:.2f}")
        else:
            print("    无可用放电候选（或价差不满足条件）")

    # 可选：展示尚有剩余时间的放电段数
    remaining_positive = sum(1 for v in discharge_remaining.values() if v > 1e-9)
    print()
    print(f"仍有剩余时间的放电段数量: {remaining_positive}/{len(discharge_remaining)}")


if __name__ == "__main__":
    main() 