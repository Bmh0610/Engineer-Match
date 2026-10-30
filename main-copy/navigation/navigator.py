"""导航模块: 场地地图 + A* 规划 + 原语分解 + 封边回退。

原语: ("rotate", dth) 原地自转 rad | ("move", dx, dy, arrival) 场地系增量 mm + 到达节点
地图从 YAML 加载(map.yaml); 障碍策略 = 见障碍封当前边, 负增量倒回后重规划。
"""
import heapq
import math

import yaml


# ---- 地图 ----

class Map:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.rotation_sign = int(data.get("rotation_sign", 1))
        self.nodes = {int(i): (v["name"], *v["xy"], v["type"])
                      for i, v in data["nodes"].items()}
        self.edges = [tuple(sorted(e[:2])) for e in data["edges"]]
        # 逐边附加费(edges 可选第三项, 缺省 0): 车道级风险差(外圈宽好走 0.1/内十字窄 0.2),
        # 只进 astar 选路代价, 不改 dist(执行距离仍是几何长度); 按边对称存取, 不分方向。
        self.fee: dict[tuple[int, int], float] = {}
        for e in data["edges"]:
            if len(e) > 2:
                self.fee[tuple(sorted(e[:2]))] = float(e[2])
        self._adj: dict[int, list[int]] = {i: [] for i in self.nodes}
        for a, b in self.edges:
            self._adj[a].append(b)
            self._adj[b].append(a)
        self.blocked: set[tuple[int, int]] = set()

    def neighbors(self, nid: int) -> list[int]:
        return [m for m in self._adj[nid] if tuple(sorted((nid, m))) not in self.blocked]

    def block_edge(self, a: int, b: int) -> None:
        self.blocked.add(tuple(sorted((a, b))))

    def dist(self, a: int, b: int) -> float:
        return math.hypot(self.nodes[b][1] - self.nodes[a][1],
                          self.nodes[b][2] - self.nodes[a][2])


# ---- 代价(抽象, 无量纲; 联调时调比例看选路行为) ----

COST = {
    "edge": 1.0,                           # 每条边按 边长/1000 计(兼顾±100随机偏移)
    "turn": 0.5,                           # 每 90° 原地转向, 180° 自动为 2 倍
    "junction": {2: 0.1, 3: 0.2, 4: 0.3},  # 途经路口费, 按节点度数自动分档: L/T/十字
}


def _dir_of(dx: float, dy: float) -> int:
    """边方向 → 离散朝向 0=+x 1=+y(南) 2=-x 3=-y(北)。"""
    if dx > 0:
        return 0
    if dx < 0:
        return 2
    return 1 if dy > 0 else 3


# ---- 加权 A* ----

def astar(m: Map, start: int, goal: int, start_dir: int = 0,
          stop_dir: int | None = None) -> list[int]:
    """在 (节点, 朝向) 状态空间上做最短路。

    代价 = Σ边(edge) + Σ转向(turn) + 途经路口费(起点/终点不收) + 末转向(对齐停车朝向)。
    返回节点序列; 不可达抛 ValueError。36 个状态, 直接 Dijkstra 不用启发。
    """
    def tcost(d1: int, d2: int) -> float:
        q = abs(d1 - d2) % 4
        return min(q, 4 - q) * COST["turn"]

    def jfee(n: int) -> float:
        return 0.0 if n in (start, goal) else COST["junction"].get(len(m._adj[n]), 0.0)

    start_s = (start, start_dir)
    dist = {start_s: 0.0}
    came: dict = {}
    openq = [(0.0, start_s)]
    while openq:
        gc, s = heapq.heappop(openq)
        if gc > dist.get(s, math.inf):
            continue
        n, d = s
        for nb in m.neighbors(n):
            na, nbpos = m.nodes[n], m.nodes[nb]
            d2 = _dir_of(nbpos[1] - na[1], nbpos[2] - na[2])
            s2 = (nb, d2)
            ng = (gc + m.dist(n, nb) / 1000 + tcost(d, d2) + jfee(n)
                  + m.fee.get(tuple(sorted((n, nb))), 0.0))
            if ng < dist.get(s2, math.inf):
                dist[s2] = ng
                came[s2] = s
                heapq.heappush(openq, (ng, s2))
    ends = []
    for d in range(4):
        g = dist.get((goal, d), math.inf)
        ends.append(g + (tcost(d, stop_dir) if stop_dir is not None else 0.0))
    total = min(ends)
    if total == math.inf:
        raise ValueError(f"节点 {goal} 不可达(起点 {start})")
    s = (goal, ends.index(total))
    path = [s[0]]
    while s in came:
        s = came[s]
        path.append(s[0])
    return path[::-1]


# ---- 导航 ----

class Navigator:
    def __init__(self, m: Map):
        self.map = m
        self.pos: int = 0            # 当前所在节点
        self.heading: float = 0.0    # 当前朝向, 地图坐标系 rad

    def plan(self, target: int, stop_heading: float | None = None) -> list:
        """从当前位置规划到 target 的原语序列(可带停车朝向)。self.path = 节点序列。"""
        start_dir = round(self.heading / (math.pi / 2)) % 4
        stop_dir = round(stop_heading / (math.pi / 2)) % 4 if stop_heading is not None else None
        path = astar(self.map, self.pos, target, start_dir, stop_dir)
        self.path = path
        prims = []
        for a, b in zip(path, path[1:]):
            na, nb = self.map.nodes[a], self.map.nodes[b]
            target_h = math.atan2(nb[2] - na[2], nb[1] - na[1])
            dth = (target_h - self.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(dth) > 1e-3:
                prims.append(("rotate", dth * self.map.rotation_sign))
                self.heading = target_h
            prims.append(("move", nb[1] - na[1], nb[2] - na[2], b))
        if stop_heading is not None:
            dth = (stop_heading - self.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(dth) > 1e-3:
                prims.append(("rotate", dth * self.map.rotation_sign))
                self.heading = stop_heading
        return prims

    def advance_to(self, node: int) -> None:
        """运动模块收到 move 的 DONE 后调用。"""
        self.pos = node

    def block_and_replan(self, target: int, dx: float, dy: float, arrival: int,
                         stop_heading: float | None = None) -> list:
        """行驶中发现障碍: 封当前边(pos→arrival), 原路倒回, 重新规划。"""
        self.map.block_edge(self.pos, arrival)
        self.heading = math.atan2(dy, dx)   # 倒回后车头仍朝原边方向
        # MOVE_FIELD 是"当前位置+增量"的闭环目标, 负增量即从任意中途位置精确倒回起点
        return [("move", -dx, -dy, self.pos)] + self.plan(target, stop_heading)
