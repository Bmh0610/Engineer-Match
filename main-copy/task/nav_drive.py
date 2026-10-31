"""导航调度层(避障思想落地): 节点检测制逐边执行 + 封边重规划。

移植自 main 架构(视觉检测 + 封边 + A*), 按 12B 实车协议(move_v 车体系)重写执行层。
与 main 版 Task1 的差异:
  ① 执行原语: MOVE_FIELD 场地系闭环 → move_v 车体系相对量 + 上位机换算(无场地系兜底);
  ② 转向: 绝对朝向 rotate_to → 相对角 z + 上位机朝向记账(打滑会失真, 无固件朝向反馈),
     转向符号全项目只在 turn_z 一处(×rotation_sign, 镜像系 rotation_sign=-1 联调锁定);
  ③ 节点检测制: 转向对准后 check_edge 扫下一条边, 命中=封边重规划; 行驶中不扫描、
     无监听线程、无中途回退——障碍都在出发前发现, 车不驶入封锁边, 回退没有触发时机;
  ④ 协议无 ESTOP: 不可达/命令静默超时/封边超限 = 异常上传硬失败, 人工重摆重跑;
  ⑤ 执行纪律(2026-10-31 用户定): 全程车头朝行进方向——只有前进+原地转向, 无横移行驶;
     避障传感器装车头, 转向对准后正对待行边(节点检测制的物理前提);
     "零转向横移执行"方案已否决。

组装示例(前相机避障版):
    from vision.camera import Camera
    from vision.detect import YOLOBackend
    from vision.obstacle import ObstacleWatcher
    from task.nav_drive import NavDriver
    backend = YOLOBackend(C.MODEL_PATH)          # 共用引擎, infer(classes=[0]) 阶段门控
    front = Camera(<前相机设备号>, "front"); front.open()
    driver = NavDriver(lk, ObstacleWatcher(backend, front), "navigation/map.yaml",
                       start_node=1, start_heading=math.pi / 2)
    driver.launch()                              # 发车对位腿(不入图), 之后才可 goto
    driver.goto(6)                               # 例: 启1 → 物料盘(自动对齐工作点朝向)
watcher=None 或 check=False = 纯跑图(不检测)。代价模型与校准见 _explain。
"""
import math
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C  # noqa: E402
from communication.protocol import Mode  # noqa: E402
from navigation.navigator import COST, Map, Navigator  # noqa: E402

# 发车对位腿(车体系 右移,前进 mm, 2026-10-31 更正: 实际距离=170/190, 初记 17/19 系口径笔误;
# move_v 分度=1mm——实车实测 move_v(100)走100mm/z=90转90° 佐证): 摆车位→车道中心节点,
# 发车后先走、不入图。启1=右移170+前进190(实测); 启2=镜像预测左移170+前进190, 待实车卷尺复核。
# (与守株版老路线首腿"右移150(起步对位)"同量级, 旧 10 倍悬案随之闭合)
START_JOG = {1: (170.0, 190.0), 3: (-170.0, 190.0)}

# 工作点停车朝向(固定系 rad): 臂=车体右侧=朝向 h-π/2 ⟹ "臂朝区域" ⟹ h=区域方向+π/2。
# 物料盘(6)车头朝西(+x)臂朝北; 粗加工(4)车头朝东(-x)臂朝南; 暂存(8)车头朝南(+y)臂朝西。
# (与 main 线 STATION_HEAD 物理口径交叉验证一致); goto 默认自动查表, 联调确认。
WORK_HEADING = {6: 0.0, 4: math.pi, 8: math.pi / 2}


def turn_z(target_h, heading, rotation_sign):
    """地图系目标朝向 → move_v 相对转角 z(度, 顺时针为负), 归一 ±π 永远转劣弧。
    z=0=无需转向(调用方跳过发送)。转向符号全项目只在此一处:
    镜像系(rotation_sign=-1)下地图系正角=物理逆时针=+z。"""
    dth = (target_h - heading + math.pi) % (2 * math.pi) - math.pi
    return -round(math.degrees(dth)) * rotation_sign


class NavDriver:
    """调度层: ObstacleWatcher(节点检测) + Map/Navigator(封边重规划) + Link.move_v(执行)。"""

    def __init__(self, lk, watcher, map_yaml, start_node, start_heading):
        self.lk = lk
        self.watcher = watcher          # ObstacleWatcher 或 None(纯跑图)
        self.nav = Navigator(Map(map_yaml))
        self.start_node = start_node
        self.start_heading = start_heading
        self.heading = start_heading    # 当前朝向(固定系 rad), 记账真值
        self.rotation_sign = self.nav.map.rotation_sign

    def _send_move(self, x, y, z, label):
        """一条 move_v 等 ACK_CHASSIS 完成码; 静默超时抛错不重发(单命令在途)。"""
        self.lk.send_and_wait(lambda: self.lk.move_v(x, y, z),
                              Mode.ACK_CHASSIS, C.CMD_TIMEOUT, label)
        time.sleep(2)                   # 动作间 2s(等车稳, 与 drive() 同节奏)

    def launch(self, jog=None):
        """发车对位腿(摆车位→车道中心节点, 车体系 右移,前进 mm)。对位不入图,
        走完才置 pos/heading 开始导航记账——须先于 goto 调用。"""
        if jog is None:
            jog = START_JOG.get(self.start_node)
        if jog:
            self._send_move(jog[0], jog[1], 0,
                            f"发车对位(右{jog[0]:+.0f} 前{jog[1]:+.0f})")
        self.nav.pos = self.start_node
        self.heading = self.start_heading

    def _explain(self, path, stop_heading=None):
        """代价明细(联调校准权重用): 按 navigator.COST/dist+fee 重算分项, 只打印。
        锚点: 1.0 代价≈4s(850mm@600mm/s+2s 命令间隙); turn 标定=实测90°总耗时/
        实测850mm总耗时×0.85; 路口费红线=同一路线费用总和<一次转向费;
        180°=2×90° 是有意偏差——固件 ±180° 对准边界是已知死区, 多罚掉头=规划天然避开。"""
        m, total = self.nav.map, 0.0
        hdir = round(self.heading / (math.pi / 2)) % 4
        print(f"  规划 {path} (朝向档{hdir}):")
        for i, (a, b) in enumerate(zip(path, path[1:])):
            dx = m.nodes[b][1] - m.nodes[a][1]
            dy = m.nodes[b][2] - m.nodes[a][2]
            d = 0 if dx > 0 else 2 if dx < 0 else (1 if dy > 0 else 3)
            q = abs(d - hdir) % 4
            edge = m.dist(a, b) / 1000
            turn = min(q, 4 - q) * COST["turn"]
            j = 0.0 if i == 0 else COST["junction"].get(len(m.neighbors(path[i])), 0.0)
            fee = m.fee.get(tuple(sorted((a, b))), 0.0)
            total += edge + turn + j + fee
            print(f"    {a}→{b}: 边{edge:.2f} 转向{turn:.2f} 路口{j:.2f} 边费{fee:.2f}")
            hdir = d
        if stop_heading is not None:
            q = abs(round(stop_heading / (math.pi / 2)) % 4 - hdir) % 4
            turn = min(q, 4 - q) * COST["turn"]
            total += turn
            print(f"    末对齐{round(math.degrees(stop_heading)):+d}°: 转向{turn:.2f}")
        print(f"    总代价 {total:.2f} (1.0≈4s)")

    def goto(self, target, stop_heading=None, check=True):
        """导航到 target 节点(节点检测制逐边循环)。

        stop_heading=None 自动查 WORK_HEADING(工作点自动对齐, 显式传 rad 可覆盖)。
        保证链: 查表 → plan 按"末转向代价"选到达方向 → 末尾补残差转身 → ACK 完成码。
        check=False 或 watcher=None = 纯跑图。
        封边超 config.MAX_OBSTACLES / 不可达 / 命令静默超时 → 异常上传(硬失败重摆重跑)。
        """
        if stop_heading is None:
            stop_heading = WORK_HEADING.get(target)
        self._blocks = 0
        while True:
            # plan() 会把 nav.heading 推进到路线末端, 每次重规划前先对齐记账真值
            self.nav.heading = self.heading
            prims = self.nav.plan(target, stop_heading)
            self._explain(self.nav.path, stop_heading)
            moves = [(p[1], p[2], p[3]) for p in prims if p[0] == "move"]
            blocked = False
            for dx, dy, arrival in moves:
                target_h = math.atan2(dy, dx)      # 该边地图系朝向(正交网格=±90°档)
                z = turn_z(target_h, self.heading, self.rotation_sign)
                if z:
                    self._send_move(0, 0, z, f"转向{z:+d}°")
                self.heading = target_h            # 转向已执行(含 z=0), 记账提交
                if check and self.watcher and self.watcher.check_edge():
                    print(f"  ⚠ 节点{self.nav.pos}: 边→{arrival} 有障碍, 封边重规划")
                    self.nav.map.block_edge(self.nav.pos, arrival)
                    self._blocks += 1
                    if self._blocks > C.MAX_OBSTACLES:
                        raise RuntimeError(
                            f"封边 {self._blocks} 次超 MAX_OBSTACLES={C.MAX_OBSTACLES}"
                            f"(疑似误报或地图全断), 中止导航")
                    blocked = True
                    break
                self._send_move(0, math.hypot(dx, dy), 0,
                                f"前进{math.hypot(dx, dy):.0f} → 节点{arrival}")
                self.nav.advance_to(arrival)
            if not blocked:
                if stop_heading is not None and self.heading != stop_heading:
                    z = turn_z(stop_heading, self.heading, self.rotation_sign)
                    if z:
                        self._send_move(0, 0, z, f"工作点对齐{z:+d}°")
                    self.heading = stop_heading
                return
