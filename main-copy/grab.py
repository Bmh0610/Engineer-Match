"""Step_Grab: 定位纠偏 → 伸臂到转盘中心 → 抓取×3 放载物盘(三阶段同一 Step)。

前提约定: 车已停在转盘面前, 臂收在观察位(跑动中臂一直在这),
末端相机视野里只有离车最近的第 1 个物料。
流程(移植 gcxl2025-main: adjust_position_to_target 骨架 + "下探→夹→抬"三段式):
  A 定位纠偏: 看(稳定帧最近物料圆心) → 死区内收工 / move_v 微调后再看,
    超时告警继续(保完赛);
  B 伸臂: move_a 到 CENTER 预设, 末端相机自检应见 3 物料(不足退回 A 重纠一次);
  C 抓取×3: 守中心位上方等目标色进窗+转盘停歇连帧确认(守株待兔) →
    出手前 move_v 微调 → 下探/夹/抬 → 移载物盘/降/张/抬 → 回中心位待命。
软失败(视觉超时/检不出)=告警+跳过, 保完赛; 硬失败(通信静默 LinkError)=抛出
终止 —— 臂/车位姿不明时继续盲编排不安全, 重摆重跑。
抓取微调走底盘 move_v: 臂与车刚性连接, 底盘微调即臂对物料的相对修正,
且与阶段 A 共用 IMG2BODY 一套轴向参数(臂系微调需另标 reach/theta 轴向, 备选)。
"""
import cv2
import math
import time

import config as C
import finder
from protocol import Mode


class StepGrab:
    def __init__(self, link, cam, backend):
        self.lk, self.cam, self.backend = link, cam, backend
        self.stats = {"grab_ok": 0, "grab_total": 0}

    # ---- 原语: 每条命令等对应完成码(单命令在途) ----

    def move_arm(self, reach, height, theta, label=""):
        self.lk.send_and_wait(
            lambda: self.lk.move_a(reach, height, theta),
            Mode.ACK_ARM, C.CMD_TIMEOUT,
            f"move_a({reach:.0f},{height:.0f},{theta:+.0f}) {label}")

    def grip(self, close):
        ang, t = C.GRIP_CLOSE if close else C.GRIP_OPEN
        self.lk.send_and_wait(lambda: self.lk.gripper(ang, t),
                              Mode.ACK_GRIPPER, C.CMD_TIMEOUT,
                              f"grip({'闭合' if close else '张开'})")

    def _move_v(self, dx_fwd, dy_right):
        """车体系微调: 前进 dx_fwd, 右移 dy_right (mm), 纠偏/抓取微调共用。"""
        self.lk.send_and_wait(lambda: self.lk.move_v(round(dy_right), round(dx_fwd), 0),
                              Mode.ACK_CHASSIS, C.CMD_TIMEOUT,
                              f"move_v(前{dx_fwd:+.0f}, 右{dy_right:+.0f})")

    @staticmethod
    def _to_body(obj):
        """图像偏差(像素) → 车体系 (前, 右)。见 finder.to_body(须喂像素)。"""
        dy_right, dx_fwd = finder.to_body(obj)
        return dx_fwd, dy_right

    # ---- 阶段 A: 定位纠偏 ----

    def _finder_observe(self):
        px = C.PX_PER_MM
        return lambda f: finder.nearest(
            finder.find_materials(self.backend, f, conf=0.5, px_per_mm=px))

    @staticmethod
    def _draw_align(frame, obj, residual, round_no, sent=False):
        """纠偏可视化: 物料框+圆心+霍夫圈+残差(供 align(visualize=True))。"""
        vis = frame.copy()
        x1, y1, x2, y2 = map(int, obj.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis, (int(obj.cx), int(obj.cy)), 4, (0, 0, 255), -1)
        if getattr(obj, "hough", None):
            hx, hy, hr = obj.hough
            cv2.circle(vis, (int(hx), int(hy)), max(int(hr), 3), (255, 255, 255), 1)
        dy_right = obj.dx_mm * C.IMG2BODY[0]
        dx_fwd = obj.dy_mm * C.IMG2BODY[1]
        tag = (f"align r{round_no}: X{dy_right:+.1f} Y{dx_fwd:+.1f}mm "
               f"res={residual:.1f}mm" + (" -> move_v sent" if sent else ""))
        cv2.putText(vis, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        return vis

    def align(self, visualize=False):
        """看-动-再看, 直到最近物料圆心进死区(gcxl adjust 骨架)。收敛返回 True。

        visualize=True: 每轮把检出画面(框/圆心/霍夫圈/残差)显示在 align 窗口
        ——需要显示环境(Nano 接屏/NoMachine), 无显示时自动降级为纯打印。
        """
        deadline = time.time() + C.ALIGN_TIMEOUT
        residual = None
        shown = False

        def show(frame_img, obj, res, i, sent=False):
            try:
                cv2.imshow("align", self._draw_align(frame_img, obj, res, i, sent))
                cv2.waitKey(1)
                return True
            except cv2.error:
                return False

        for i in range(1, C.ALIGN_MAX_ITER + 1):
            left = max(0.5, deadline - time.time())
            obj, frame_img = finder.find_stable(self._finder_observe(), self.cam,
                                                timeout=left)
            if obj is None:
                print(f"  [纠偏] 第{i}轮未检出稳定物料, 放弃对齐(继续流程)")
                return False
            dx_fwd, dy_right = self._to_body(obj)
            residual = math.hypot(dx_fwd, dy_right)
            if residual <= C.ALIGN_TOL_MM:
                if i > 1:
                    print(f"  [纠偏] 第{i}轮收工, 残差 {residual:.1f}mm")
                if visualize and frame_img is not None:
                    shown = show(frame_img, obj, residual, i)
                return True
            if time.time() > deadline:
                break
            print(f"  [纠偏] 第{i}轮: 前{dx_fwd:+.1f} 右{dy_right:+.1f} "
                  f"(残差 {residual:.1f}mm) → move_v")
            if visualize and frame_img is not None:
                shown = show(frame_img, obj, residual, i)
            self._move_v(dx_fwd, dy_right)
            if visualize and shown and frame_img is not None:
                show(frame_img, obj, residual, i, sent=True)
            time.sleep(C.ALIGN_SETTLE)   # 等车沉降再取帧
        print(f"  [纠偏] 超时/超次数, 残差 "
              f"{residual if residual is not None else -1:.1f}mm, 带误差继续(保完赛)")
        return False

    # ---- 阶段 B: 伸臂到转盘中心 ----

    def extend_center(self):
        """move_a 到中心位预设, 末端相机自检应见 3 物料。"""
        self.move_arm(*C.CENTER, label="伸臂到转盘中心")
        f = self.cam.read_latest()
        n = 0
        if f is not None:
            n = len(finder.find_materials(self.backend, f, conf=0.5,
                                          px_per_mm=C.PX_PER_MM))
        if n < 3:
            print(f"  !! 中心位只见 {n}/3 个物料: 可能未对准转盘中心或物料在背面 !!")
        return n >= 3

    # ---- 阶段 C: 守株待兔 + 三段式抓放 ----

    def grab_one(self, color):
        """等目标色进窗+转盘停歇 → 微调 → 抓(下探/夹/抬) → 放载物盘(降/张/抬)
        → 回中心位。软失败(超时)返回 False。"""
        m, _f = finder.wait_for_color(self.backend, self.cam, color,
                                      px_per_mm=C.PX_PER_MM)
        if m is None:
            print(f"  !! {color} 超时未进抓取窗(含换料等待), 跳过该色(计时不中断) !!")
            return False
        print(f"  确认 {color} conf={m.conf:.2f} 偏差px=({m.dx:+.1f},{m.dy:+.1f})")
        # 按点控制机械臂: 物料中心偏差(相对主点) → 臂增量(底盘不参与抓取)
        dy_right, dx_fwd = finder.to_body(m)   # 车体系 (右, 前) mm
        reach_c, h_c, theta_c = C.CENTER
        d_reach = dy_right + C.GRIP_REACH_COMP   # 物料偏臂侧→多伸; 夹爪差15补偿
        d_theta = (math.degrees(math.atan2(dx_fwd, reach_c))
                   if reach_c > 0 else 0.0)      # 物料偏车头→逆时针(正)
        print(f"  按点抓取 {color}: 物料 右{dy_right:+.1f} 前{dx_fwd:+.1f}mm → "
              f"Δ伸出{d_reach:+.1f} Δ回转{d_theta:+.2f}°")
        self.move_arm(reach_c + d_reach, C.H_GRAB, theta_c + d_theta,
                      label="按点对位下探")
        self.grip(close=True)
        self.move_arm(reach_c, h_c, theta_c, label="抬升")
        self.move_arm(*C.CARRIER, label="移到载物盘上方")
        self.move_arm(C.CARRIER[0], C.H_PLACE, C.CARRIER[2], label="下降放料")
        self.grip(close=False)
        self.move_arm(*C.CARRIER, label="抬升离盘")
        self.move_arm(*C.CENTER, label="回中心位待命")
        return True

    def run(self, colors=None):
        """三阶段串行: A+B(不收敛退回重纠一次) → C 逐色抓取。返回统计。"""
        colors = colors or C.GRAB_COLORS
        print(f"== Step_Grab: 纠偏 → 伸臂 → 抓取 {colors} ==")
        for attempt in (1, 2):
            self.align()
            if self.extend_center():
                break
            if attempt == 1:
                print("  退回观察位, 重纠一次")
                self.move_arm(*C.OBSERVE, label="回观察位")
        for color in colors:
            self.stats["grab_total"] += 1
            if self.grab_one(color):
                self.stats["grab_ok"] += 1
                print(f"  完成 {color}: 已放载物盘 "
                      f"({self.stats['grab_ok']}/{self.stats['grab_total']})")
        print(f"== Step_Grab 完成: 抓放 {self.stats['grab_ok']}/{self.stats['grab_total']} ==")
        return self.stats
