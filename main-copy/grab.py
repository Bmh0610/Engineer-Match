"""Step_Grab: 定位纠偏 → 伸臂到转盘中心 → 抓取×3 放载物盘(三阶段同一 Step)。

前提约定: 车已停在转盘面前, 臂收在观察位(跑动中臂一直在这),
末端相机视野里只有离车最近的第 1 个物料。
流程(移植 gcxl2025-main: adjust_position_to_target 骨架 + "下探→夹→抬"三段式):
  A 定位纠偏: 看(稳定帧最近物料圆心) → 死区内收工 / move_v 微调后再看,
    超时告警继续(保完赛);
  C 抓取×3(定一抓三, 无伸臂对位步骤): 守主点上方等目标色进窗+转盘停歇
    连帧确认 → 按点控制机械臂(物料偏差→Δ伸出+夹爪15补偿/Δ回转) →
    下探/夹/抬 → 移载物盘/降/张/抬 → 回观察位待命。
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

        def _find(f):
            mats = finder.find_materials(self.backend, f, conf=0.5,
                                         px_per_mm=px)
            m = finder.nearest_to_axis(mats, f.shape)  # 主点正下方(离中心最近)
            if m is not None:                  # 圆盘定位精化
                finder.refine_material_stages(f, m, px)
            self._frame_shape = f.shape
            return m
        return _find

    @staticmethod
    def _draw_align(frame, obj, residual, round_no, sent=False):
        """纠偏可视化: 物料框+圆心+定位圈+残差(供 align(visualize=True))。"""
        vis = frame.copy()
        x1, y1, x2, y2 = map(int, obj.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis, (int(obj.cx), int(obj.cy)), 4, (0, 0, 255), -1)
        if getattr(obj, "ellipse", None):
            (hx, hy), (MA, ma), ang = obj.ellipse
            cv2.ellipse(vis, (int(hx), int(hy)),
                        (max(int(MA / 2), 3), max(int(ma / 2), 3)), ang, 0, 360,
                        (255, 255, 255), 1)
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
                print(f"  ✓ 纠偏结束 (残差 {residual:.1f}mm, {i} 轮)")
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
        print(f"  纠偏结束(带误差 {residual if residual is not None else -1:.1f}mm, "
              f"保完赛)")
        return False

    # ---- 阶段 B: 伸臂到转盘中心 ----

    # ---- 定点抓3(固定点位, 无臂视觉纠偏; 三层计数) ----

    def classify_slots(self, mats):
        """稳定帧三物料 → 角色字典 {近端/车头端/车尾端: Material}。

        近端 = 离画面中心最近(纠偏目标); 其余按图像 x 分:
        画面左 = 车头端(GRAB_FRONT), 画面右 = 车尾端(GRAB_REAR)
        —— 实拍定案; FRONT_REAR_SWAP 可翻(保险)。不足3个返回 None。
        """
        if len(mats) < 3:
            return None
        h, w = (self._frame_shape[:2] if hasattr(self, "_frame_shape")
                else (720, 1280))
        near = min(mats, key=lambda m: (m.cx - w / 2) ** 2 + (m.cy - h / 2) ** 2)
        others = [m for m in mats if m is not near]
        left, right = sorted(others, key=lambda m: m.cx)
        if C.FRONT_REAR_SWAP:
            left, right = right, left
        return {"near": near, "front": left, "rear": right}

    def grab_at_slot(self, slot_key, color, carrier, ack_counter):
        """定点抓一颗: 查角色点位 → 序列抓取 → 放物料盘 carrier 点。

        近端 7 步(下探一发直达——示教值已含相机-夹爪15mm差):
          下探→夹→抬2090→移盘→下探放→张→抬回;
        端位 8 步(伸150+转30 路径长, 先到上方再竖直下探防刮):
          移上方→下探→夹→抬2090→移盘→下探放→张→抬回。
        每步 send_and_wait; 每条完成回执 ack_counter[0] += 1(三层计数的单颗层)。
        返回 True=完成(子计数达 7/8)。
        """
        r, h, t = {"near": C.GRAB_NEAR, "front": C.GRAB_FRONT,
                   "rear": C.GRAB_REAR}[slot_key]
        # 抓取段
        if slot_key == "near":
            self.move_arm(r, h, t, label=f"近端下探({color})")      # 1
        else:
            self.move_arm(r, C.H_LIFT, t, label=f"端位上方({color})")  # 1
            self.move_arm(r, h, t, label="竖直下探")                # 2
        self.grip(close=True)                                       # +1
        self.move_arm(r, C.H_LIFT, t, label="抬到最高")             # +1
        # 放置段(三角色统一)
        self.move_arm(*carrier, label="移到物料盘上方")             # +1
        self.move_arm(carrier[0], carrier[1] - 200, carrier[2],
                      label="下探放料")                             # +1
        self.grip(close=False)                                      # +1
        self.move_arm(*C.OBSERVE, label="抬回观察位")               # +1
        ack_counter[0] += 8 if slot_key != "near" else 7
        expect = 8 if slot_key != "near" else 7
        return ack_counter[0] >= expect   # 单颗完成校验(简化: 累计口径)

    def goto_scan_pos(self):
        """到判色位(纠偏后伸 X+55mm+最高): 视野覆盖三物料, 判颜色/分类在此。"""
        self.move_arm(*C.SCAN_POS, label="到判色位")

    def run_grab3(self, task):
        """抓3 环节: 先到判色位(看全三物料) → 每停歇窗口重判角色 → 任务码颜色序定点抓 → 放盘。

        三层计数: 单动作 send_and_wait → 单颗 7/8 → 总 23(GRAB_ACK_TOTAL)。
        总数计满 = 环节结束; 窗口超时未满 = 看门狗告警(保完赛继续)。
        """
        counter = [0]   # 累计完成回执数
        grabbed = []
        self.goto_scan_pos()   # 先到判色位: 视野看全三物料再判颜色
        for color in task["batch1_colors"]:
            # 等停歇 + 识别(稳定帧): 三物料齐 → 分类角色
            obj, frame = finder.find_stable(self._finder_observe(), self.cam,
                                            timeout=C.WAIT_COLOR_TIMEOUT)
            if obj is None:
                print(f"  !! 窗口超时未等到停歇, {color} 跳过(保完赛)")
                continue
            mats = finder.find_materials(self.backend, frame, conf=0.5,
                                         px_per_mm=C.PX_PER_MM)
            slots = self.classify_slots(mats)
            if slots is None:
                print(f"  !! 物料不足 3 个, {color} 跳过(保完赛)")
                continue
            # 任务码颜色 → 角色
            target_key = next((k for k, m in slots.items() if m.color == color),
                              None)
            if target_key is None:
                print(f"  !! {color} 不在视野(可能已抓/识别失败), 跳过")
                continue
            carrier = {"P1": C.CARRIER_P1, "P2": C.CARRIER_P2,
                       "P3": C.CARRIER_P3}[f"P{len(grabbed) + 1}"]
            print(f"  抓 {color}: 角色={target_key} → 物料盘 P{len(grabbed)+1}")
            before = counter[0]
            self.grab_at_slot(target_key, color, carrier, counter)
            grabbed.append(color)
            print(f"    子计数 +{counter[0] - before} (累计 {counter[0]})")
        total = counter[0]
        if total >= C.GRAB_ACK_TOTAL:
            print(f"  ✓ 抓3 环节结束: 回执 {total}/{C.GRAB_ACK_TOTAL}")
        else:
            print(f"  !! 看门狗: 抓3 回执 {total}/{C.GRAB_ACK_TOTAL} 未满"
                  f" —— 有物料被跳过/异常, 保完赛继续")
        self.stats["grab_ok"], self.stats["grab_total"] = len(grabbed), 3
        return grabbed

    def run(self, task=None):
        """纠偏(定一) → 抓3(定点+计数)。task=qr.parse_task 结果或 None(默认码)。"""
        if task is None:
            from qr import parse_task
            task = parse_task(C.DEFAULT_TASK_CODE)
        print(f"== Step_Grab: 纠偏 → 抓3 {task['batch1_colors']} ==")
        self.align()
        self.run_grab3(task)
        return self.stats
