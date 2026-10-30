"""上位机 ↔ STM32 串口协议(12 字节定长帧, 电控 2026-10 约束)。

帧: FF | MODE u16 | X i16 | Y i16 | Z i16 | CRC16 | FE
大端; CRC16-Modbus 覆盖 MODE~Z 共 8 字节。
坐标系约定(与电控逐条对过):
  0x0001 move_v 底盘: 车体系相对位移, 原点=车体中心、初始 0;
      +x = 机械臂一侧(右), +y = 车头, z = 旋转角°(顺时针/右转为负, 逆时针为正
      —— 2026-10-20 实车确认, 最初"+朝臂侧即右转"的说法反了); 单位 mm/°
  0x0002 move_a 机械臂: X = 伸出距离 mm, Y = 离地高度 mm, Z = 回转角度°
      (顺时针/往车尾为负, 逆时针/往车头为正)
  0x0003 夹爪: X = 角度(分度 0.1°), Y = 执行时间 ms
0x0004 舵盘: X = 目标角度(分度 1°, 逆时针+/顺时针-, 从结束位标志起算)
      闭合 (1400, 500), 张开 (800, 500) —— 数值在 config.GRIP_CLOSE/GRIP_OPEN
上行: 0x0011 底盘完成 | 0x0012 机械臂完成 | 0x0013 夹爪完成 | 0x0014 舵盘完成
      | 0x2201 一键启动·短按(→启停区1) | 0x2202 一键启动·长按(→启停区2)
分立完成码区分谁完成, 不带失败码, 数据字段为 0。
⚠️ 这里的 0x2201 与 14B 协议《上下位机通信协议-给电控.md》的 0x2201(启动, 无区分,
全 0 帧)不是一回事——两条协议线串口不共存; 长短按与舵盘完成帧口径见根目录
《12字节协议-给电控-一键启动与舵盘完成帧.md》(2026-10-29 定, 旧 0x0020 退役)。
超时策略: 单命令在途(全流程同时只有一条未决命令), 单窗口 35s 静默即抛错
不重发 —— 下位机无同帧去重, 重发会双走。
"""
import struct
import time

import serial

FRAME_LEN = 12
_FMT = ">Hhhh"   # mode x y z


class Mode:
    MOVE_V = 0x0001        # 底盘: 车体系相对位移 (x 右, y 前, z 右转°)
    MOVE_A = 0x0002        # 机械臂: (伸出距离, 离地高度, 回转角°)
    GRIPPER = 0x0003       # 夹爪: (角度 0.1°, 执行时间 ms)
    DIAL = 0x0004          # 舵盘转位: X=目标角度(分度1°, 逆时针+/顺时针-,
                           #  从结束位标志起算; 2026-10-29 电控定)
    ACK_CHASSIS = 0x0011   # 底盘完成
    ACK_ARM = 0x0012       # 机械臂完成
    ACK_GRIPPER = 0x0013   # 夹爪完成
    ACK_DIAL = 0x0014      # 舵盘完成: 任何舵盘命令(回零找结束位/转位)完成即发
                           # (命令-回执式, 不做上电主动上报; 下行 MODE 待电控定)
    BUTTON_START = 0x2201  # 一键启动·短按(T<2s)→启停区1 (旧 0x0020 退役, 2026-10-29 换值)
    BUTTON_START_2 = 0x2202  # 一键启动·长按(T≥2s)→启停区2


class LinkError(RuntimeError):
    pass


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def frame(mode: int, x=0, y=0, z=0) -> bytes:
    body = struct.pack(_FMT, mode, int(x), int(y), int(z))
    return b"\xFF" + body + struct.pack(">H", crc16(body)) + b"\xFE"


class Frame:
    def __init__(self, buf: bytes):
        self.mode, self.x, self.y, self.z = struct.unpack(_FMT, buf[1:9])

    def __repr__(self):
        return f"Frame(0x{self.mode:04X}, {self.x}, {self.y}, {self.z})"


def decode(buf: bytearray):
    """定长剥帧: 找 FF → 凑 12 字节 → 验 FE → 验 CRC。
    半帧保留返回 None; 坏帧丢弃继续找, 找不到返回 None。"""
    while True:
        start = buf.find(b"\xFF")
        if start < 0:
            buf.clear()
            return None
        if start > 0:
            del buf[:start]
        if len(buf) < FRAME_LEN:
            return None
        if (buf[FRAME_LEN - 1] == 0xFE
                and struct.unpack_from(">H", buf, 9)[0] == crc16(buf[1:9])):
            f = Frame(bytes(buf[:FRAME_LEN]))
            del buf[:FRAME_LEN]
            return f
        del buf[:1]   # 伪帧头或坏帧, 从下一字节重找


class Link:
    def __init__(self, port: str, baudrate: int = 115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.05)
        except serial.SerialException as e:
            import os
            if not os.path.exists(port):
                raise RuntimeError(
                    f"{port} 不存在: 板上 CH340 桥未枚举——整车断电重上电/拔插"
                    f" Nano 侧 USB 线后重试 (ls {port}* 确认)") from e
            if "Permission" in str(e):
                raise RuntimeError(
                    f"{port} 无权限: sudo chmod 666 {port} 临时开, "
                    f"或 usermod -aG dialout 后重新登录(永久)") from e
            raise
        self._rx = bytearray()

    # ---- 下行 ----

    def move_v(self, dx: float, dy: float, dz: float = 0.0):
        """底盘(0x0001): 参数=预期原值(mm/°, 无×10! test_paotu 实车验证)。
        车体系相对位移: dx 右, dy 前, dz 旋转°(顺时针/右转为负;
        ⚠️ z 超出 ±180 会被固件 NormalizeAngle 归一化——z=-900 实测转 180°)。"""
        self.ser.write(frame(Mode.MOVE_V, round(dx), round(dy), round(dz)))

    def move_a(self, reach: float, height: float, theta: float = 0.0):
        """机械臂(0x0002): 参数=预期×10(0.1mm/0.1°, 电控样例与实测验证)。
        XYZ 为绝对目标: X 伸出距离(+x 与车同向), Y 离地高度(最高参数2090
        = 预期209mm), Z 回转角°(逆时针/朝车头+; 绝对语义, 回 0 即归位)。"""
        self.ser.write(frame(Mode.MOVE_A, round(reach), round(height), round(theta)))

    def gripper(self, angle_01deg: int, time_ms: int):
        """夹爪: 角度(0.1° 分度, 如 1300=130.0°), 执行时间 ms。"""
        self.ser.write(frame(Mode.GRIPPER, angle_01deg, time_ms))

    def dial(self, angle_deg: int):
        """舵盘(0x0004): 转盘转位, X=目标角度(分度 1°, 逆时针+/顺时针-, 从结束位起算)。
        完成回执 ACK_DIAL(0x0014)。"""
        self.ser.write(frame(Mode.DIAL, int(angle_deg), 0, 0))

    # ---- 上行 ----

    def poll(self) -> list:
        self._rx.extend(self.ser.read(256))
        frames = []
        while True:
            f = decode(self._rx)
            if f is None:
                return frames
            frames.append(f)

    # ---- 单命令在途 ----

    def flush(self):
        """清残留回执, 防上一条的迟到完成码冒充本条结果。"""
        while self.poll():
            pass

    def send_and_wait(self, send, ack_mode: int, timeout: float = 35.0,
                      label: str = "") -> Frame:
        """发一条命令并等它的完成码。单窗口超时抛错不重发(重发会双走)。
        靠"同一时刻只有一条未决命令"的纪律保证完成码不会张冠李戴。"""
        self.flush()
        send()
        deadline = time.time() + timeout
        while time.time() < deadline:
            for f in self.poll():
                if f.mode == ack_mode:
                    return f
            time.sleep(0.03)
        raise LinkError(
            f"{label or hex(ack_mode)} {timeout:.0f}s 无完成回执: "
            f"帧未达或回执被干扰; 不重发(下位机无同帧去重), 重摆重跑")
