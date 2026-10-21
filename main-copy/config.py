"""main-copy 集中配置: 串口/相机/模型/标定/纠偏/抓取/预设位姿。

改参数只改这个文件。带"待示教/待标定"的项留占位, 由操作手自己填。
坐标系定义见 protocol.py 模块注释。
"""
import math

# ---- 通信 ----
SERIAL_PORT = "/dev/ttyUSB0"   # USB 直连; 排针备用=/dev/ttyTHS1
CMD_TIMEOUT = 35.0             # 单命令单窗口(秒): 静默不重发(防双走)

# ---- 相机 / 模型 ----
CAMERA_ID_END = 0              # 末端相机(眼在手, 垂直向下); 编号随 USB 口变, 实测后只改这
MODEL_PATH = "/home/popcorn/桌面/train yolov11/best.pt"   # 统一 10 类; .pt/.engine 自适应

# ---- 标定 ----
# 内参: python3 tools/calib_intr.py 三步生成 vision/calib_end.yaml
# 比例尺: python3 tools/calib_ratio.py 实测(px/mm), 填这里:
PX_PER_MM = 4.054              # 2026-10-20 标定: 6 次两点测距均值(±0.027, 离散 0.7%),
                               # 交叉验证 650.8/4.054=160.5mm vs 尺量 166mm 差 3.3% ✓
                               # 相机工作高度变了要重标; None=回退 1.0 并告警
# 注: 若以后两个工作位置高度不同(臂升降改变相机高度), 拆成 PX_PER_MM_OBSERVE /
#     PX_PER_MM_CENTER 两值, 并把 grab.py 里的引用按阶段分开即可。

# ---- 阶段A 定位纠偏(移植 gcxl adjust_position_to_target 骨架) ----
ALIGN_TOL_MM = 3.0       # 死区: 残差 ≤ 此值收工(gcxl 4px×0.72 同量级)
ALIGN_MAX_ITER = 4       # 最多几轮 看-动-再看
ALIGN_TIMEOUT = 30.0     # 总超时(秒), 超时告警继续(保完赛)
ALIGN_SETTLE = 0.3       # 每轮 move_v 后等车沉降再取帧
STABLE_FRAMES = 2        # 连续 N 帧中心位移 < STABLE_TOL_PX 才采用(滤框抖)
STABLE_TOL_PX = 2.0
# 图像偏差→车体系符号: (图像x→车体右 dy, 图像y→车体前 dx) 的系数。
# 末端相机绕光轴安装角未定, 联调时车前后平移看符号, 只改这里。
IMG2BODY = (1.0, -1.0)

# ---- 阶段C 守株待兔(转盘停歇时物料静止再出手) ----
GRAB_WINDOW = (0.30, 0.70, 0.30, 0.70)   # 抓取窗(画面归一化 x0,x1,y0,y1)
CONFIRM_CONF = 0.85      # 颜色确认门槛: 认错色=0 分, 认不出只是慢
CONFIRM_FRAMES = 3       # 连续 N 帧同色才认(滤反光闪色)
FRAME_GAP = 0.05         # 采帧间隔 > 帧周期, 保证不是同一帧
WAIT_COLOR_TIMEOUT = 30.0   # 单色等待超时(含换料等待), 超时跳过该色
ADJUST_ON_GRAB = True    # 出手前对该物料做一次 move_v 微调(死区内自动跳过)

# ---- 物料(10 类模型里的物料子集, 类别号见 detect.py) ----
CLS2COLOR = {1: "red", 2: "green", 3: "blue",
             4: "yellow", 5: "light_blue", 6: "black"}
MATERIAL_CLS = list(CLS2COLOR)
GRAB_COLORS = ["red", "green", "blue"]   # 默认抓取顺序(调试用; 正式由任务码传入)

# ---- 夹爪(电控给定: 角度 0.1° 分度, 执行时间 ms) ----
GRIP_CLOSE = (1300, 500)   # 闭合: 130.0°, 500ms
GRIP_OPEN = (800, 500)     # 张开: 80.0°, 500ms

# ---- 预设位姿(臂系: reach 伸出距离 mm, height 离地高度 mm, theta 回转角°) ----
# 全部待示教, 由你自己摆好臂后填数(示教法: 电控调试口读实际位姿抄数)。
OBSERVE = (0.0, 0.0, 0.0)   # 观察位: 臂缩回+走车安全高, 跑动中臂一直在这
CENTER = (0.0, 0.0, 0.0)    # 中心位: 转盘中心正上方(height=该处安全高)
CARRIER = (0.0, 0.0, 0.0)   # 载物盘位: 车载物盘正上方(height=该处安全高)
H_GRAB = 0.0                # 取料高度(离地 mm): 转盘面 80~100 + 夹爪行程, 示教定
H_PLACE = 0.0               # 放料高度(离地 mm): 载物盘面高, 示教定

# ---- 显示调试帧窗口(联调用, 比赛关) ----
SHOW_WINDOW = False
