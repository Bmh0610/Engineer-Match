"""YOLO 统一 10 类识别后端: .pt(ultralytics) / .engine(TensorRT) 双后端。

统一 10 类单模型: 障碍物 + 物料 6 色 + 数字圆环 1/2/3。
本模块只回答"帧里有什么、在哪": 帧进 → 检测框出(类号/类名/置信度/xyxy)。
颜色白名单、圆心偏差换算等解码逻辑在 finder.py 做。

后端按模型路径后缀自适应:
  .pt  → ultralytics(笔记本开发; Nano 亦可但需 torch)
  .engine → TensorRT 10(Nano 部署, JetPack 自带 TRT + cuda-python 管显存;
           engine 由 trtexec --onnx=best.onnx --fp16 生成, 无 NMS 端到端,
           输出 [1, 4+nc, N], NMS 在本模块后处理做)
infer() 接口两后端完全一致。
"""
from collections import namedtuple
from threading import Lock

CLASSES = {
    0: "obstacle_black",
    1: "object_red",
    2: "object_green",
    3: "object_blue",
    4: "object_yellow",
    5: "object_light_blue",
    6: "object_black",
    7: "circle1",
    8: "circle2",
    9: "circle3",
}

# 网络输出用低门槛(conf 0.25), 业务阈值(如颜色确认 0.85 才出手)归解码层自定
DEFAULTS = {"imgsz": 960, "conf": 0.25, "iou": 0.45}

Detection = namedtuple("Detection", "cls_id cls conf xyxy")   # xyxy = (x1, y1, x2, y2) 像素


class _TrtRunner:
    """TensorRT 10 推理(移植 gcxl2025-main trt_utils 思路, cuda-python 管显存)。

    假设静态输入 (1,3,H,W) float32; 输出 [1, 4+nc, N] 原始预测(NMS 本地做)。
    """

    def __init__(self, path, imgsz):
        import numpy as np
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.np, self.trt, self.cudart = np, trt, cudart
        self.imgsz = imgsz
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, 'rb') as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            io.append((name,
                       self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                       tuple(self.engine.get_tensor_shape(name)),
                       trt.nptype(self.engine.get_tensor_dtype(name))))
        self.in_name = next(n for n, is_in, _, _ in io if is_in)
        self.outs = [(n, s, d) for n, is_in, s, d in io if not is_in]
        self.in_shape = dict((n, s) for n, is_in, s, _ in io if is_in)[self.in_name]
        (err,) = cudart.cudaFree(0)                    # 激活 primary context
        assert err == cudart.cudaError_t.cudaSuccess, err
        err, self.stream = cudart.cudaStreamCreate()
        self.d_in, self.in_nbytes = None, int(np.prod(self.in_shape)) * 4
        self.d_out = {}
        self.h_out = {}
        for n, s, d in self.outs:
            nb = int(np.prod(s)) * np.dtype(d).itemsize
            err, ptr = cudart.cudaMalloc(nb)
            assert err == cudart.cudaError_t.cudaSuccess, err
            self.d_out[n] = ptr
            self.h_out[n] = np.empty(s, d)
            self.context.set_tensor_address(n, ptr)
        self._in_alloc()

    def _in_alloc(self):
        nb = self.in_nbytes
        err, ptr = self.cudart.cudaMalloc(nb)
        assert err == self.cudart.cudaError_t.cudaSuccess, err
        self.d_in = ptr
        self.context.set_tensor_address(self.in_name, ptr)

    def run(self, blob):
        """blob: 预处理好的 CHW float32(已 letterbox 到输入尺寸) → list[np 数组]。
        按 engine 输入 shape 对齐(补 batch 维), 防字节数不匹配。"""
        np, cudart = self.np, self.cudart
        blob = np.ascontiguousarray(blob, np.float32).reshape(self.in_shape)
        (err,) = cudart.cudaMemcpyAsync(self.d_in, blob.ctypes.data, self.in_nbytes,
                                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                        self.stream)
        assert err == cudart.cudaError_t.cudaSuccess, err
        assert self.context.execute_async_v3(self.stream)
        outs = []
        for n, s, d in self.outs:
            nb = int(np.prod(s)) * np.dtype(d).itemsize
            (err,) = cudart.cudaMemcpyAsync(self.h_out[n].ctypes.data, self.d_out[n], nb,
                                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                            self.stream)
            assert err == cudart.cudaError_t.cudaSuccess, err
        (err,) = cudart.cudaStreamSynchronize(self.stream)
        assert err == cudart.cudaError_t.cudaSuccess, err
        return [self.h_out[n].copy() for n, _, _ in self.outs]

    def letterbox(self, frame):
        """等比缩放+灰边填充到输入尺寸, 返回 (blob CHW float32, 缩放参数)。"""
        import cv2
        np = self.np
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[(self.imgsz - nh) // 2:(self.imgsz - nh) // 2 + nh,
               (self.imgsz - nw) // 2:(self.imgsz - nw) // 2 + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), r, (self.imgsz - nw) // 2, (self.imgsz - nh) // 2


def _postprocess(pred, nc, imgsz, conf_thres, iou_thres):
    """[1, 4+nc, N] 原始预测(如 960 输入 N=18900) → (cls, conf, xyxy@960) 列表。"""
    import cv2
    import numpy as np
    p = pred[0]                                   # (4+nc, N)
    boxes_cxcywh, scores_all = p[:4], p[4:4 + nc]
    cls_ids = scores_all.argmax(0)
    confs = scores_all.max(0)
    keep = confs >= conf_thres
    if not keep.any():
        return []
    b = boxes_cxcywh[:, keep].T                   # (M,4) cxcywh
    confs_k, cls_k = confs[keep], cls_ids[keep]
    xy = b[:, :2]
    wh = b[:, 2:4]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    idx = cv2.dnn.NMSBoxes(xyxy.tolist(), confs_k.tolist(), conf_thres, iou_thres)
    out = []
    for i in np.array(idx).reshape(-1):
        out.append((int(cls_k[i]), float(confs_k[i]),
                    tuple(float(v) for v in xyxy[i])))
    return out


class YOLOBackend:
    def __init__(self, model_path: str):
        self.path = model_path
        self.lock = Lock()   # 多消费方共用一个引擎时串行化
        self._warmup()

    def _warmup(self):
        import numpy as np
        self.infer(np.zeros((self.imgsz(), self.imgsz(), 3), np.uint8))

    def imgsz(self):
        if self.path.endswith('.engine'):
            import numpy as np
            r = _TrtRunner(self.path, DEFAULTS["imgsz"])   # 先建一次读输入尺寸
            s = r.in_shape[-1]
            del r
            return int(s)
        return DEFAULTS["imgsz"]

    def infer(self, frame, classes=None):
        """识别一帧(BGR ndarray), 返回 list[Detection]。

        classes: 保留的类别 id 列表(如物料 [1..6]); None = 全部 10 类。
        """
        with self.lock:
            if self.path.endswith('.engine'):
                return self._infer_trt(frame, classes)
            return self._infer_ul(frame, classes)

    def _infer_ul(self, frame, classes):
        from ultralytics import YOLO
        if not hasattr(self, 'model'):
            self.model = YOLO(self.path, task="detect")
        results = self.model.predict(frame, classes=classes, verbose=False, **DEFAULTS)
        out = []
        for box in results[0].boxes:
            cid = int(box.cls)
            out.append(Detection(cid, CLASSES[cid], float(box.conf),
                                 tuple(float(v) for v in box.xyxy[0])))
        return out

    def _infer_trt(self, frame, classes):
        import numpy as np
        if not hasattr(self, 'trt'):
            self.trt = _TrtRunner(self.path, DEFAULTS["imgsz"])
        blob, r, pad_x, pad_y = self.trt.letterbox(frame)
        preds = self.trt.run(blob)
        nc = len(CLASSES)
        dets = _postprocess(preds[0], nc, DEFAULTS["imgsz"],
                            DEFAULTS["conf"], 0.45)
        out = []
        for cid, conf, (x1, y1, x2, y2) in dets:
            if classes and cid not in classes:
                continue
            # 960 坐标 → 原图: 去灰边, 除缩放
            x1, x2 = (x1 - pad_x) / r, (x2 - pad_x) / r
            y1, y2 = (y1 - pad_y) / r, (y2 - pad_y) / r
            out.append(Detection(cid, CLASSES.get(cid, f"id{cid}"), conf,
                                 (x1, y1, x2, y2)))
        return out
