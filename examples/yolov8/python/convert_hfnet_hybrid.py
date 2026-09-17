import os
import re
import sys
from rknn.api import RKNN
from rknn.utils import onnx_edit

MODEL_ONNX = './hfnet_backbone.onnx'
MODEL_NCHW = './hfnet_backbone_nchw.onnx'
DATASET_TXT = './dataset.txt'
OUTPUT_RKNN = './hfnet_backbone_rk3566_i8.rknn'

def edit_quantization_cfg(cfg_path):
    """
    使用 YAML 规范安全地修改 .quantization.cfg 配置文件：
    将 Softmax 以及整个 detector head 相关层都设为 float16，
    防止 Softmax 接收到 INT8 (dtype 1) 输入导致报错。
    """
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found!")
        return False

    # 读取原始 cfg 文本内容以检索网络中的所有层名称
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 扫描与 detector、Softmax 相关的节点名
    EXCLUDE_KEYWORDS = [
        'DepthToSpace', 'depth_to_space', 'Slice', 'slice', 
        'Reshape', 'reshape', 'Transpose', 'transpose', 
        'Concat', 'concat', 'Split', 'split', 'Gather', 'gather',
        'Shape', 'shape', 'Squeeze', 'squeeze', 'Unsqueeze', 'unsqueeze'
    ]

    candidates = set()
    for line in raw_text.splitlines():
        line = line.strip()
        if any(k in line for k in ['detector', 'Softmax', 'logits']) and not any(ex in line for ex in EXCLUDE_KEYWORDS):
            tokens = re.findall(r'[\w\-\/\.]+(?::\d+)?', line)
            for t in tokens:
                if any(k in t for k in ['detector', 'Softmax', 'logits']) and not any(ex in t for ex in EXCLUDE_KEYWORDS):
                    if not t.endswith('.cfg') and not t.endswith('.model') and not t.endswith('.onnx'):
                        candidates.add(t)

    # 兜底添加核心计算节点（过滤掉结构变换节点）
    core_candidates = [
        "pred/local_head/detector/Softmax",
        "pred/local_head/detector/Softmax:0",
        "pred/local_head/detector/Conv2D",
        "pred/local_head/detector/Conv2D:0",
        "pred/local_head/detector/Conv",
        "pred/local_head/detector/Conv:0",
        "pred/local_head/detector/BiasAdd",
        "pred/local_head/detector/BiasAdd:0",
        "pred/local_head/detector/logits",
        "pred/local_head/detector/logits:0"
    ]
    for c in core_candidates:
        if not any(ex in c for ex in EXCLUDE_KEYWORDS):
            candidates.add(c)

    print(f"--> Found {len(candidates)} detector/Softmax related nodes to set as float16:")

    # 尝试使用 ruamel.yaml 或 pyyaml 加载并修改字典结构
    try:
        from ruamel.yaml import YAML
        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml_parser.load(f)
        if cfg is None:
            cfg = {}
        if 'custom_quantize_layers' not in cfg or cfg['custom_quantize_layers'] is None:
            cfg['custom_quantize_layers'] = {}
        
        for layer in sorted(candidates):
            cfg['custom_quantize_layers'][layer] = 'float16'
            print(f"    - '{layer}': float16")

        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml_parser.dump(cfg, f)

    except Exception as e:
        print(f"Using manual formatted write due to: {e}")
        # 如果 yaml 库加载失败，采用标准带引号的格式写入
        lines = []
        for line in raw_text.splitlines():
            if line.strip().startswith("custom_quantize_layers:"):
                continue
            lines.append(line)
        
        lines.append("custom_quantize_layers:")
        for layer in sorted(candidates):
            lines.append(f'  "{layer}": float16')
            print(f'    - "{layer}": float16')
        
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

    print("--> Successfully patched quantization.cfg.")
    return True

def main():
    # 1. 转换 ONNX 输入排布 (NHWC -> NCHW)
    if os.path.exists(MODEL_ONNX):
        print('--> 1. Editing ONNX model layout (NHWC -> NCHW)')
        try:
            ret = onnx_edit(
                model=MODEL_ONNX,
                export_path=MODEL_NCHW,
                inputs_transform={'image': 'a,b,c,d->a,d,b,c'}
            )
            if ret != 0:
                print('Edit ONNX model failed!')
                sys.exit(ret)
            print(f'Done. Saved NCHW model to {MODEL_NCHW}')
        except Exception as e:
            print(f'onnx_edit note: {e}')
    else:
        if not os.path.exists(MODEL_NCHW):
            print(f"Error: Neither {MODEL_ONNX} nor {MODEL_NCHW} exists!")
            sys.exit(-1)

    # 2. 初始化 RKNN 对象并配置平台
    rknn = RKNN(verbose=True)
    print('--> 2. Config model for RK3566')
    rknn.config(
        mean_values=[[0]],
        std_values=[[255.0]],
        target_platform='rk3566'
    )

    # 3. 载入模型
    print('--> 3. Loading ONNX model')
    ret = rknn.load_onnx(
        model=MODEL_NCHW,
        inputs=['image'],
        input_size_list=[[1, 1, 480, 640]]
    )
    if ret != 0:
        print('Load ONNX model failed!')
        rknn.release()
        sys.exit(ret)

    # 4. 执行混合量化 Step 1
    print('--> 4. Running hybrid_quantization_step1...')
    if not os.path.exists(DATASET_TXT):
        print(f"Error: {DATASET_TXT} not found! Please prepare dataset.txt first.")
        rknn.release()
        sys.exit(-1)

    ret = rknn.hybrid_quantization_step1(dataset=DATASET_TXT, proposal=False)
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        rknn.release()
        sys.exit(ret)
    rknn.release()

    # 5. 自动修改生成的 .quantization.cfg 文件
    cfg_file = MODEL_NCHW.replace('.onnx', '.quantization.cfg')
    model_file = MODEL_NCHW.replace('.onnx', '.model')
    data_file = MODEL_NCHW.replace('.onnx', '.data')

    if not edit_quantization_cfg(cfg_file):
        print("Failed to patch quantization.cfg!")
        sys.exit(-1)

    # 6. 执行混合量化 Step 2 并导出模型
    print('--> 5. Running hybrid_quantization_step2...')
    rknn2 = RKNN(verbose=True)
    ret = rknn2.hybrid_quantization_step2(
        model_input=model_file,
        data_input=data_file,
        model_quantization_cfg=cfg_file
    )
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        rknn2.release()
        sys.exit(ret)

    print(f'--> 6. Exporting RKNN model to {OUTPUT_RKNN}')
    ret = rknn2.export_rknn(OUTPUT_RKNN)
    if ret != 0:
        print('Export RKNN model failed!')
        rknn2.release()
        sys.exit(ret)

    print('===========================================================')
    print(f'SUCCESS! Hybrid quantized model saved to: {OUTPUT_RKNN}')
    print('===========================================================')
    rknn2.release()

if __name__ == '__main__':
    main()
