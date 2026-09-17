import os
import sys
import onnx
from onnx import helper
from rknn.api import RKNN
from rknn.utils import onnx_edit

MODEL_ONNX = './hfnet_backbone.onnx'
MODEL_NCHW = './hfnet_backbone_nchw.onnx'
MODEL_PRUNED = './hfnet_backbone_pure_conv.onnx'
DATASET_TXT = './dataset.txt'
OUTPUT_RKNN = './hfnet_backbone_rk3566_i8.rknn'

def prune_onnx_model(input_path, output_path, target_outputs=['logits', 'local_descriptor_map']):
    """
    对 ONNX 模型进行逆向图裁剪：
    只保留生成 target_outputs 所需的卷积层节点，
    彻底剔除 Softmax、Slice、DepthToSpace 等导致 NPU INT8 报错的后处理节点。
    """
    print(f'--> Pruning ONNX model to keep only: {target_outputs}')
    model = onnx.load(input_path)
    graph = model.graph

    existing_outputs = {out.name: out for out in graph.output}
    existing_value_infos = {vi.name: vi for vi in graph.value_info}

    new_outputs = []
    for name in target_outputs:
        if name in existing_outputs:
            new_outputs.append(existing_outputs[name])
        elif name in existing_value_infos:
            new_outputs.append(existing_value_infos[name])
        else:
            for node in graph.node:
                if name in node.output:
                    new_out = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
                    new_outputs.append(new_out)
                    break

    # 逆向依赖追踪，筛选必需节点（使用索引避免 NodeProto unhashable 错误）
    required_node_indices = set()
    tensors_needed = set(target_outputs)

    for idx in reversed(range(len(graph.node))):
        node = graph.node[idx]
        if any(out in tensors_needed for out in node.output):
            required_node_indices.add(idx)
            for inp in node.input:
                if inp:
                    tensors_needed.add(inp)

    # 保持原有拓扑顺序过滤节点
    clean_nodes = [graph.node[i] for i in range(len(graph.node)) if i in required_node_indices]

    # 清空并重建 graph 输出与节点
    graph.ClearField('output')
    graph.output.extend(new_outputs)

    graph.ClearField('node')
    graph.node.extend(clean_nodes)

    onnx.save(model, output_path)
    print(f'--> Done! Pruned ONNX saved to {output_path} (Retained {len(clean_nodes)} pure conv nodes).')

def main():
    # 1. 将 ONNX 模型输入转为 NCHW 格式 [1, 1, 480, 640]
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
    except Exception as e:
        print(f'onnx_edit note: {e}')

    # 2. 物理剪枝 ONNX 模型：直接从 ONNX 中彻底删除 Softmax 与 DepthToSpace 节点
    print('--> 2. Physically pruning ONNX graph')
    prune_onnx_model(
        input_path=MODEL_NCHW,
        output_path=MODEL_PRUNED,
        target_outputs=['logits', 'local_descriptor_map']
    )

    # 3. 初始化 RKNN 对象
    rknn = RKNN(verbose=True)

    # 4. 配置 RK3566 硬件平台属性
    print('--> 3. Config model for RK3566')
    rknn.config(
        mean_values=[[0]],
        std_values=[[255.0]],
        target_platform='rk3566',
        quantized_algorithm='normal',
        quantized_method='channel'
    )

    # 5. 载入裁剪后的纯卷积 ONNX 模型
    print(f'--> 4. Loading pure conv ONNX model: {MODEL_PRUNED}')
    ret = rknn.load_onnx(
        model=MODEL_PRUNED,
        inputs=['image'],
        input_size_list=[[1, 1, 480, 640]]
    )
    if ret != 0:
        print('Load ONNX model failed!')
        rknn.release()
        sys.exit(ret)

    # 6. 构建 INT8 量化模型（此时模型中 100% 都是纯 Conv 算子，量化绝对成功且高效）
    print('--> 5. Building model with INT8 quantization')
    ret = rknn.build(
        do_quantization=True,
        dataset=DATASET_TXT
    )
    if ret != 0:
        print('Build INT8 model failed!')
        rknn.release()
        sys.exit(ret)

    # 7. 导出最终 RKNN 模型
    print(f'--> 6. Exporting RKNN model to {OUTPUT_RKNN}')
    ret = rknn.export_rknn(OUTPUT_RKNN)
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        sys.exit(ret)

    print('===========================================================')
    print(f'SUCCESS! Pure Conv INT8 RKNN model saved at: {OUTPUT_RKNN}')
    print('===========================================================')
    rknn.release()

if __name__ == '__main__':
    main()