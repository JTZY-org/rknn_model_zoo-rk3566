import os
import sys
from rknn.api import RKNN

MODEL_ONNX = './hfnet_backbone.onnx'
DATASET_TXT = './dataset.txt'
OUTPUT_RKNN = './hfnet_backbone_rk3566_i8.rknn'

def main():
    if not os.path.exists(MODEL_ONNX):
        print(f"Error: {MODEL_ONNX} not found!")
        sys.exit(-1)

    if not os.path.exists(DATASET_TXT):
        print(f"Error: {DATASET_TXT} not found! Please prepare dataset.txt first.")
        sys.exit(-1)

    rknn = RKNN(verbose=True)

    print('--> 1. Config model for RK3566 (NHWC Native Layout)')
    # 使用原生 NHWC 布局，Softmax 位于最后一维 (axis=-1)，完全支持 RK3566 NPU INT8 硬件加速
    rknn.config(
        mean_values=[[0]],
        std_values=[[255.0]],
        target_platform='rk3566',
        quantized_algorithm='normal',
        quantized_method='channel'
    )

    print(f'--> 2. Loading native ONNX model: {MODEL_ONNX}')
    # HFNet 原生 ONNX 模型为 NHWC 格式: [1, 480, 640, 1]
    ret = rknn.load_onnx(
        model=MODEL_ONNX,
        inputs=['image'],
        input_size_list=[[1, 480, 640, 1]]
    )
    if ret != 0:
        print('Load ONNX model failed!')
        rknn.release()
        sys.exit(ret)

    print('--> 3. Building model with INT8 quantization')
    ret = rknn.build(
        do_quantization=True,
        dataset=DATASET_TXT
    )
    if ret != 0:
        print('Build INT8 model failed!')
        rknn.release()
        sys.exit(ret)

    print(f'--> 4. Exporting RKNN model to {OUTPUT_RKNN}')
    ret = rknn.export_rknn(OUTPUT_RKNN)
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        sys.exit(ret)

    print('===========================================================')
    print(f'SUCCESS! Native NHWC INT8 model saved to: {OUTPUT_RKNN}')
    print('===========================================================')
    rknn.release()

if __name__ == '__main__':
    main()
