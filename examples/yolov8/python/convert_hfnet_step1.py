import os
import sys
from rknn.api import RKNN
from rknn.utils import onnx_edit

MODEL_ONNX = './hfnet_backbone.onnx'
MODEL_NCHW = './hfnet_backbone_nchw.onnx'
DATASET_TXT = './dataset.txt'

def main():
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

    rknn = RKNN(verbose=True)
    print('--> 2. Config model for RK3566')
    rknn.config(
        mean_values=[[0]],
        std_values=[[255.0]],
        target_platform='rk3566'
    )

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

    print('--> 4. Running hybrid_quantization_step1...')
    ret = rknn.hybrid_quantization_step1(dataset=DATASET_TXT, proposal=False)
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        rknn.release()
        sys.exit(ret)

    print('========================================================================')
    print('Step 1 Complete!')
    print('Generated config: ./hfnet_backbone_nchw.quantization.cfg')
    print('Please check/edit custom_quantize_layers in the cfg, then run Step 2.')
    print('========================================================================')
    rknn.release()

if __name__ == '__main__':
    main()
