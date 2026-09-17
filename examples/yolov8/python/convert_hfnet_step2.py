import os
import sys
from rknn.api import RKNN

MODEL_FILE = './hfnet_backbone_nchw.model'
DATA_FILE = './hfnet_backbone_nchw.data'
CFG_FILE = './hfnet_backbone_nchw.quantization.cfg'
OUTPUT_RKNN = './hfnet_backbone_rk3566_i8.rknn'

def main():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE) and os.path.exists(CFG_FILE)):
        print("Error: Required files from Step 1 are missing! Please run Step 1 first.")
        sys.exit(-1)

    rknn = RKNN(verbose=True)

    print('--> Running hybrid_quantization_step2...')
    ret = rknn.hybrid_quantization_step2(
        model_input=MODEL_FILE,
        data_input=DATA_FILE,
        model_quantization_cfg=CFG_FILE
    )
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        rknn.release()
        sys.exit(ret)

    print(f'--> Exporting RKNN model to {OUTPUT_RKNN}')
    ret = rknn.export_rknn(OUTPUT_RKNN)
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        sys.exit(ret)

    print('========================================================================')
    print(f'SUCCESS! Hybrid quantized model saved to: {OUTPUT_RKNN}')
    print('========================================================================')
    rknn.release()

if __name__ == '__main__':
    main()
