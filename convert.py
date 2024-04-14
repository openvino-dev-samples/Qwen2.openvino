from transformers import AutoTokenizer, AutoConfig
from optimum.intel import OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM

import os
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        default='Qwen/Qwen1.5-0.5B-Chat',
                        required=False,
                        type=str,
                        help='orignal model path')
    parser.add_argument('-p',
                        '--precision',
                        required=False,
                        default="int4",
                        type=str,
                        choices=["fp16", "int8", "int4"],
                        help='fp16, int8 or int4')
    parser.add_argument('-o',
                        '--output',
                        default='./Qwen1.5-0.5B-ov',
                        required=False,
                        type=str,
                        help='Required. path to save the ir model')
    args = parser.parse_args()

    ir_model_path = Path(args.output)
    if ir_model_path.exists() == False:
        os.mkdir(ir_model_path)
        
    compression_configs = {
        "sym": False,
        "group_size": 128,
        "ratio": 0.8,
    }

    print("====Exporting IR=====")
    if args.precision == "int4":
        ov_model = OVModelForCausalLM.from_pretrained(args.model_id, export=True,
                                                      compile=False, quantization_config=OVWeightQuantizationConfig(
                                                          bits=4, **compression_configs))
    elif args.precision == "int8":
        ov_model = OVModelForCausalLM.from_pretrained(args.model_id, export=True,
                                                      compile=False, load_in_8bit=True)
    else:
        ov_model = OVModelForCausalLM.from_pretrained(args.model_id, export=True,
                                                      compile=False, load_in_8bit=False)

    ov_model.save_pretrained(ir_model_path)

    print("====Exporting tokenizer=====")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(ir_model_path)
