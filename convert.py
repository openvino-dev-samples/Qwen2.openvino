import os
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from optimum.intel import OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-m', '--model_id', default='Qwen/Qwen1.5-0.5B-Chat', required=False, type=str, help='Original model path')
    parser.add_argument('-p', '--precision', required=False, default="int4", type=str, choices=["fp16", "int8", "int4"], help='Precision: fp16, int8, or int4')
    parser.add_argument('-o', '--output', required=False, type=str, help='Path to save the IR model')
    parser.add_argument('-ms', '--modelscope', action='store_true', help='Download model from Model Scope')
    args = parser.parse_args()

    # Set output path
    ir_model_path = Path(args.model_id.split("/")[1] + '-ov') if args.output is None else Path(args.output)
    ir_model_path.mkdir(exist_ok=True)  # Create directory if it doesn't exist

    # Define compression configurations
    compression_configs = {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    }

    # Download model from ModelScope if specified
    if args.modelscope:
        print("====Downloading model from ModelScope=====")
        from modelscope import snapshot_download
        model_path = snapshot_download(args.model_id, cache_dir='./')
    else:
        model_path = args.model_id

    # Export IR model based on precision
    print("====Exporting IR=====")
    try:
        if args.precision == "int4":
            ov_model = OVModelForCausalLM.from_pretrained(
                model_path, export=True, compile=False,
                quantization_config=OVWeightQuantizationConfig(bits=4, **compression_configs)
            )
        elif args.precision == "int8":
            ov_model = OVModelForCausalLM.from_pretrained(
                model_path, export=True, compile=False, load_in_8bit=True
            )
        else:
            ov_model = OVModelForCausalLM.from_pretrained(
                model_path, export=True, compile=False, load_in_8bit=False
            )
    except Exception as e:
        print(f"Error exporting IR model: {e}")
        return

    # Save the IR model
    print("====Saving IR=====")
    try:
        ov_model.save_pretrained(ir_model_path)
    except Exception as e:
        print(f"Error saving IR model: {e}")
        return

    # Export and save the tokenizer
    print("====Exporting tokenizer=====")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(ir_model_path)
    except Exception as e:
        print(f"Error exporting tokenizer: {e}")
        return

    # Export tokenizer for OpenVINO
    print("====Exporting IR tokenizer=====")
    try:
        from optimum.exporters.openvino.convert import export_tokenizer
        export_tokenizer(tokenizer, ir_model_path)
    except Exception as e:
        print(f"Error exporting IR tokenizer: {e}")
        return

    print("====Finished=====")

if __name__ == '__main__':
    main()
