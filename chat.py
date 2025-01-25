import argparse
from typing import List, Tuple
from threading import Thread
import torch
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
from transformers import (AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids: List[int]):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] in self.token_ids

def convert_history_to_token(history: List[Tuple[str, str]], tokenizer: AutoTokenizer) -> torch.Tensor:
    """Convert chat history to model input tokens"""
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})
    
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-m', '--model_path', required=True, type=str, 
                       help='Path to OpenVINO optimized model directory')
    parser.add_argument('-l', '--max_sequence_length', default=256, type=int,
                       help='Maximum number of new tokens to generate (default: 256)')
    parser.add_argument('-d', '--device', default='CPU', type=str,
                       choices=['CPU', 'GPU', 'AUTO'], help='Inference device (default: CPU)')
    args = parser.parse_args()

    # Validate model path
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")

    # Initialize model and tokenizer
    try:
        print("====Loading tokenizer====")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        print("====Compiling model====")
        ov_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": ""
        }
        ov_model = OVModelForCausalLM.from_pretrained(
            args.model_path,
            device=args.device,
            ov_config=ov_config,
            config=AutoConfig.from_pretrained(args.model_path),
        )
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Setup streaming and stopping criteria
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    stop_tokens = StoppingCriteriaList([StopOnTokens([151643, 151645])])

    history = []
    print("====Starting conversation====")
    try:
        while True:
            try:
                input_text = input("用户: ").strip()
                if not input_text:
                    continue
                
                if input_text.lower() == 'stop':
                    break
                
                if input_text.lower() == 'clear':
                    history = []
                    print("AI助手: 对话历史已清空")
                    continue

                print("Qwen2-OpenVINO:", end=" ", flush=True)
                history.append([input_text, ""])
                
                # Generate response
                model_inputs = convert_history_to_token(history, tokenizer)
                generate_kwargs = {
                    "input_ids": model_inputs,
                    "max_new_tokens": args.max_sequence_length,
                    "temperature": 0.1,
                    "do_sample": True,
                    "top_p": 1.0,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "streamer": streamer,
                    "stopping_criteria": stop_tokens,
                    "pad_token_id": 151645,
                }

                # Start generation thread
                t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
                t1.start()

                # Stream output
                partial_text = ""
                for new_text in streamer:
                    print(new_text, end="", flush=True)
                    partial_text += new_text
                
                history[-1][1] = partial_text
                print("\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError during generation: {e}")
                history.pop() if history else None

    finally:
        print("\nCleaning up resources...")

if __name__ == "__main__":
    main()
