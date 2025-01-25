import argparse
import openvino_genai
from pathlib import Path

def streamer(subword):
    """Callback function for streaming output, returns False to continue streaming"""
    print(subword, end='', flush=True)
    return False

def main():
    parser = argparse.ArgumentParser(
        description='OpenVINO Generative AI Chat Interface',
        add_help=False
    )
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-m', '--model_path', 
                        required=True,
                        type=str,
                        help='Path to the OpenVINO IR model directory')
    parser.add_argument('-l', '--max_sequence_length',
                        default=256,
                        type=int,
                        help='Maximum number of tokens to generate (default: 256)')
    parser.add_argument('-d', '--device',
                        default='CPU',
                        choices=['CPU', 'GPU', 'AUTO'],
                        help='Device for inference (default: CPU)')
    args = parser.parse_args()

    # Validate model path
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")

    try:
        # Initialize pipeline
        pipe = openvino_genai.LLMPipeline(args.model_path, args.device)
        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = args.max_sequence_length

        pipe.start_chat()
        print("Chat session started. Type your message (Ctrl+C to exit)...")
        
        while True:
            try:
                prompt = input('\nQuestion:\n> ')
                if not prompt.strip():
                    continue
                
                print('\nAnswer:')
                pipe.generate(prompt, config, streamer)
                print('\n' + '-'*40)
                
            except KeyboardInterrupt:
                print("\nExiting chat session...")
                break
            except Exception as e:
                print(f"\nError during generation: {e}")
                break

    finally:
        # Ensure proper cleanup
        if 'pipe' in locals():
            pipe.finish_chat()

if __name__ == "__main__":
    main()
