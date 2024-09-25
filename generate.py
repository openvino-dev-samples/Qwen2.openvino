from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig
from benchmark_util_4_29 import BenchmarkWrapper
import time

ov_config = {"PERFORMANCE_HINT": "LATENCY",
             "NUM_STREAMS": "1", "CACHE_DIR": ""}

model_dir = "./Qwen2-7B-Instruct-ov-sym-int4-1.0"

tokenizer = AutoTokenizer.from_pretrained(
    model_dir)
print("====Compiling model====")
ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device="GPU",
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir),
)
ov_model = BenchmarkWrapper(ov_model, do_print=True)


prompt = "What is AI?"

inputs = tokenizer(prompt, return_tensors="pt")
start = time.time()
outputs = ov_model.generate(**inputs, max_new_tokens=50)
end = time.time()
print("Total time: ", (end-start)*1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))