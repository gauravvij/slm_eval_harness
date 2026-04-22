import time
import psutil
import json
import os
from llama_cpp import Llama

def measure_ram_at_context(model_path, n_ctx):
    print(f"Measuring RAM for n_ctx={n_ctx}...")
    # Load model with specific context window
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=32,
        verbose=False
    )
    
    # Warm up with a small prompt
    prompt = "Explain the concept of quantization in 50 words."
    llm(prompt, max_tokens=10)
    
    # Measure peak RSS
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    
    # Cleanup
    del llm
    time.sleep(2)
    
    return round(ram_gb, 2)

def run_scaling_test():
    model_path = "/root/slm_eval_harness/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    contexts = [4096, 8192, 16384, 32768]
    results = {}
    
    for ctx in contexts:
        ram = measure_ram_at_context(model_path, ctx)
        results[str(ctx)] = ram
        
    output_path = "/root/slm_eval_harness/reports/context_scaling_gemma4.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Context scaling results saved to {output_path}")

if __name__ == "__main__":
    run_scaling_test()
