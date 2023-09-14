from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from lora import ExLlamaLora
import perplexity
from perplexity import Perplexity
import time
import torch
import torch.nn.functional as F
import argparse
import json
import math
import sys
import os
import glob
import model_init

torch.cuda._lazy_init()
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision = 10)
torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

cache = None
model = None

def begin():
    global model, cache

    if cache is None: cache = ExLlamaCache(model)
    else: cache.current_seq_len = 0


def next_logits(input_ids, apply_lora, last_id_only = True, input_mask = None):
    global model, cache

    # n_logits = None
    # a = 0
    # while a < input_ids.shape[-1]:
    #     b = min(input_ids.shape[-1], a + 2048)
    #     n_logits = model.forward(input_ids[:, a:b], cache, last_id_only, lora = apply_lora, input_mask = input_mask)
    #     a = b

    n_logits = model.forward(input_ids, cache, last_id_only, lora=apply_lora, input_mask=input_mask)
    return n_logits


def tokenize(text):
    global tokenizer

    return tokenizer.encode(text)


def timer(name, func):
    t = time.time()
    ret = func()
    t = time.time() - t
    print(f" ** Time, {name}: {t:.2f} seconds")
    return ret


mem_base = {}
mem_last = {}
for dev in torch_devices:
    torch.cuda.reset_peak_memory_stats(dev)
    mem_base[dev] = mem_last[dev] = torch.cuda.max_memory_allocated(dev)

def mem(name, total = False):
    global mem_base, mem_last

    res = f" ** VRAM, {name}: "
    first = True

    for device in torch_devices:
        mem_c = torch.cuda.max_memory_allocated(device)
        mem_this = mem_c - mem_last[device] if not total else mem_c - mem_base[device]
        mem_last[device] = mem_c

        if not first: res += " - "
        first = False
        res += f"[{device}] {mem_this / (1024 ** 2):,.2f} MB"

    print(res)


# Parse arguments

parser = argparse.ArgumentParser(description = "Benchmark tests for ExLlama")

model_init.add_args(parser)
perplexity.add_args(parser)

parser.add_argument("-p", "--perf", action = "store_true", help = "Benchmark speed and VRAM usage")
parser.add_argument("-v", "--validate", action = "count", help = "Run validation check and generate some sample output; specify twice for a more thorough test")
parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")
parser.add_argument("-ld", "--lora_dir", type = str, help = "Path to LoRA config and binary. to use during benchmark")
parser.add_argument( "--pl", type = int, help = "Prompt sequence length", default = 32)

args = parser.parse_args()

model_init.post_parse(args)
perplexity.post_parse(args)
model_init.get_model_files(args)

# Paths

if args.lora_dir is not None:
    args.lora_config = os.path.join(args.lora_dir, "adapter_config.json")
    args.lora = os.path.join(args.lora_dir, "adapter_model.bin")

# Feedback

print_opts = []
if args.perf: print_opts.append("perf")
if args.validate: print_opts.append("validate")
if args.perplexity: print_opts.append("perplexity")
if args.perplexity_token: print_opts.append("perplexity_token")

model_init.print_options(args, print_opts)

# Globals

model_init.set_globals(args)

# Instantiate model

config = model_init.make_config(args)

model = timer("Load model", lambda: ExLlama(config))
tokenizer = timer("Load tokenizer", lambda: ExLlamaTokenizer(args.tokenizer))

model_init.print_stats(model)

torch.cuda.reset_peak_memory_stats("cuda")
mem("Model")

cache = ExLlamaCache(model)
mem("Cache")

# Load LoRA

lora = None
if args.lora:
    print(f" -- LoRA config: {args.lora_config}")
    print(f" -- Loading LoRA: {args.lora}")
    if args.lora_config is None:
        print(f" ## Error: please specify lora path to adapter_config.json")
        sys.exit()
    lora = ExLlamaLora(model, args.lora_config, args.lora)
    if lora.bias_ignored:
        print(f" !! Warning: LoRA zero bias ignored")

# Test sequence
with open("prompt.json") as f:
    prompt_pool = json.load(f)



prompts_len = args.pl
max_seq_len = args.length
gen_tokens = max_seq_len - prompts_len

prompts = prompt_pool["llama2"][str(prompts_len)]
input_ids = tokenizer.encode(prompts)

print(f"input_ids {input_ids.size()}")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print(f"Prompts length: {prompts_len}, generate length: {gen_tokens}")



# Benchmark memory and performance

if args.perf:

    begin()


    model.config.matmul_recons_thd = 4
    generator = ExLlamaGenerator(model, tokenizer, cache)
    generator.settings.top_k = 1
    generator.lora = lora
    
    print("warm up..")
    for i in range(5):
        text,_ = generator.generate_simple(prompts, max_new_tokens = gen_tokens)
        print(f" ** Generation: {repr(text)}")
    
    print("benchmarking...")
    for i in range(10):
        t = time.time()
        text,ids = generator.generate_simple(prompts, max_new_tokens = gen_tokens)
        t = time.time() - t
        print(f"latency {t:.3f}, total tokens: {ids.size()[0]}")
        real_gen_tokens = ids.size()[0] - input_ids.size()[1]
        print(f"real generate tokens {real_gen_tokens}")
        #print(f" ** Generation: {repr(text)}")
        #print(ids)
        print(f" ** Speed: {real_gen_tokens / t:.3f} tokens/second")



