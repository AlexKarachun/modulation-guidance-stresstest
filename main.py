from IPython.display import display
import types
import torch
from functools import partial
from diffusers import FluxPipeline
import diffusers
from models.flux_schnell import encode_prompt, forward_modulation_guidance, register_gate_scaling
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor
import json


# Import a model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", dtype=torch.bfloat16).to('cuda')
pipe.set_progress_bar_config(disable=True)


# Define the hyperparametrs: 
# 1. Prompts: Generation prompt, positive and negative prompts
# 2. Modulation guidance strength (w)
# 3. Guidance start layer
prompt_positive = "Ultra-detailed, photorealistic, cinematic"
prompt_negative = "Low-res, flat, cartoonish"

w = 3
start_layer = 5

# conditions per prompt (row j of the batch gets condition j % N_COND):
# 0: x | 1: x* | 2: x* with norm of x | 3: x with norm of x* | 4: x with gates * GATE_FACTOR
N_COND = 5
GATE_FACTOR = 1.03



# Get pooled CLIP embeddings
clip_positive = encode_prompt(pipe=pipe, prompt=prompt_positive)
clip_negative = encode_prompt(pipe=pipe, prompt=prompt_negative)

forward_logger = []

# Change forward of the pipe using the prompts and guidance weight
forward_modulation_guidance_partial = partial(forward_modulation_guidance, 
                                      pooled_projections_1=clip_positive, 
                                      pooled_projections_0=clip_negative,
                                      w=w, start_layer=start_layer, 
                                      log=forward_logger, n_cond=N_COND)
pipe.transformer.forward = types.MethodType(forward_modulation_guidance_partial, pipe.transformer)
register_gate_scaling(pipe.transformer, GATE_FACTOR, start_layer=start_layer, n_cond=N_COND, cond=4)







REPO_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    REPO_DIR / 'dataset/coco5000.csv',
    sep='|',
    names=['idx', 'text']
)

data = df.values.tolist()

'''
data = 
[[0, 'A woman stands flying a kite with both hands.'],
 [1, 'Many people on their bikes are near a pink vehicle.'],
 [2, 'A man holds a huge remote control in a store.'],
 [3, 'Little boys are playing t ball on a field.'],
 [4, 'The reflection of a man in barbers chair getting a trim'],
 [5, 'A group of girls on a field playing soccer'],
...
'''



main_dir = REPO_DIR / 'generations'
save_dir = main_dir / 'coco5000'
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / 'config.json', 'w') as f:
    
    json.dump({
        'p+': prompt_positive,
        'p-': prompt_negative,
        'w': w, 
        'start_layer': start_layer,
        'num_inference_steps': 4,
        'seed': 'prompt idx (same noise for all conditions of a prompt)',
        'gate_factor': GATE_FACTOR,
        'conditions': {
            '0': 'x',
            '1': 'x* = x + w(x+ - x-)',
            '2': 'x* rescaled to norm of x',
            '3': 'x rescaled to norm of x*',
            '4': 'x, gates of blocks >= start_layer multiplied by gate_factor',
        },
        'files': '{idx}_{cond}.png',
    }, 
    fp=f,
    indent=2,
    ensure_ascii=False,
    )
    

with ThreadPoolExecutor(max_workers=2) as saver:
    
    
    
    ratio_sum = 0
    n = 0
    
    pbar = tqdm(data, desc=f'промптики', position=0)
    for idx, prompt in pbar:
        
        if all((save_dir / f"{idx}_{i}.png").exists() for i in range(N_COND)):
            continue
            

        images = pipe([prompt] * N_COND,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=[torch.Generator("cpu").manual_seed(idx) for _ in range(N_COND)],
                    output_type='pil').images

        

        for i, img in enumerate(images):
            saver.submit(img.save, save_dir / f"{idx}_{i}.png", compress_level=1)
        
        
        with open(save_dir/'logs.pkl', 'ab') as f:
            for item in forward_logger:
                item['prompt_idx'] = idx
                pickle.dump(item, file=f)
        
        
        norms = forward_logger[-1]['norm(temb_mix,dim=-1)']
        ratio = (norms[1] / norms[0]).item()
        
        ratio_sum += ratio
        n += 1
        
        pbar.set_postfix({'x*/x': f'{ratio_sum / n:.3f}'}, idx=idx)

        forward_logger.clear()




'''
B=4 -> 12.2s per batch ~ 3s per img ~ 17h total

cuncur
B=4 -> 10.85s per batch ~ 2.7s per img ~ 15h total



coco20 ~112M -> 27.4 gb total




hf upload AlexKarachun/mod_stress generations/coco5000 . --repo-type dataset \
    --every 10
    

hf upload AlexKarachun/mod_stress generations/coco5000 . --repo-type dataset 


    
    
до экспа
- 

'''


