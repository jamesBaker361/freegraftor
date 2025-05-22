import os
# os.environ['FLUX_DEV'] = '/workspace/yzb/pretrained/huggingface/diffusers/FLUX.1-dev'
# os.environ['GROUNDING_DINO'] = '/workspace/yzb/pretrained/huggingface/grounding-dino-tiny'
# os.environ['SAM'] = './sam_vit_h_4b8939.pth'

import argparse
from pathlib import Path
import json 
from collections import defaultdict
from pipeline import FreeGraftorPipeline, ConceptConfig

def main(args, models=None):
    pipeline = FreeGraftorPipeline(
        models=models, 
        image_cache_dir=args.image_cache_dir,
        image_info_cache_dir=args.image_info_cache_dir
    )
    
    concept_configs = []
    if Path(args.concept_config_path).is_file():
        with open(args.concept_config_path, 'r') as f:
            raw_concept_configs = json.load(f)
        
        for raw_concept_config in raw_concept_configs:
            concept_configs.append(ConceptConfig(**raw_concept_config))
    else:
        for class_name, image_path in zip(args.class_names, args.image_paths):
            concept_configs.append(ConceptConfig(class_name=class_name, image_path=image_path)) 
            
    seed = args.start_seed
    
    for idx in range(args.num_images):   
        def get_nested_dict():
            return defaultdict(get_nested_dict)
        info = get_nested_dict()
        
        info['seed'] = seed
        info['num_steps'] = args.num_steps
        info['guidance'] = args.guidance
        info['width'] = args.width if args.width % 16 == 0 else args.width - args.width % 16
        info['height'] = args.height if args.height % 16 == 0 else args.height - args.height % 16
        
        info['start_inject_step'] = args.start_inject_step
        info['end_inject_step'] = args.end_inject_step
        info['inject_block_ids'] = list(range(args.start_inject_block, args.end_inject_block+1))
        info['sim_threshold'] = args.sim_threshold
        info['cyc_threshold'] = args.cyc_threshold
        info['inject_match_dropout'] = args.inject_match_dropout
        
        gen_image = pipeline(
            concept_configs=concept_configs,
            prompt=args.gen_prompt,
            template_prompt=args.template_prompt,
            template_path=args.template_path,
            output_dir=args.output_dir,
            clear_image_cache=args.clear_image_cache,
            clear_image_info_cache=args.clear_image_info_cache,
            offload=args.offload,
            info=info
        )
        
        seed += 1
    
    return models

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_prompt', type=str, default="")
    parser.add_argument('--gen_prompt', type=str, default="A can is placed on a desk, next to a laptop.")
    parser.add_argument('--template_path', type=str, default='')
    parser.add_argument('--concept_config_path', type=str, default='configs/can.json')  
    parser.add_argument('--class_names', nargs='+', default=[]) 
    parser.add_argument('--image_paths', nargs='+', default=[])
    
    parser.add_argument('--image_cache_dir', type=str, default='./image_cache')
    parser.add_argument('--image_info_cache_dir', type=str, default='./image_info_cache')
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--clear_image_cache', action='store_true')
    parser.add_argument('--clear_image_info_cache', action='store_true')
    
    parser.add_argument('--guidance', type=float, default=3)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--num_images', type=int, default=2)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    
    parser.add_argument('--start_inject_step', type=int, default=0)
    parser.add_argument('--end_inject_step', type=int, default=25)
    parser.add_argument('--start_inject_block', default=0)
    parser.add_argument('--end_inject_block', default=56)   
    parser.add_argument('--sim_threshold', type=float, default=0.2)
    parser.add_argument('--cyc_threshold', type=float, default=1.5)
    parser.add_argument('--inject_match_dropout', type=float, default=0.2)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)