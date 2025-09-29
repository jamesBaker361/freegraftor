import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
import wandb
from transformers import AutoProcessor, CLIPModel
#import ImageReward as RM
from style_rl.image_utils import concat_images_horizontally
from style_rl.eval_helpers import DinoMetric
from style_rl.prompt_list import real_test_prompt_list
from datasets import load_dataset,Dataset
from src.pipeline import FreeGraftorPipeline, ConceptConfig

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--model",type=str,default="person",help="person or object")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/mtg")
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--project_name",type=str,default="baseline")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--size",type=int,default=256)
parser.add_argument('--start_inject_step', type=int, default=0)
parser.add_argument('--end_inject_step', type=int, default=25)
parser.add_argument('--start_inject_block', default=0)
parser.add_argument('--end_inject_block', default=56)   
parser.add_argument('--sim_threshold', type=float, default=0.2)
parser.add_argument('--cyc_threshold', type=float, default=1.5)
parser.add_argument('--inject_match_dropout', type=float, default=0.2)
parser.add_argument('--guidance', type=float, default=3)
parser.add_argument('--num_steps', type=int, default=25)
parser.add_argument('--offload', action='store_true')
parser.add_argument('--start_seed', type=int, default=0)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--height', type=int, default=256)
parser.add_argument("--object",type=str,default="character")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/freegraftor")

def main(args):
    #ir_model=RM.load("ImageReward-v1.0")
        
        
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    dino_metric=DinoMetric(accelerator.device)

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    

    data=load_dataset(args.src_dataset, split="train")

    background_data=load_dataset("jlbaker361/real_test_prompt_list",split="train")
    background_dict={row["prompt"]:row["image"] for row in background_data}

    text_score_list=[]
    image_score_list=[]
    image_score_background_list=[]
    #ir_score_list=[]
    dino_score_list=[]

    output_dict={
        "image":[],
        "augmented_image":[],
        "text_score":[],
        "image_score":[],
        "dino_score":[],
        "prompt":[]
    }

    info={}

    info['seed'] = 123
    info['num_steps'] = args.num_inference_steps
    info['guidance'] = args.guidance
    info['width'] = args.width if args.width % 16 == 0 else args.width - args.width % 16
    info['height'] = args.height if args.height % 16 == 0 else args.height - args.height % 16
    
    info['start_inject_step'] = args.start_inject_step
    info['end_inject_step'] = args.end_inject_step
    info['inject_block_ids'] = list(range(args.start_inject_block, args.end_inject_block+1))
    info['sim_threshold'] = args.sim_threshold
    info['cyc_threshold'] = args.cyc_threshold
    info['inject_match_dropout'] = args.inject_match_dropout
    info["image_info"]={"latents":{}}

    pipeline=FreeGraftorPipeline(device="cuda",torch_dtype=torch_dtype)


    for k,row in enumerate(data):
        if k==args.limit:
            break
        
        prompt=real_test_prompt_list[k%len(real_test_prompt_list)]
        background_image=background_dict[prompt]
        image=row["image"]

        object=args.object
        if "object" in row:
            object=row["object"]

        concept_configs=[ConceptConfig(class_name=object,image=image)]

        augmented_image = pipeline(
            concept_configs=concept_configs,
            prompt=prompt,
            template_prompt=object,
            #template_path=args.template_path,
            #output_dir=args.output_dir,
            #clear_image_cache=args.clear_image_cache,
            #clear_image_info_cache=args.clear_image_info_cache,
            offload=args.offload,
            info=info
        )


        concat=concat_images_horizontally([image,augmented_image])

        accelerator.log({
            f"image_{k}":wandb.Image(concat)
        })
        with torch.no_grad():
            inputs = processor(
                    text=[prompt], images=[image,augmented_image,background_image], return_tensors="pt", padding=True
            )

            outputs = clip_model(**inputs)
            image_embeds=outputs.image_embeds.detach().cpu()
            text_embeds=outputs.text_embeds.detach().cpu()
            logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
        #accelerator.print("logits",logits_per_text.size())

        image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]

        [_,text_score,__]=logits_per_text
        [_,image_score,image_score_background]=image_similarities
        #ir_score=ir_model.score(prompt,augmented_image)
        dino_score=dino_metric.get_scores(image, [augmented_image])

        text_score_list.append(text_score.detach().cpu().numpy())
        image_score_list.append(image_score)
        image_score_background_list.append(image_score_background)
        #ir_score_list.append(ir_score)
        dino_score_list.append(dino_score)

        output_dict["augmented_image"].append(augmented_image)
        output_dict["image"].append(image)
        output_dict["dino_score"].append(dino_score)
        output_dict["image_score"].append(image_score)
        output_dict["text_score"].append(text_score)
        output_dict["prompt"].append(prompt)


    accelerator.log({
        "text_score_list":np.mean(text_score_list),
        "image_score_list":np.mean(image_score_list),
        "image_score_background_list":np.mean(image_score_background_list),
        #"ir_score_list":np.mean(ir_score_list),
        "dino_score_list":np.mean(dino_score_list)
    })

    Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)



if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")