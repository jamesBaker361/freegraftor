import os
import shutil
from typing import List
from dataclasses import dataclass, field
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from einops import rearrange

from flux.sampling import denoise_fireflow as denoise_fn
from flux.sampling import prepare, get_schedule, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5

from core.collage_creation import CollageCreator
from utils.load_and_save import load_image, save_image, load_image_info, save_image_info

@dataclass
class ConceptConfig:
    class_name: str
    image_path: str
    image: Image.Image
    scale: float = field(default=1.)
    x_bias: int = field(default=0)
    y_bias: int = field(default=0)
    angle: float = field(default=0.)
    flip: bool = field(default=False)
    alignment: str = field(default="center")

class FreeGraftorPipeline:
    def __init__(self, models=None, device="cuda", image_cache_dir='./image_cache', image_info_cache_dir='./image_info_cache',torch_dtype=torch.float16):
        self.device = device
        if models is None:
            models = {}
        self.load_flux(models)
        
        self.image_cache_dir = image_cache_dir
        self.image_info_cache_dir = image_info_cache_dir
        
        self.collage_creator = CollageCreator(models=models, device=self.device, image_cache_dir=image_cache_dir, image_info_cache_dir=image_info_cache_dir)

    def load_flux(self, models):
        if 't5' in models:
            self.t5 = models['t5']
        else:
            self.t5 = load_t5(os.getenv('FLUX_DEV'), self.device, max_length=512)
            models['t5'] = self.t5
        if 'clip' in models:
            self.clip = models['clip']
        else:
            self.clip = load_clip(os.getenv('FLUX_DEV'), self.device)
            models['clip'] = self.clip
        if 'flow' in models:
            self.flow = models['flow']
        else:
            self.flow = load_flow_model('flux-dev', self.device)
            models['flow'] = self.flow
        if 'ae' in models:
            self.ae = models['ae']
        else:
            self.ae = load_ae('flux-dev', self.device)
            models['ae'] = self.ae
            
    def onload(self, modules=['ae']):
        if 't5' in modules:
            self.t5 = self.t5.to(self.device)
        if 'clip' in modules:
            self.clip = self.clip.to(self.device)
        if 'ae' in modules:
            self.ae = self.ae.to(self.device)
        if 'flow' in modules:
            self.flow = self.flow.to('cpu')
        if 'collage_creator' in modules:
            self.collage_creator.onload()
        
    def offload(self, modules=['ae']):
        if 't5' in modules:
            self.t5 = self.t5.to('cpu')
        if 'clip' in modules:
            self.clip = self.clip.to('cpu')
        if 'ae' in modules:
            self.ae = self.ae.to('cpu')
        if 'flow' in modules:
            self.flow = self.flow.to('cpu')
        if 'collage_creator' in modules:
            self.collage_creator.offload()
        
        torch.cuda.empty_cache()
        
    @torch.inference_mode()
    def generate_template(self, prompt: str, info, offload=False, callback=None)->Image.Image:
        torch.manual_seed(info['seed'])
        init_noise = torch.randn((1, info['height'] * info['width'] // 256 , 64)).to(self.device).to(torch.bfloat16)
        x = torch.randn((1, 16, info['height'] // 8, info['width'] // 8)).to(self.device).to(torch.bfloat16)
        inp_gen = prepare(self.t5, self.clip, x, prompt=prompt)
        inp_gen['img'] = init_noise
        timesteps = get_schedule(info['num_steps'], inp_gen["img"].shape[1], shift=True)
        if offload:
            self.offload()
        gen_x = denoise_fn(self.flow, **inp_gen, timesteps=timesteps, guidance=info['guidance'], inverse=False, info=info, callback=callback)
        if offload:
            self.onload()
        gen_x = unpack(gen_x.float(), info['height'], info['width'])
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            x = self.ae.decode(gen_x)  
        x = x.clamp(-1, 1).float()
        x = rearrange(x[0], "c h w -> h w c") 
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img 
        
    def create_collage(self, prompt, concept_configs, info, offload=False, callback=None)->tuple[Image.Image,torch.Tensor]:
        template = self.generate_template(prompt, info, offload=offload, callback=callback)
            
        collage, collage_mask = self.collage_creator(template, concept_configs)
        
        collage_mask_array = np.array(collage_mask).astype(np.uint8) // 255
        collage_mask_tensor = torch.tensor(collage_mask_array).unsqueeze(dim=0).unsqueeze(dim=0).to(device=self.device, dtype=torch.bfloat16)
        collage_mask_tensor = transforms.Resize((collage_mask_tensor.shape[2]//16, collage_mask_tensor.shape[3]//16))(collage_mask_tensor)
        collage_mask_tensor = collage_mask_tensor.flatten()          
        
        return collage, collage_mask_tensor
    
    @torch.inference_mode()
    def invert(self, pil_image=None, prompt="", info=None, offload=False, callback=None)->dict:
        #pil_image = load_image(image_path, pil_image)
        x = self.encode_image(pil_image)
        inp_inv = prepare(self.t5, self.clip, x, prompt=prompt)
        timesteps = get_schedule(info['num_steps'], inp_inv["img"].shape[1], shift=True)
        if offload:
            self.offload()
        z = denoise_fn(self.flow, **inp_inv, timesteps=timesteps, guidance=1, inverse=True, info=info, callback=callback)
        if offload:
            self.onload()
        image_info = {
            'z': z.cpu(),
            'x': x.cpu(),
            'image': torch.tensor(np.array(pil_image)),
            'txt': inp_inv['txt'].cpu(),
            'vec': inp_inv['vec'].cpu(),
        }
        for t_str in info['image_info']['latents']:
            assert len(info['image_info']['latents'][t_str]['ref_imgs']) == 1
            image_info[t_str] = info['image_info']['latents'][t_str]['ref_imgs'][-1]
            info['image_info']['latents'][t_str]['ref_imgs'] = []
        return image_info
            
    @torch.inference_mode()
    def encode_image(self, image)->torch.Tensor:
        image_array = np.array(image)
        image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 127.5 - 1
        image = image.unsqueeze(0) 
        image = image.to(self.device)
        image = self.ae.encode(image.to()).to(torch.bfloat16)
        return image    
    

    
    @torch.inference_mode()
    def invert_and_record(self, pil_image:Image.Image, info=None, offload=False, callback=None)->dict:
        '''if image_path is None:
            image_path = save_image(pil_image)'''
        #image_info = load_image_info(image_path, self.image_info_cache_dir)
        #if not (image_info and 'z' in image_info):
        image_info = self.invert(pil_image=pil_image, prompt="", info=info, offload=offload, callback=callback)
            #save_image_info(image_info, image_path, self.image_info_cache_dir)
        return image_info
    
    @torch.inference_mode()
    def final_generation(self, prompt, all_image_info, info, offload=False, callback=None)->Image.Image:
        for image_info in all_image_info:
            for key, value in image_info.items():
                if 't_' in key:
                    t_str = key
                    if not isinstance(info['image_info']['latents'][t_str]['ref_imgs'], list):
                        info['image_info']['latents'][t_str]['ref_imgs'] = []
                    info['image_info']['latents'][t_str]['ref_imgs'].append(value)
        init_noise = image_info['z'].to(self.device)
        x = torch.randn((1, 16, info['height'] // 8, info['width'] // 8)).to(self.device).to(torch.bfloat16)
        inp_gen = prepare(self.t5, self.clip, x, prompt=prompt)
        
        inp_gen['ref_vecs'] = [all_image_info[idx]['vec'].cuda() for idx in range(len(all_image_info))]
        inp_gen['ref_txts'] = [all_image_info[idx]['txt'].cuda() for idx in range(len(all_image_info))]
        inp_gen['img'] = init_noise.cuda()
        inp_gen['ref_imgs'] = [all_image_info[idx]['z'].cuda() for idx in range(len(all_image_info))]
        inp_gen['ref_masks'] = [all_image_info[idx]['mask'].cuda() for idx in range(len(all_image_info))]
        
        timesteps = get_schedule(info['num_steps'], inp_gen["img"].shape[1], shift=True)
        if offload:
            self.offload()
        gen_x, ref_x_recons = denoise_fn(self.flow, **inp_gen, timesteps=timesteps, guidance=info['guidance'], inverse=False, info=info, callback=callback)
        if offload:
            self.onload()
        gen_x = unpack(gen_x.float(), info['width'], info['height'])
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            x = self.ae.decode(gen_x)
        x = x.clamp(-1, 1).float()
        x = rearrange(x[0], "c h w -> h w c")
        gen_image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            
        return gen_image
    
    def __call__(
        self,
        concept_configs: List[ConceptConfig],
        prompt: str,
        template_prompt: str = None,
        template_path: str = None,
        #output_dir: str = 'inference_results', 
        clear_image_cache: bool = False,
        clear_image_info_cache: bool = False,
        offload: bool = False,
        info = None,
        callback=None
    ):
        #Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        if not template_prompt:
            template_prompt = prompt
        
        collage, collage_mask_tensor = self.create_collage(template_prompt, concept_configs, info, template_path, offload=offload, callback=callback)
        #collage_path = save_image(collage, self.image_cache_dir, "collage")
        
        collage_info = self.invert_and_record(collage, info=info, offload=offload, callback=lambda x,y: callback(x+info['num_steps'], y) if callback else None)
        collage_info['mask'] = collage_mask_tensor.cpu()
        
        gen_image = self.final_generation(prompt, [collage_info], info, offload=offload, callback=lambda x,y: callback(x+info['num_steps']*2, y) if callback else None)
        
        seed = info['seed']
        #save_image(gen_image, output_dir, f"seed{seed}")
        
        if clear_image_cache:
            shutil.rmtree(self.image_cache_dir)
        if clear_image_info_cache:
            shutil.rmtree(self.image_info_cache_dir)
            
        return gen_image