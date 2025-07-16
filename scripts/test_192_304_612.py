import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import datetime

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    logger.log("creating samples...")
    all_images = []

    device = th.device("cuda" if th.cuda.is_available() else "cpu")


    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()} 

        shape = (args.batch_size, 1, model_kwargs['low_res'].shape[2], model_kwargs['low_res'].shape[3], model_kwargs['low_res'].shape[4])
        
        # A fixed random seed was used to maintain consistent initial noise across different patches from the same dataset
        if device == "cuda":
            th.cuda.manual_seed_all(10)
        else:
            th.manual_seed(10)
        noise = th.randn(*shape, device=device)

        sample = diffusion.p_sample_loop(
            model,
            shape, 
            noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = sample.permute(0, 1, 3, 4, 2)
        sample = sample.contiguous() 

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample) 
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples\n")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples] 
    arr = arr.reshape((arr.shape[0],arr.shape[2],arr.shape[3],arr.shape[4])) 

    # Reassemble 7 patches back to a volume with shape [192, 304, 672] before overlap processing
    arr_result = np.zeros((192,304,672))     
    index = 0
    for i in range(7):
        arr_result[:, :, i*96:(i+1)*96] = arr[index,:,:,:]
        index += 1
    
    # Process overlaps and restore to original dimensions [192, 304, 612]
    final_result = reverse_2_original_dimension(arr_result)

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in final_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{datetime.datetime.now().strftime('%H%M%S%f')}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, final_result)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        low_pet = obj["arr_0"][0] 

        # Divide [192, 304, 612] into 7 overlapping patches of size [192, 304, 96].
        # Each patch overlaps with the next by 10 pixels.
        image_arr = np.zeros((7,192,304,96))
        index = 0
        for i in (0, 86, 172, 258, 344, 430, 516):
            image_arr[index,:,:,:] = low_pet[:, :, i:i+96]
            index += 1
    
    # Preprocessing: convert to SUV (assumed already done), clip to 16, then normalize to [0, 1]
    image_arr[image_arr>16] = 16
    image_arr = image_arr/16

    rank = dist.get_rank()
    logger.log('rank:{%d}' % (rank))
    
    num_ranks = dist.get_world_size()
    logger.log('num_ranks:{%d}' % (num_ranks))

    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            logger.log('rank:{%d}, i:{%d}, buffer_len:{%d}' % (rank, i, len(buffer)))

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch.unsqueeze(0) 
                batch = batch.permute(0, 1, 4, 2, 3)
                logger.log('batch_shape:{%d,%d,%d,%d,%d}' % (batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]))
                res = dict(low_res=batch)
                yield res
                buffer, label_buffer = [], []


def reverse_2_original_dimension(quadra_3D_672):
    # Merge 7 overlapping patches to reconstruct original volume [192, 304, 612].
    # Weighted average blending is used in overlapping areas to ensure smooth transitions.
    quadra_3D_612 = np.zeros((192, 304, 612))
    index_a = [0, 96, 182, 268, 354, 440, 526] # Start indices for final patch central parts placements
    index_b = [86, 172, 258, 344, 430, 516] # Start indices of 10-pixel overlaps
    
    # First non-overlapping segment
    quadra_3D_612[:,:, 0:86] = quadra_3D_672[:,:,0:86]
    for i in range(0,7):
        if i > 0:
            # Place the central non-overlapping parts of each patch
            quadra_3D_612[:,:, index_a[i]:index_a[i]+76] = quadra_3D_672[:,:,96*i+10:96*i+86]

        # Blend 10-pixel overlapping regions between adjacent patches using weighted averaging
        if i <= 5:
            for j in range(0,10):
                quadra_3D_612[:,:, index_b[i]+j:index_b[i]+j+1] = quadra_3D_672[:,:, 96*i+86+j:96*i+86+j+1]*(1-j/9) + quadra_3D_672[:,:, 96*(i+1)+j:96*(i+1)+j+1]*(j/9)

    return quadra_3D_612


def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
