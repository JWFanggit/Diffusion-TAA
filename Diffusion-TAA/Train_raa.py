import argparse
import logging
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
from mydataset  import RAADataset
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from diffusers import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange
from vivitt import Accident
from torch.utils.tensorboard import SummaryWriter
from pipline_ac import process_trible_data
logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    origin_h5_file: str,
    normal_h5_file: str,
    abnorm_h5_filr: str,
    pool: str,
    output_dir: str,
    root_path: str,
    validation_steps: int = 100,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    image_size: int=224,
    patch_size: int=16,
    num_frames: int=5,
    input_dim: int=192,
    hidden_dim: int=128,
    num_classes: int=2,
    heads: int=8,
    depth: int=3,
    in_channels:int=3,
    dim_head:int=64,
    scale_dim:int=4,
    dropout: float=0.,
    emb_dropout: float=0.,
    drop_path_rate:float=0.0,
    max_grad_norm: float = 10.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    # *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    log_dir=r"/media/ubuntu/My Passport/log/log_0.1"
    writer=SummaryWriter("log_0.1")
    # Make one log on every process with the configuration for debugging.
    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    # Load Accident models.
    model = Accident(image_size,patch_size,num_classes,num_frames,input_dim,depth,heads,pool, in_channels, dim_head, dropout,
    emb_dropout, scale_dim, drop_path_rate)
    # Freeze vae and text_encoder
    model.requires_grad_(True)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )#在config文件里面定义

    train_dataset = RAADataset(origin_h5_file,normal_h5_file,abnorm_h5_filr)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=True, drop_last=True)

    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, shuffle=False,
    #     pin_memory=True, drop_last=True)
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model,optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_depthprecision == "bf16":
        weight_dtype = torch.bfloat16
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("accident prediction")
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # if resume_from_checkpoint:
        # if resume_from_checkpoint != "latest":
        #     path = os.path.basename(resume_from_checkpoint)
        # else:
            # Get the most recent checkpoint
    # dirs = os.listdir(output_dir)
    # dirs = [d for d in dirs if d.startswith("checkpoint")]
    # dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    # path = dirs[-1]
    # accelerator.print(f"Resuming from checkpoint {path}")
    # accelerator.load_state(os.path.join(output_dir, path))
# global_step = int(path.split("-")[1])
#
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    # ce_loss=torch.nn.CrossEntropyLoss(reduction='none')

    for epoch in range(0 ,10):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                device=torch.device("cuda",3)
                ov=batch["ov"].to(weight_dtype)
                batch_size=ov.shape[0]
                nv=batch["nv"].to(weight_dtype)
                av=batch["av"].to(weight_dtype)
                label_nv = [(torch.tensor([[1, 0]])) for _ in range(batch_size)]
                label_nv=torch.stack(label_nv,dim=0).squeeze(1).to(device,dtype=torch.int32)
                label_av=[(torch.tensor([[0, 1]])) for _ in range(batch_size)]
                label_av = torch.stack(label_av, dim=0).squeeze(1).to(device,dtype=torch.int32)
                tai_nv=[(torch.tensor([-1])) for _ in range(batch_size)]
                tai_nv=torch.stack(tai_nv, dim=0).squeeze(1).to(device,dtype=torch.int32)
                tai_av=batch["tai_av"].to(device,torch.int32)
                start_id=batch["start_id"].to(device,torch.int32)
                triple_input=[ov,nv,av]
                triple_tai=[tai_nv,tai_nv,tai_av]
                triple_label=[label_nv,label_nv,label_av]
                start=[start_id,tai_av]
                # writer.add_scalar('Loss/train',loss1,epoch*len(train_dataloader)+step)
                loss= model(triple_input, triple_tai, triple_label, start)
                # writer.add_scalar('Loss/train', loss, epoch * len(train_dataloader) + global_step)
                # writer.add_scalar('Loss/train', loss, global_step)
                # while global_step <max_train_steps:
                optimizer.zero_grad()

                accelerator.backward(loss)

                writer.add_scalar('Loss/train',loss,global_step)
                # avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                # train_loss += avg_loss.item() / gradient_accumulation_steps
                # accelerator.backward(loss)
                print("steps.{}".format(loss), loss)
                if accelerator.sync_gradients:
                    accelerator.clip_gcrad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad()
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
#
#             # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
#
                if global_step % checkpointing_steps== 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        # save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
#
#                 if global_step % validation_steps == 0:
#                     if accelerator.is_main_process:
#                         for idx, batch in enumerate(val_dataloader):
        # logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        # progress_bar.set_postfix(**logs)
        # if epoch == 2:c
        if global_step >=max_train_steps:
            writer.close()
        # if global_step >= max_train_steps:
        #     writer.close()
            break

# # Create the pipeline using the trained modules and save it.
#     accelerator.wait_for_everyone()
#     if accelerator.is_main_process:
#         unet = accelerator.unwrap_model(unet)
#         accelerator.save(unet.state_dict(),save_path)
# #
# #     accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unwrap", type=str, default=None)
    parser.add_argument("--config", type=str, default="./assets/EQTAA.yaml")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
