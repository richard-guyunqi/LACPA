import torch
from chess_utils import ChessData, ResidualNet, loss_fn
from tqdm import tqdm
import wandb
import json
from datetime import datetime
import os

device = 'cuda:0'

now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%Hh%Mm%Ss")

args = {
    # Hyper parameters
    'input_dim': 773,
    'output_dim': 1,
    'hidden_dim': 2048,

    # Model parameters
    'save_ckpt_path': '/remote_training/richard/a/chess/ckpt',
    'load_ckpt_path': None,

    # Training Parameters
    'lr': 1e-3,
    'batch_size': 512,
    'epochs': 300,
    'date_time': date_time,
    
}

# Model Parameters & Initialization
model = ResidualNet(args['input_dim'], args['hidden_dim'], args['output_dim'])
model = model.to(device)
if args['load_ckpt_path'] is not None:
    model = model.load_state_dict(torch.load(args['load_ckpt_path']))

# Data Initialization
train_dataset = ChessData('/remote_training/richard/a/chess/data/exp.pth', )
val_dataset = ChessData('/remote_training/richard/a/chess/data/val.pth')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=16)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=16)

# Training metadata
steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * args['epochs']
train_metadata = {
    'train_loss':[],
    'train_grad':[],
    'val_loss': [],
}

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['lr'], total_steps=total_steps)

# Wandb initialization
wandb_args = dict(
    entity="yrichard",
    project='chess',
)
wandb.init(**wandb_args)
wandb.config.update(args)   # Log all hyperparameters from args


# Training loop
for epoch in range(args['epochs']):
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc='Train')

    for batch_idx, (inputs, labels) in pbar:
        step_idx = batch_idx + steps_per_epoch * epoch
    
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        train_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
        optimizer.step()
        scheduler.step()

        train_metadata['train_loss'].append(loss.item())
        train_metadata['train_grad'].append(train_grad.item())

        pbar.set_postfix(loss=loss.item(), grad=train_grad.item())

        wandb.log(
            {
                'train_loss': train_metadata['train_loss'][-1],
                'train_grad': train_metadata['train_grad'][-1],
            },
            step=step_idx,
            commit=True
        )

    
    # Eval (per epoch)
    model.eval()
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=True, desc='Val')
    with torch.no_grad():
        val_losses = []
        for batch_idx, (inputs, labels) in pbar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            loss = loss_fn(preds, labels)
            val_losses.append(loss.item())

        train_metadata['val_loss'].append(sum(val_losses) / len(val_losses))

        wandb.log(
            {
                'val_loss': train_metadata['val_loss'][-1],
            },
            step=step_idx + 1,
            commit=True
        )        
    model.train()
    
    # Save ckpt
    ckpt_dir = os.path.join(args['save_ckpt_path'], f'{date_time}')
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f'model_{epoch}.pth')
    torch.save(model.state_dict(), ckpt_path)

















