import torch
from chess_utils import ChessData_conv, ResidualNet, loss_fn, ConvNet
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
    'input_channels': 11,
    'hidden_dim': 256,
    'output_dim': 1,

    # Model parameters
    'save_ckpt_path': '/remote_training/richard/a/chess/ckpt',
    'load_ckpt_path': None,

    # Data parameters
    'train_data_path': '/remote_training/richard/a/chess/data/train_conv_v4.pth',
    'val_data_path': '/remote_training/richard/a/chess/data/val_conv_v4.pth',

    # Training Parameters
    'lr': 5e-5,
    'batch_size': 512,
    'epochs': 300,
    'date_time': date_time,
    
}

# Model Parameters & Initialization
model = ConvNet(args['input_channels'], args['hidden_dim'], args['output_dim'])
model = model.to(device)
if args['load_ckpt_path'] is not None:
    model = model.load_state_dict(torch.load(args['load_ckpt_path']))

# Data Initialization
train_dataset = ChessData_conv(args['train_data_path'])
val_dataset = ChessData_conv(args['val_data_path'])
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
    project='chess_conv',
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

        value_preds, _ = model(inputs)
        # print(value_preds.shape)
        # print(labels.shape)
        loss = loss_fn(value_preds, labels.unsqueeze(1))

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

            value_preds, _ = model(inputs)
            loss = loss_fn(value_preds, labels)
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

















