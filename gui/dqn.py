import torch
from chess_utils import ChessData_conv, ConvNet, board_to_two_tensors, tensor_to_board, moves_to_tensor, display_board
from tqdm import tqdm
import json
import os
import random
import wandb
import chess
import chess.svg
from IPython.display import display, SVG
from datetime import datetime
from tensorflow.keras.models import load_model
from players import HumanPlayer, AIPlayer
from draw import draw_background, draw_pieces

from_model = load_model('models/1200-elo/from.h5', compile=False)
to_model = load_model('models/1200-elo/to.h5', compile=False)

now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%Hh%Mm%Ss")

def calculate_diff(tensor_board, scalar=1):
    sums = torch.sum(tensor_board, dim=(1, 2))
    score = 0.025 * (sums[0]) + 0.075 * (sums[1]) + 0.0875 * (sums[2]) + 0.125 * (sums[3]) + 0.225 * (sums[4])
    return scalar * score

def epsilon_select(move_scores, epsilon):
    """Select a move using epsilon-greedy strategy."""
    if random.random() < epsilon:
        return random.choice(list(move_scores.keys()))  # Random move
    else:
        return max(move_scores.items(), key=lambda x: x[1])[0]  # Best move

def opponent_play(model, gamma=0.98, epsilon=0.3, device="cuda:0"):
    # Forward pass -- generate all the moves
    board = chess.Board()
    step_count = 0
    moves_list = []
    data_list = []      # Format: (state_t, state_{t+1}, r_t)
    
    model_turn = random.choice(['black', 'white'])
    move_order = []
    if model_turn == 'white':
        ai = AIPlayer(colour='black', from_model=from_model, to_model=to_model)
        move_order = ['model', 'ai']
        print('model white, ai black')
    else:
        ai = AIPlayer(colour='white', from_model=from_model, to_model=to_model)
        move_order = ['ai', 'model']
        print('ai white, model black')

    tensor_me_cur, _ = board_to_two_tensors(board, device)
    order_index = 0
    
    # while not board.is_game_over() and step_count < 50:  
    while not board.is_game_over():  
        turn = move_order[order_index]
        order_index = 1 - order_index
        step_count += 1  

        if turn == 'model':
            legal_moves = list(board.legal_moves)
            move_scores = {legal_move: None for legal_move in legal_moves}
            tensor_me_cur, _ = board_to_two_tensors(board, device)

            for legal_move in legal_moves:
                board.push(legal_move)

                if len(legal_moves) > 1 and board.can_claim_threefold_repetition():
                    board.pop()
                    del move_scores[legal_move]
                    continue

                if not board.turn:      # white moving
                    tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device)   
                else:       # black moving
                    tensor_oppo_after, tensor_me_after = board_to_two_tensors(board, device)   

                # Compute value for the move
                value_me_after, _ = model(tensor_me_after.unsqueeze(0))
                value_oppo_after, _ = model(tensor_oppo_after.unsqueeze(0))
                # move_scores[legal_move] = (value_me_after - value_oppo_after).item()    
                move_scores[legal_move] = (value_me_after - value_oppo_after + calculate_diff(tensor_me_after)).item()    

                board.pop()


            if len(move_scores) == 0:
                return None, None, None

            # move = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
            move = epsilon_select(move_scores, epsilon)
            # print(move)
            board.push(move)
            moves_list.append(move)

            # Register the data
            if not board.turn:      # white moving
                tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device) 
            else:       # black moving
                tensor_oppo_after, tensor_me_after = board_to_two_tensors(board, device)
    
            data_list.append([tensor_me_cur, tensor_me_after, tensor_oppo_after, 0])

        else:
            move = ai.move(board=board, human_white=(model_turn=='white'))
            moves_list.append(move)

        if step_count % 50 == 0:
            print(step_count)

    # Assign rewards at the end of the game
    result = board.result()  # e.g., "1-0", "0-1", "1/2-1/2"
    
    if result == "1-0":
        final_reward = 1  # White wins
    elif result == "0-1":
        final_reward = -1  # Black wins
    else:
        final_reward = 0  # Draw

    # Apply gamma-discounted rewards
    discounted_reward = final_reward
    for i in range(len(data_list) - 1, -1, -1):  # Iterate from last to first move
        data_list[i][3] += torch.tensor([discounted_reward])
        discounted_reward *= gamma  # Decay the reward

    return data_list, result, moves_list



def self_play(model, gamma=0.98, epsilon=0.3, device="cuda:0"):
    # Forward pass -- generate all the moves
    board = chess.Board()
    step_count = 0
    moves_list = []
    data_list = []      # Format: (state_t, state_{t+1}, r_t)
    
    tensor_me_cur, _ = board_to_two_tensors(board, device)

    # while not board.is_game_over() and step_count < 50:  
    while not board.is_game_over():  
        step_count += 1  
        legal_moves = list(board.legal_moves)
        move_scores = {legal_move: None for legal_move in legal_moves}
        
        for legal_move in legal_moves:
            board.push(legal_move)

            if len(legal_moves) > 1 and board.can_claim_threefold_repetition():
                board.pop()
                del move_scores[legal_move]
                continue

            if not board.turn:      # white moving
                tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device)   
            else:       # black moving
                tensor_oppo_after, tensor_me_after = board_to_two_tensors(board, device)   

            # Compute value for the move
            value_me_after, _ = model(tensor_me_after.unsqueeze(0))
            value_oppo_after, _ = model(tensor_oppo_after.unsqueeze(0))
            # move_scores[legal_move] = (value_me_after - value_oppo_after).item()    
            move_scores[legal_move] = (value_me_after - value_oppo_after + calculate_diff(tensor_me_after)).item()    

            board.pop()

        if len(move_scores) == 0:
            return None, None, None

        # move = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        move = epsilon_select(move_scores, epsilon)
        # print(move)
        board.push(move)
        moves_list.append(move)

        # Register the data
        if not board.turn:      # white moving
            tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device) 
        else:       # black moving
            tensor_oppo_after, tensor_me_after = board_to_two_tensors(board, device)
  
        data_list.append([tensor_me_cur, tensor_me_after, tensor_oppo_after, 0])
        tensor_me_cur = tensor_oppo_after

        if step_count % 50 == 0:
            print(step_count)

    # Assign rewards at the end of the game
    result = board.result()  # e.g., "1-0", "0-1", "1/2-1/2"
    
    if result == "1-0":
        final_reward = 1  # White wins
    elif result == "0-1":
        final_reward = -1  # Black wins
    else:
        final_reward = 0  # Draw

    # Apply gamma-discounted rewards
    discounted_reward = final_reward
    for i in range(len(data_list) - 1, -1, -1):  # Iterate from last to first move
        data_list[i][3] += torch.tensor([discounted_reward])
        discounted_reward *= gamma  # Decay the reward

    return data_list, result, moves_list


args = {
    'epochs': 300,
    'plays_per_epoch': 10,
    'batch_size': 256,
    'minibatch_size': 32,
    'lr': 1e-5,
    "save_ckpt_path": '/remote_training/richard/a/chess/stage2_ckpts',
}

device = 'cuda:0'

model = ConvNet(11, 256, 1)
model = model.to(device)
# model.load_state_dict(torch.load('/remote_training/richard/a/chess/ckpt/v3_good/model_15.pth'))
model.load_state_dict(torch.load('/remote_training/richard/a/chess/ckpt/v4_good/model_9.pth'))

wandb_args = dict(
    entity="yrichard",
    project='chess_stage2',
)
wandb.init(**wandb_args)
wandb.config.update(args)   # Log all hyperparameters from args


optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = lr)

for epoch in range(args['epochs']):
    # Create a replay buffer through self-playing
    buffer = []
    
    for play in tqdm(range(args['plays_per_epoch']), desc='self-play'):
        data_list = None
        while data_list is None:
            data_list, result, moves_list = opponent_play(model, epsilon=.2)
            print("Self play failed.")
        buffer += data_list

    # Sample some minibatches from the replay buffer
    pbar = tqdm(range(args['batch_size']), desc='training')
    losses = []
    for batch in pbar:
        outputs = random.sample(buffer, args['minibatch_size']) # buffer[random.randint(0, len(buffer) - 1)]
        tensor_me_cur_list = []
        tensor_me_after_list = []
        tensor_oppo_after_list = []
        reward_list = []

        for tensor_me_cur, tensor_me_after, tensor_oppo_after, reward in outputs:
            tensor_me_cur_list.append(tensor_me_cur)
            tensor_me_after_list.append(tensor_me_after)
            tensor_oppo_after_list.append(tensor_oppo_after)
            reward_list.append(reward)

        tensor_me_cur = torch.stack(tensor_me_cur_list).to(device)
        tensor_me_after = torch.stack(tensor_me_after_list).to(device)
        tensor_oppo_after = torch.stack(tensor_oppo_after_list).to(device)
        reward = torch.stack(reward_list).to(device)

        value_me_cur, _ = model(tensor_me_cur)
        value_me_after, _ = model(tensor_me_after)
        value_oppo_after, _ = model(tensor_oppo_after)

        label = reward.unsqueeze(1) + value_me_after - value_oppo_after
        
        loss = torch.nn.functional.mse_loss(label, value_me_cur)
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wandb.log(
        {
            'train_loss': sum(losses) / len(losses),
        },
        step=epoch,
        commit=True
    )

    # Save ckpt
    ckpt_dir = os.path.join(args['save_ckpt_path'], f'{date_time}')
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f'model_{epoch}.pth')
    torch.save(model.state_dict(), ckpt_path)
    