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
ai = AIPlayer(colour='white', from_model=from_model, to_model=to_model)


def ai_on_bench(before_set, after_set):
    accuracy_list = []

    # Run our model on both benchs
    for i in tqdm(range(len(before_set))):
    # for i in tqdm(range(200)):
        # Restore the board
        board = tensor_to_board(before_set[i])
        
        try:
            move = ai.move(board=board, human_white=False)
        except:
            continue

        tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device) 
        accuracy_list.append(torch.equal(after_set[i], tensor_me_after))

    return sum(accuracy_list) / len(accuracy_list)

def test_on_bench(before_set, after_set):
    accuracy_list = []

    # Run our model on both benchs
    for i in tqdm(range(len(before_set))):
    # for i in tqdm(range(200)):
        # Restore the board
        board = tensor_to_board(before_set[i])
        
        # Get the legal moves
        legal_moves = list(board.legal_moves)
        move_scores = {legal_move: None for legal_move in legal_moves}

        for legal_move in legal_moves:
            board.push(legal_move)

            tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device)   

            # Compute value for the move
            value_me_after, _ = model(tensor_me_after.unsqueeze(0))
            value_oppo_after, _ = model(tensor_oppo_after.unsqueeze(0))
            # move_scores[legal_move] = (value_me_after - value_oppo_after).item()    
            move_scores[legal_move] = (value_me_after - value_oppo_after + calculate_diff(tensor_me_after)).item()    

            board.pop()

        # move = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        move = epsilon_select(move_scores, epsilon) 

        board.push(move)
        tensor_me_after, tensor_oppo_after = board_to_two_tensors(board, device) 
        accuracy_list.append(torch.equal(after_set[i], tensor_me_after))

    return sum(accuracy_list) / len(accuracy_list)

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

device = "cuda:0"
epsilon = 0

model = ConvNet(11, 256, 1)
model = model.to(device)
# model.load_state_dict(torch.load('/remote_training/richard/a/chess/ckpt/v3_good/model_15.pth'))
# model.load_state_dict(torch.load('/remote_training/richard/a/chess/stage2_ckpts/v4_good/model_9.pth'))
model.load_state_dict(torch.load('/remote_training/richard/a/chess/stage2_ckpts/2025_03_13_06h22m24s/model_55.pth'))

# Load data for two benchmarks
capture_before_set = torch.load('/remote_training/richard/a/chess/data/capture_before.pth')
capture_after_set = torch.load('/remote_training/richard/a/chess/data/capture_after.pth')
endgame_before_set = torch.load('/remote_training/richard/a/chess/data/endgame_before.pth')
endgame_after_set = torch.load('/remote_training/richard/a/chess/data/endgame_after.pth')

capture_accuracy_list = []


model_capture_accuracy = test_on_bench(capture_before_set, capture_after_set)
model_endgame_accuracy = test_on_bench(endgame_before_set, endgame_after_set)

ai_capture_accuracy = ai_on_bench(capture_before_set, capture_after_set)
ai_endgame_accuracy = ai_on_bench(endgame_before_set, endgame_after_set)

print(f'Model -- capture_accuracy: {model_capture_accuracy}, endgame_accuracy: {model_endgame_accuracy}.')
print(f'AI -- capture_accuracy: {ai_capture_accuracy}, endgame_accuracy: {ai_endgame_accuracy}.')







