import torch
import chess
from torch import nn
import chess.svg
from IPython.display import display, SVG




class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dim, kernel_size=3, dropout=.2):
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            *[ConvNetBlock(hidden_dim, kernel_size, dropout) for i in range(39)],
        )
        
        self.policy_decoder = None
        self.value_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1, padding=1),
            Flatten_board(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        embeddings = self.encoder(x)
        return self.value_decoder(embeddings), None

class ConvNetBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=.2):
        super(ConvNetBlock, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=1),
            nn.BatchNorm2d(hidden_dim),
            # nn.Dropout(dropout),
        )
        self.rectifier = nn.ReLU()

    def forward(self, x):
        return self.rectifier(x + self.first(x)) 

class Flatten_board(nn.Module):
    def __init__(self):
        super(Flatten_board, self).__init__()
    
    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, -1)


class ResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(ResidualNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ResidualFCBlock(hidden_dim, dropout=dropout),
            ResidualFCBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return torch.nn.functional.tanh(self.model(x))



class ResidualFCBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(ResidualFCBlock, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x):
        return x + self.fc_block(x)
    

def loss_fn(preds, labels):
    return torch.nn.functional.mse_loss(preds, labels)

class ChessData_conv(torch.utils.data.Dataset):
    def __init__(self, total_tensors_path, transform=None):
        self.tensors = torch.load(total_tensors_path)

        self.states = self.tensors[:, :-1, :, :]
        self.labels = self.tensors[:, -1, 0, 0]

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]
    
    def __len__(self):
        return self.states.shape[0]


class ChessData(torch.utils.data.Dataset):
    def __init__(self, total_tensors_path, transform=None):
        self.tensors = torch.load(total_tensors_path)

        self.states = self.tensors[:, :-1]
        self.labels = self.tensors[:, -1]

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]
    
    def __len__(self):
        return self.states.shape[0]

import torch
import chess

def board_to_two_tensors(board, device='cuda:0'):
    piece_order = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]

    def board_to_single_tensor(board, flip=False):
        tensor = torch.zeros((6, 8, 8), dtype=torch.float32)

        for i, piece_type in enumerate(piece_order):
            # White pieces
            for square in board.pieces(piece_type, chess.WHITE):
                row, col = divmod(square, 8)
                if flip:
                    tensor[i, 7 - row, 7 - col] = -1
                else:
                    tensor[i, row, col] = 1

            # Black pieces
            for square in board.pieces(piece_type, chess.BLACK):
                row, col = divmod(square, 8)
                if flip:
                    tensor[i, 7 - row, 7 - col] = 1
                else:
                    tensor[i, row, col] = -1

        castling_tensor = torch.tensor([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=torch.float32)

        turn_tensor = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
        if flip:
            castling_tensor = castling_tensor[[2, 3, 0, 1]]
            turn_tensor = 1.0 - turn_tensor

        # print(tensor.shape)
        # print(castling_tensor.shape)        # (4, )
        # print(turn_tensor.shape)        # (1, )

        full_tensor = torch.cat([tensor, castling_tensor.view(4, 1, 1).expand(4, 8, 8), turn_tensor.view(1, 1, 1).expand(1, 8, 8)], dim=0)

        return full_tensor

    original_tensor = board_to_single_tensor(board, flip=False)
    flipped_tensor = board_to_single_tensor(board, flip=True)

    return original_tensor.to(device), flipped_tensor.to(device)

def tensor_to_board(tensor):
    piece_order = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]

    board = chess.Board(None)  # Empty board
    piece_tensor = tensor[:6]
    castling_tensor = tensor[6:10][:, 0, 0]
    turn_tensor = tensor[10, 0, 0]

    # print(piece_tensor.shape, castling_tensor.shape, turn_tensor)

    for i, piece_type in enumerate(piece_order):
        for row in range(8):
            for col in range(8):
                if piece_tensor[i, 7 - row, col] == 1:
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
                if piece_tensor[i, 7 - row, col] == -1:
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))

    board.turn = chess.WHITE if turn_tensor == 1 else chess.BLACK

    # Set castling rights
    if castling_tensor[0]:
        board.castling_rights |= chess.BB_H1
    if castling_tensor[1]:
        board.castling_rights |= chess.BB_A1
    if castling_tensor[2]:
        board.castling_rights |= chess.BB_H8
    if castling_tensor[3]:
        board.castling_rights |= chess.BB_A8

    return board

# def board_to_two_tensors(board, device='cuda:0'):
#     piece_order = [
#         chess.PAWN, chess.KNIGHT, chess.BISHOP,
#         chess.ROOK, chess.QUEEN, chess.KING
#     ]

#     def board_to_single_tensor(board, flip=False):
#         tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

#         for i, piece_type in enumerate(piece_order):
#             # White pieces
#             for square in board.pieces(piece_type, chess.WHITE):
#                 row, col = divmod(square, 8)
#                 if flip:
#                     tensor[i + 6, 7 - row, 7 - col] = 1
#                 else:
#                     tensor[i, row, col] = 1

#             # Black pieces
#             for square in board.pieces(piece_type, chess.BLACK):
#                 row, col = divmod(square, 8)
#                 if flip:
#                     tensor[i, 7 - row, 7 - col] = 1
#                 else:
#                     tensor[i + 6, row, col] = 1

#         castling_tensor = torch.tensor([
#             board.has_kingside_castling_rights(chess.WHITE),
#             board.has_queenside_castling_rights(chess.WHITE),
#             board.has_kingside_castling_rights(chess.BLACK),
#             board.has_queenside_castling_rights(chess.BLACK)
#         ], dtype=torch.float32)

#         turn_tensor = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0], dtype=torch.float32)
#         if flip:
#             castling_tensor = castling_tensor[[2, 3, 0, 1]]
#             turn_tensor = 1.0 - turn_tensor

#         # print(tensor.shape)
#         # print(castling_tensor.shape)        # (4, )
#         # print(turn_tensor.shape)        # (1, )

#         full_tensor = torch.cat([tensor, castling_tensor.view(4, 1, 1).expand(4, 8, 8), turn_tensor.view(1, 1, 1).expand(1, 8, 8)], dim=0)

#         return full_tensor

#     original_tensor = board_to_single_tensor(board, flip=False)
#     flipped_tensor = board_to_single_tensor(board, flip=True)

#     return original_tensor.to(device), flipped_tensor.to(device)

# def tensor_to_board(tensor):
#     piece_order = [
#         chess.PAWN, chess.KNIGHT, chess.BISHOP,
#         chess.ROOK, chess.QUEEN, chess.KING
#     ]

#     board = chess.Board(None)  # Empty board
#     piece_tensor = tensor[:12]
#     castling_tensor = tensor[12:16][:, 0, 0]
#     turn_tensor = tensor[16, 0, 0]

#     # print(piece_tensor.shape, castling_tensor.shape, turn_tensor)

#     for i, piece_type in enumerate(piece_order):
#         for row in range(8):
#             for col in range(8):
#                 if piece_tensor[i, 7 - row, col]:
#                     square = chess.square(col, 7 - row)
#                     board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
#                 if piece_tensor[i + 6, 7 - row, col]:
#                     square = chess.square(col, 7 - row)
#                     board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))

#     board.turn = chess.WHITE if turn_tensor == 1 else chess.BLACK

#     # Set castling rights
#     if castling_tensor[0]:
#         board.castling_rights |= chess.BB_H1
#     if castling_tensor[1]:
#         board.castling_rights |= chess.BB_A1
#     if castling_tensor[2]:
#         board.castling_rights |= chess.BB_H8
#     if castling_tensor[3]:
#         board.castling_rights |= chess.BB_A8

#     return board

# def moves_to_tensor(df_row, gamma=0.98):
#     moves = df_row['moves'].split()
#     winner = df_row['winner']

#     board = chess.Board()
#     tensors = []

#     result_value = 1.0 if winner == 'white' else 0.0

#     board_tensor_white, board_tensor_black = board_to_two_tensors(board)

#     total_moves = len(moves)

#     white_value = torch.tensor([result_value * (gamma ** (total_moves))])
#     black_value = torch.tensor([-result_value * (gamma ** (total_moves))])

#     full_tensor_white = torch.cat([board_tensor_white, white_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)
#     full_tensor_black = torch.cat([board_tensor_black, black_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)
    
#     tensors.append(full_tensor_white)
#     tensors.append(full_tensor_black)


#     for i in range(len(moves)):
#         move = moves[i]
#         board.push_san(move)
#         board_tensor_white, board_tensor_black = board_to_two_tensors(board)

#         white_value = torch.tensor([result_value * (gamma ** (total_moves - i - 1))])
#         black_value = torch.tensor([-result_value * (gamma ** (total_moves - i - 1))])

#         full_tensor_white = torch.cat([board_tensor_white, white_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)
#         full_tensor_black = torch.cat([board_tensor_black, black_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)

#         tensors.append(full_tensor_white)
#         tensors.append(full_tensor_black)

#     return torch.stack(tensors)

def moves_to_tensor(df_row, gamma=0.98, alpha=1):
    moves = df_row['moves'].split()
    winner = df_row['winner']

    board = chess.Board()
    tensors = []

    result_value = 1.0 if winner == 'white' else 0.0

    board_tensor_white, board_tensor_black = board_to_two_tensors(board)

    total_moves = len(moves)
    cur_advantage = calculate_diff(board_tensor_white, alpha)

    white_value = torch.tensor([result_value * (gamma ** (total_moves)) + cur_advantage])
    black_value = torch.tensor([-result_value * (gamma ** (total_moves)) - cur_advantage])

    full_tensor_white = torch.cat([board_tensor_white, white_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)
    full_tensor_black = torch.cat([board_tensor_black, black_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)
    
    tensors.append(full_tensor_white)
    tensors.append(full_tensor_black)

    for i in range(len(moves)):
        move = moves[i]
        board.push_san(move)
        board_tensor_white, board_tensor_black = board_to_two_tensors(board)

        cur_advantage = calculate_diff(board_tensor_white, alpha)
        white_value = torch.tensor([result_value * (gamma ** (total_moves - i - 1) + cur_advantage)])
        black_value = torch.tensor([-result_value * (gamma ** (total_moves - i - 1) - cur_advantage)])

        full_tensor_white = torch.cat([board_tensor_white, white_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)
        full_tensor_black = torch.cat([board_tensor_black, black_value.view(1, 1, 1).expand(1, 8, 8)], dim=0)

        tensors.append(full_tensor_white)
        tensors.append(full_tensor_black)

    return torch.stack(tensors)


def display_board(board):
    display(SVG(chess.svg.board(board=board, size=400)))
    return
