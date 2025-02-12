import torch
import chess
from torch.utils.data import Dataset
from typing import List, Tuple

from utils import *


class ChessDataset(Dataset):
    def __init__(
        self,
        moves: List[Tuple[str, str, str, str]],
        mask_move: bool = False,
        add_legal_moves: bool = False,
    ):
        self.moves = moves  # (from, to, piece, turn, castling_rights, board_fen)
        self.mask_move = mask_move
        self.add_legal_moves = add_legal_moves  # encode legal moves into board tensor

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        _from, _to, piece, turn, castling_rights, board_fen = self.moves[idx]
        fen = board_fen + " " + turn + " " + castling_rights + " - 0 1"
        board = chess.Board(fen)
        # feature is the board tensor
        x = transform_board(
            board,
            mask_loc=_from if self.mask_move else None,
            add_legal_moves=self.add_legal_moves,
        )
        # label is one-hot encoded _from
        y = torch.zeros(64)
        # convert _to name to index, igoring promotion(last letter)
        _to = chess.parse_square(_to[:2])
        y[_to] = 1
        # if play side is black, flip the board both two ways and the label
        if board.turn == chess.BLACK:
            x = x.flip(1).flip(2)
            y = y.flip(0)

        return x, y


class ChessGroupedDataset(Dataset):
    def __init__(self, grouped_moves: List[Tuple[str, dict]]):
        self.grouped_moves = grouped_moves

    def __len__(self):
        return len(self.grouped_moves)

    def __getitem__(self, idx):
        board_fen, piece_prob = self.grouped_moves[idx]
        board = chess.Board(board_fen)
        # feature is the board tensor
        x = transform_board(board)
        # label is one-hot encoded _from
        y = piece_distribution_to_label(piece_prob)
        # if play side is black, flip the board both two ways and the label
        if board.turn == chess.BLACK:
            x = x.flip(1).flip(2)
            y = y.flip(0)

        return x, y
