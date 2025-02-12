import chess
import chess.pgn
from chess.pgn import BaseVisitor
import io
import torch
from typing import List, Tuple
import pickle
from time import time


class Visitor(BaseVisitor):
    def __init__(self):
        self.comments = []
        self.moves = []

    def visit_comment(self, comment: str) -> None:
        self.comments.append(comment)

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.moves.append(move)

    def result(self):
        return self.moves, self.comments


def read_pgn(pgn):
    pgn = io.StringIO(pgn)
    moves, _ = chess.pgn.read_game(pgn, Visitor=Visitor)
    return moves


def read_data_sample(path: str) -> List[Tuple[str, str, str]]:
    # read list of tuples from pickle file
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


# encode board into 6 channel tensor
# each channel represents a different type of piece
# 0: empty square
# 1: white pieces, 2: black pieces
# 1st channel: pawns, 2nd channel: knights, 3rd channel: bishops
# 4th channel: rooks, 5th channel: queens, 6th channel: kings
def transform_board(
    board: chess.Board, mask_loc: str | int = None, add_legal_moves: bool = False
) -> torch.Tensor:
    piece_map = {
        "p": 1,
        "n": 2,
        "b": 3,
        "r": 4,
        "q": 5,
        "k": 6,
        "P": 7,
        "N": 8,
        "B": 9,
        "R": 10,
        "Q": 11,
        "K": 12,
    }
    board_tensor = torch.zeros(12, 8, 8)
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                piece_symbol = piece.symbol()
                board_tensor[piece_map[piece_symbol] - 1, i, j] = 1

    if add_legal_moves:  # add new channels of legal moves to the board tensor
        legal_moves = board.legal_moves
        legal_move_channel = torch.zeros(1, 8, 8)
        for move in legal_moves:
            to_square = move.to_square
            legal_move_channel[0, to_square % 8, to_square // 8] = 1
            # check if the move is a capture
            if board.is_capture(move):
                legal_move_channel[0, to_square % 8, to_square // 8] *= 2
        board_tensor = torch.cat((board_tensor, legal_move_channel), dim=0)

    if mask_loc is not None:  # add new channels of mask to the board tensor
        mask_tensor = torch.zeros(1, 8, 8)
        # convert mask_loc san to square index
        assert (
            mask_loc is not None or type(mask_loc) is int or type(mask_loc) is str
        ), "mask_loc must be a string(square notation) or int(index)"
        if type(mask_loc) == "str":
            mask_square = chess.parse_square(mask_loc)
        else:
            mask_square = mask_loc

        piece = board.piece_at(mask_square)
        if piece is None:
            raise ValueError(f"no piece at mask square {mask_loc}")
        mask_tensor[0, mask_square % 8, mask_square // 8] = 1
        board_tensor = torch.cat((board_tensor, mask_tensor), dim=0)

    return board_tensor


# transform piece distribution to tensor label
# piece_distribution = {'e1': 0.5, 'd2': 0.2, ...}
# piece_distribution ==> tensor(64)
def piece_distribution_to_label(piece_distribution):
    label = torch.zeros(64)
    for i, piece in enumerate(piece_distribution):
        idx = chess.parse_square(piece)
        label[idx] = piece_distribution[piece]

    return label


def filter_by_piece(samples: List[Tuple[str, str, str, str]], piece: str):
    return list(filter(lambda x: x[2].lower() == piece, samples))


# group samples with the same board state, produce a probability distribution of moves, ingore the move
def group_board(samples):
    board_grouped = {}

    # count the number of moves for each piece in each board state
    for sample in samples:
        _from, _to, piece, board_fen = sample
        board_position = board_fen.split(" ")[0]

        if board_position not in board_grouped:

            board_grouped[board_position] = {"total": 0}

        for idx, subkey in enumerate(
            [
                _from
                #   , _to
            ]
        ):
            # if idx == 0: # specicial prefix to denote from or to
            #     subkey = f"f{subkey}"
            # else:
            #     subkey = f"t{subkey}"
            if subkey not in board_grouped[board_position]:
                board_grouped[board_position][subkey] = 0
            board_grouped[board_position][subkey] += 1

        board_grouped[board_position]["total"] += 1

    # convert the count to probability distribution
    for board_position in board_grouped:

        total = board_grouped[board_position]["total"]

        for subkey in board_grouped[board_position]:
            if subkey == "total":
                continue
            board_grouped[board_position][subkey] /= total

        del board_grouped[board_position]["total"]

    # dict to list

    board_grouped = list(board_grouped.items())
    return board_grouped


def training(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    train_loss = []
    start = time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                elapsed = time() - start
                # number of batches per epoch

                print(
                    f"epoch {epoch+1}/{num_epochs} | batch {i+1}/{len(dataloader)} | loss: {running_loss / 100} | elap: {elapsed:.2f}s"
                )
                train_loss.append(running_loss / 100)
                running_loss = 0.0
    return train_loss
