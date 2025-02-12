import sys
sys.path.append('../src')
import pygame
import chess
import torch
from model import ChessNet
from utils import transform_board

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load piece selector model
CKPT_DIR = "../src/checkpoint/above_1800"
piece_sel_ckpt_path = f"{CKPT_DIR}/relu_epc15_batsize256_lr0.003.ckpt"
piece_selector = ChessNet(
    num_channels=13, num_classes=64, activation=torch.nn.LeakyReLU(0.05), dropout=0.6
).to(device)
piece_selector.load_state_dict(torch.load(piece_sel_ckpt_path, map_location=device))

# Load move selectors for each piece
suffix = "_relu_epc15_batsize256_lr0.003"
piece_type = ["r", "n", "b", "q", "k", "p"]
move_sel_ckpt_paths = {
    piece: f"{CKPT_DIR}/{piece}{suffix}.ckpt" for piece in piece_type
}
move_selectors = {
    piece: ChessNet(
        num_channels=14, num_classes=64, activation=torch.nn.ReLU(inplace=True)
    ).to(device)
    for piece in move_sel_ckpt_paths
}
for src_piece in move_selectors:
    move_selectors[src_piece].load_state_dict(
        torch.load(move_sel_ckpt_paths[src_piece], map_location=device)
    )

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
TILE_SIZE = SCREEN_WIDTH // 8
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 43)

# Load chess piece images
PIECE_IMAGES = {}
image_mapping = {
    'p': 'black_pawn.gif', 'n': 'black_knight.gif', 'b': 'black_bishop.gif',
    'r': 'black_rook.gif', 'q': 'black_queen.gif', 'k': 'black_king.gif',
    'P': 'white_pawn.gif', 'N': 'white_knight.gif', 'B': 'white_bishop.gif',
    'R': 'white_rook.gif', 'Q': 'white_queen.gif', 'K': 'white_king.gif'
}
for piece, filename in image_mapping.items():
    PIECE_IMAGES[piece] = pygame.image.load(f"assets/{filename}")
    PIECE_IMAGES[piece] = pygame.transform.scale(PIECE_IMAGES[piece], (TILE_SIZE, TILE_SIZE))

# Initialize screen
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Automatic Chess")

# Initialize chess board
board = chess.Board()

# Helper functions
def draw_board():
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            screen.blit(PIECE_IMAGES[piece.symbol()], (col * TILE_SIZE, (7 - row) * TILE_SIZE))

def highlight_square(square):
    row, col = divmod(square, 8)
    pygame.draw.rect(screen, HIGHLIGHT, pygame.Rect(col * TILE_SIZE, (7 - row) * TILE_SIZE, TILE_SIZE, TILE_SIZE), 5)

def get_square_from_mouse(pos):
    x, y = pos
    col = x // TILE_SIZE
    row = 7 - (y // TILE_SIZE)
    return chess.square(col, row)

# BOT logic integration
def select_piece(board, model, get_square_name=False):
    input = transform_board(board, add_legal_moves=True)
    turn = board.turn

    legal_start_squares = [move.from_square for move in board.legal_moves]
    legal_start_squares = list(set(legal_start_squares))  # Remove duplicates

    if turn == chess.BLACK:
        input = input.flip(1).flip(2)

    input = input.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input)
        if turn == chess.BLACK:
            output = output.flip(1)
        indices = torch.argsort(output, descending=True).squeeze(0)

        for idx in indices:
            if idx.item() in legal_start_squares:
                selected_piece = idx.item()
                break

    if get_square_name:
        selected_piece = chess.SQUARE_NAMES[selected_piece]
    return selected_piece

def select_move(board, src_piece, models, get_square_name=False):
    legal_moves = [move for move in list(board.legal_moves) if move.from_square == src_piece]
    assert len(legal_moves) > 0, "No legal moves for the selected piece"

    piece_type = board.piece_at(src_piece).symbol().lower()
    model = models[piece_type]
    turn = board.turn

    input = transform_board(board.copy(), mask_loc=src_piece, add_legal_moves=True)
    if turn == chess.BLACK:
        input = input.flip(1).flip(2)

    input = input.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input)
        if turn == chess.BLACK:
            output = output.flip(1)
        indices = torch.argsort(output, descending=True).squeeze(0)

        for idx in indices:
            move = chess.Move(src_piece, idx.item())
            if move in legal_moves:
                selected_move = idx.item()
                break

    if get_square_name:
        selected_move = chess.SQUARE_NAMES[selected_move]
    return selected_move

class BOTPlayer:
    def __init__(self, colour, from_model, to_model):
        self.colour = colour
        self.from_model = from_model
        self.to_model = to_model

    def move(self, board):
        src_idx = select_piece(board, self.from_model)
        dist_idx = select_move(board, src_idx, self.to_model)
        move = chess.Move(src_idx, dist_idx)
        board.push(move)

running = True
selected_square = None
bot_player = BOTPlayer('black', piece_selector, move_selectors)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if board.turn == chess.WHITE:
                square = get_square_from_mouse(pygame.mouse.get_pos())
                if selected_square is None:
                    if board.piece_at(square) and board.piece_at(square).color == chess.WHITE:
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if board.is_pseudo_legal(move):
                        board.push(move)
                    selected_square = None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                board.reset()
                selected_square = None

    if board.is_checkmate() or board.is_stalemate() or not board.pieces(chess.KING, chess.WHITE) or not board.pieces(chess.KING, chess.BLACK):
        print("Game over!")
        running = False
        break

    if board.turn == chess.BLACK and not board.is_game_over():
        bot_player.move(board)

    draw_board()
    if selected_square is not None:
        highlight_square(selected_square)
    draw_pieces()
    pygame.display.flip()

pygame.quit()




