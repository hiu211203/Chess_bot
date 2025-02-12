from model import ChessNet
import torch
import torch.nn as nn
from utils import *
from dataset import *

# TRAINING SETUP
EPOCHS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 256
LEAKY_SLOPE = 0.05
DROPOUT = 0.5
PIECE_TYPE = ["p", "n", "b", "r", "q", "k"]
DATA_PATH = "data/test_1450_to_1800_15kCUTOFF.pkl"
SAVE_DIR = "checkpoint/move_selection"


def train_on_piece(src_sample, piece):
    # load dataset
    samples = filter_by_piece(src_sample, piece)
    ds = ChessDataset(samples, mask_move=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(ds)} samples")

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(
        num_classes=64,
        num_channels=7,
        dropout=DROPOUT,
        activation=nn.LeakyReLU(inplace=True, negative_slope=LEAKY_SLOPE),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = training(model, dl, criterion, optimizer, device, num_epochs=EPOCHS)

    # save model
    file_name = f"{piece}_leakyrelu{LEAKY_SLOPE}_epc{EPOCHS}_batsize{BATCH_SIZE}_lr{LEARNING_RATE}"
    torch.save(model.state_dict(), f"{SAVE_DIR}/{file_name}.ckpt")
    print("Model saved")

    # save loss history
    with open(f"log/{file_name}.log", "wb") as f:
        pickle.dump(loss_history, f)
    print("Loss history saved")


if __name__ == "__main__":
    src_sample = read_data_sample(DATA_PATH)
    print(f"Loaded {DATA_PATH} with {len(src_sample)} samples")
    for piece in PIECE_TYPE:
        print(f"Training on piece {piece}")
        print("=" * 50)
        train_on_piece(src_sample, piece)
