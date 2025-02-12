# Chess Move Recommender

This project implements a chess move recommender system using neural networks for selecting the best piece and move on a chessboard. The game includes a graphical user interface (GUI) for gameplay, and a bot plays as the opponent (black pieces).

## Dataset

The dataset used in this project is the [Lichess Database](https://www.kaggle.com/datasets/shkarupylomaxim/chess-games-dataset-lichess-2017-may). 

## Features

- Piece Selection: A neural network recommends the best piece to move.
- Move Selection: Separate neural networks for each type of chess piece recommend the best move for the selected piece.
- gui: Built with Pygame for an interactive chess-playing experience, where the human player (white pieces) can play against the bot (black pieces).

## Setup Instructions


1. Clone this repository:

   ```bash
   git clone https://github.com/hiu211203/Chess-move-recommender
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the src/checkpoint folder contains the trained models:

- Piece Selector Model: relu_epc15_batsize256_lr0.003.ckpt
- Move Selector Models: Trained models for each piece type, e.g., p_relu_epc15_batsize256_lr0.003.ckpt, r_relu_epc15_batsize256_lr0.003.ckpt.

4. Start the game:

   ```bash
   cd gui
   python app.py
   ```

## Contact

For any questions or feedback, please reach out to [hieu211203@gmail.com](mailto:hieu211203@gmail.com).

---

Enjoy playing! 🚀

