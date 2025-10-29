# FDS-Pokemon: Pokémon Battle Winner Prediction

This project predicts the winner of a **Generation 1 Pokémon battle** from the perspective of Player 1.

Each battle is stored as a `.jsonl` entry containing:
- Team details for Player 1 and the opponent’s lead  
- The first 30 turns of the battle (Pokémon, HP, moves, statuses)  
- The outcome label: `player_won` (1 = win, 0 = loss)

## Goal
Build a machine learning model that predicts whether Player 1 wins using team information and early battle dynamics.

## Workflow
1. Parse `.jsonl` files → extract features  
2. Train a classifier (e.g., LightGBM, RandomForest)  
3. Predict outcomes for test battles  
4. Output a `submission.csv` with columns:


## Evaluation
Submissions are scored using **Accuracy** on the hidden test set.

## Tools
Python · pandas · scikit-learn · lightgbm
