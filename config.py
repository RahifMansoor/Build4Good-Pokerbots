# PARAMETERS TO CONTROL THE BEHAVIOR OF THE GAME ENGINE
# DO NOT REMOVE OR RENAME THIS FILE

# --- Bot Configuration ---

PLAYER_1_NAME = "A"                 # Optional: Name for Player 1 in logs (e.g., "MyBot")
PLAYER_1_PATH = "./python_skeleton" # <<< EDIT THIS to the folder of Player 1's code

# NO TRAILING SLASHES ARE ALLOWED IN PATHS

PLAYER_2_NAME = "B"                 # Optional: Name for Player 2 in logs (e.g., "ChatBot")
PLAYER_2_PATH = "./player_chatbot"  # <<< EDIT THIS to the folder of Player 2's code
                                    # Example values: "./python_skeleton", "./player_chatbot", "./all_in_bot", "./my_custom_bot_folder"


# --- Logging ---

# GAME PROGRESS IS RECORDED HERE
GAME_LOG_FILENAME = "gamelog"       # Name of the detailed game log file created after a match

# PLAYER_LOG_SIZE_LIMIT IS IN BYTES
PLAYER_LOG_SIZE_LIMIT = 524288      # Max size for the individual bot output logs (A.txt, B.txt)


# --- Timeouts and Game Clock ---

# STARTING_GAME_CLOCK AND TIMEOUTS ARE IN SECONDS
ENFORCE_GAME_CLOCK = True           # Set to False to disable the 180s time limit for testing/debugging
STARTING_GAME_CLOCK = 180.0         # Total time each bot gets for the entire match (challenge rule)
BUILD_TIMEOUT = 30.0                # Max time allowed for bot compilation/setup (if commands.json is used)
CONNECT_TIMEOUT = 30.0              # Max time allowed for a bot to connect to the engine
PLAYER_TIMEOUT = 180.0              # Timeout for player interaction (specifically for player_chatbot)


# --- Game Rules (Fixed for the Challenge) ---

# THE GAME VARIANT FIXES THE PARAMETERS BELOW
# CHANGE ONLY FOR TRAINING OR EXPERIMENTATION
NUM_ROUNDS = 5000                   # Number of hands played in a match (Challenge rule)
                                    # You might lower this (e.g., 100) for faster testing runs
STARTING_STACK = 500                # Chips each player starts with each round
BIG_BLIND = 10                      # Big blind amount
SMALL_BLIND = 5                     # Small blind amount