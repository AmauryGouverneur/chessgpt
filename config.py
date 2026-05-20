# config.py — single source of truth for all project constants

# ---------------------------------------------------------------------------
# Vocabulary — printable ASCII (option A)
# ---------------------------------------------------------------------------

# printable ASCII: space (32) to tilde (126) = 95 characters
VOCAB      = [chr(i) for i in range(32, 127)]
UNK_CHAR   = "\x00"   # maps to id 0 for unknown characters
TOKEN2ID   = {c: i + 1 for i, c in enumerate(VOCAB)}  # ids 1-95
TOKEN2ID[UNK_CHAR] = 0
ID2TOKEN   = {i: c for c, i in TOKEN2ID.items()}
VOCAB_SIZE = 96   # 95 printable ASCII + 1 UNK

PAD_ID     = TOKEN2ID[" "]   # space as padding — blends naturally into text
UNK_ID     = 0

# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

BLOCK_SIZE = 1024
HEADER_LEN = 0    # no special header tokens — header is part of the text

# ---------------------------------------------------------------------------
# Curriculum phases
# ---------------------------------------------------------------------------

PHASES = {
    1: {"elo_min": 1000, "elo_max": 1600, "max_moves": 10},
    2: {"elo_min": 1000, "elo_max": 2000, "max_moves": 20},
    3: {"elo_min": 1000, "elo_max": 2400, "max_moves": 40},
    4: {"elo_min": 1000, "elo_max": 9999, "max_moves": 999},
    5: {"elo_min": 2000, "elo_max": 9999, "max_moves": 999},
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

N_EMBD   = 512
N_LAYER  = 12
N_HEAD   = 8
DROPOUT  = 0.1

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MIN_ELO   = 1000
MIN_MOVES = 10

# ---------------------------------------------------------------------------
# Flash
# ---------------------------------------------------------------------------

FLASH = TRUE