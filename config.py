# config.py — single source of truth for all project constants

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

CONTROL  = ["[PAD]", "[WIN]", "[DRAW]", "[LOSS]", "[GAME_END]"]
PIECES   = ["N", "B", "R", "Q", "K"]
FILES    = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS    = ["1", "2", "3", "4", "5", "6", "7", "8"]
SPECIALS = ["O-O-O", "O-O", "x", "+", "#", "[PROMOTION]"]

VOCAB    = CONTROL + PIECES + FILES + RANKS + SPECIALS
assert len(VOCAB) == 32

TOKEN2ID = {tok: i for i, tok in enumerate(VOCAB)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}
VOCAB_SIZE = 32

PAD_ID       = TOKEN2ID["[PAD]"]
WIN_ID       = TOKEN2ID["[WIN]"]
DRAW_ID      = TOKEN2ID["[DRAW]"]
LOSS_ID      = TOKEN2ID["[LOSS]"]
GAME_END_ID  = TOKEN2ID["[GAME_END]"]
PROMOTION_ID = TOKEN2ID["[PROMOTION]"]

# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

BLOCK_SIZE  = 512
HEADER_LEN  = 1   # [WIN | DRAW | LOSS]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MIN_ELO     = 1200
MIN_MOVES   = 10