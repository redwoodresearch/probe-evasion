"""m48's chess accuracy check: positions from random self-play, so none is in a puzzle book.

Two questions, each with a unique answer found by brute force with python-chess (no engine needed):
- mate1: the side to move has exactly one move that checkmates;
- undefended: exactly one of the opponent's pieces (not the king, not a pawn) is attacked by the side to move and
  defended by nothing.
Three ways of showing the position: FEN, a piece list, and a text board.
"""

import random
import re

import chess

PIECE_NAMES = {chess.KING: "King", chess.QUEEN: "Queen", chess.ROOK: "Rook", chess.BISHOP: "Bishop", chess.KNIGHT: "Knight", chess.PAWN: "pawn"}
FORMATS = ("fen", "pieces", "board")
QUESTIONS = ("mate1", "undefended")
SIDE = {chess.WHITE: "White", chess.BLACK: "Black"}
MIN_PLIES, MAX_PLIES = 12, 70
CHESS_ENDING = "Answer in one sentence, without explaining. Do not show any working."


def mating_moves(board: chess.Board) -> list[chess.Move]:
    out = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            out.append(m)
        board.pop()
    return out


def undefended_targets(board: chess.Board) -> list[int]:
    side, opp = board.turn, not board.turn
    return [sq for sq, pc in board.piece_map().items()
            if pc.color == opp and pc.piece_type not in (chess.KING, chess.PAWN)
            and board.attackers(side, sq) and not board.attackers(opp, sq)]


def find_position(rng: random.Random, question: str) -> chess.Board:
    """Play random legal moves from the start; return the first position (one per game, after MIN_PLIES) with a
    unique answer. Undefended positions also need the side to move not in check and no mate in one (so the question
    is the obvious thing to do)."""
    while True:
        board = chess.Board()
        stop = rng.randint(MIN_PLIES, MAX_PLIES)
        for ply in range(stop):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply + 1 < MIN_PLIES or board.is_game_over():
                continue
            if question == "mate1" and len(mating_moves(board)) == 1:
                return board
            if question == "undefended" and not board.is_check() and len(undefended_targets(board)) == 1 and not mating_moves(board):
                return board


def piece_list(board: chess.Board) -> str:
    parts = []
    for color in (chess.WHITE, chess.BLACK):
        pieces = sorted(((pc.piece_type, sq) for sq, pc in board.piece_map().items() if pc.color == color), key=lambda x: (-x[0], x[1]))
        named = [f"{PIECE_NAMES[t]} {chess.square_name(sq)}" for t, sq in pieces]
        parts.append(f"{SIDE[color]}: {', '.join(named)}.")
    return " ".join(parts)


def text_board(board: chess.Board) -> str:
    rows = [f"{8 - i} {' '.join(line.split())}" for i, line in enumerate(str(board).splitlines())]
    return "\n".join(rows + ["  a b c d e f g h"])


def show(board: chess.Board, fmt: str) -> str:
    if fmt == "fen":
        return f"Chess position (FEN): {board.fen()}"
    if fmt == "pieces":
        return f"Chess position. {piece_list(board)}"
    return f"Chess position (uppercase letters are White, lowercase are Black, dots are empty squares):\n{text_board(board)}"


def question_text(board: chess.Board, question: str) -> str:
    side, opp = SIDE[board.turn], SIDE[not board.turn]
    if question == "mate1":
        return f"{side} to move. Which move gives checkmate? Give it in algebraic notation."
    return f"{side} to move. Which of {opp}'s pieces, not counting pawns or the king, is attacked by {side} and not defended by anything? Give its square."


def chess_item(rng: random.Random, question: str) -> dict:
    board = find_position(rng, question)
    answer = board.san(mating_moves(board)[0]) if question == "mate1" else chess.square_name(undefended_targets(board)[0])
    return {"fen": board.fen(), "question": question, "answer": answer,
            "prompts": {fmt: f"{show(board, fmt)}\n{question_text(board, question)} {CHESS_ENDING}" for fmt in FORMATS}}


MOVE_TOKEN = re.compile(r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[a-h][1-8][a-h][1-8][qrbn]?)[+#]?", re.I)
SQUARE = re.compile(r"\b([a-h][1-8])\b")


def first_move(board: chess.Board, reply: str) -> chess.Move | None:
    """The first token in the reply that parses as a legal move (SAN, then UCI)."""
    for m in MOVE_TOKEN.finditer(reply):
        tok = m.group(1)
        for parse in (board.parse_san, board.parse_uci):
            try:
                return parse(tok)
            except ValueError:
                continue
    return None


def chess_check(item: dict, reply: str) -> dict:
    board = chess.Board(item["fen"])
    if item["question"] == "mate1":
        mv = first_move(board, reply)
        got = board.san(mv) if mv else None
        return {"correct": got == item["answer"], "extracted": got}
    sq = SQUARE.findall(reply)
    return {"correct": bool(sq) and sq[0] == item["answer"], "extracted": sq[0] if sq else None}
