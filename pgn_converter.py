def convert_coord_to_square(coord):
    """
    Convert your coordinate system (x=vertical 0->7, y=horizontal 0->7)
    into chess notation:
    (0,0) -> a8
    (7,7) -> h1
    """
    x, y = coord
    file = chr(ord('a') + (7-y))        # y=0->a, y=7->h
    rank = x + 1              # x=0->8, x=7->1
    return file + str(rank)
def is_valid_move(piece_type, before_tile, after_tile, color=True, capture=False):
    """
    Validate if a move is valid based ONLY on piece movement rules.
    Does NOT consider check, pins, or board occupancy.
    
    piece_type: string ("pawn", "rook", "knight", "bishop", "queen", "king")
    before_tile: (row, col)
    after_tile: (row, col)
    color: True for Black, False for White
    """
    r1, c1 = before_tile
    r2, c2 = after_tile
    
    dr = r2 - r1
    dc = c2 - c1
    
    abs_dr = abs(dr)
    abs_dc = abs(dc)

    # --- PAWN ---
    if piece_type == "pawn":
        direction = -1 if color else 1  # white moves upward (toward row 0)
        
        # normal 1-step forward
        if dr == direction and dc == 0:
            return True
        
        # 2-step forward from starting row
        start_row = 6 if color else 1
        if r1 == start_row and dr == 2*direction and dc == 0:
            return True
        
        # diagonal capture
        if dr == direction and abs_dc == 1 and capture == True:
            return True
        
        return False

    # --- ROOK ---
    if piece_type == "rook":
        return (dr == 0 and dc != 0) or (dc == 0 and dr != 0)

    # --- BISHOP ---
    if piece_type == "bishop":
        return abs_dr == abs_dc and abs_dr != 0

    # --- QUEEN ---
    if piece_type == "queen":
        return (
            (dr == 0 and dc != 0) or 
            (dc == 0 and dr != 0) or 
            (abs_dr == abs_dc and abs_dr != 0)
        )

    # --- KING ---
    if piece_type == "king":
        return abs_dr <= 1 and abs_dc <= 1 and not (dr == 0 and dc == 0)

    # --- KNIGHT ---
    if piece_type == "knight":
        return (abs_dr, abs_dc) in [(2,1), (1,2)]

    return False
def piece_to_letter(piece_name):
    """
    Convert your piece string to PGN letter.
    Pawns = ""
    """
    name = piece_name.split("_")[1]  # pawn, knight, etc.

    letter_map = {
        "pawn": "",
        "knight": "N",
        "bishop": "B",
        "rook": "R",
        "queen": "Q",
        "king": "K"
    }
    return letter_map[name]


def moves_to_pgn(moves):
    """
    Input: list of tuples (moved_piece, before_move, after_move, killed)
    Output: PGN string (no headers, just movetext)
    """

    pgn_moves = []
    move_number = 1
    temp_move_pair = []

    for moved_piece, before, after, killed in moves:

        piece_letter = piece_to_letter(moved_piece)
        from_square = convert_coord_to_square(before)
        to_square   = convert_coord_to_square(after)

        is_capture = (killed is not None and killed != "None" and killed != "")

        # Pawn notation (file for capture, nothing for normal move)
        if piece_letter == "":
            if is_capture:
                move_txt = f"{from_square[0]}x{to_square}"
            else:
                move_txt = f"{to_square}"
        else:
            # Piece move
            if is_capture:
                move_txt = f"{piece_letter}x{to_square}"
            else:
                move_txt = f"{piece_letter}{to_square}"

        # Append into move pairs (White first, then Black)
        if moved_piece.startswith("w_"):
            # Start a new move number
            if temp_move_pair:  
                # if previous was black-only or leftover
                pgn_moves.append(f"{move_number}. " + " ".join(temp_move_pair))
                move_number += 1
                temp_move_pair = []
            temp_move_pair.append(move_txt)

        else:  # black move
            if not temp_move_pair:
                # If black moves first accidentally
                temp_move_pair.append("...")  # placeholder for missing white move
                temp_move_pair.append(move_txt)
                pgn_moves.append(f"{move_number}" + " ".join(temp_move_pair))
                move_number += 1
                temp_move_pair = []
                continue
            temp_move_pair.append(move_txt)
            pgn_moves.append(f"{move_number}. " + " ".join(temp_move_pair))
            move_number += 1
            temp_move_pair = []

    # If game ends with white move only
    if temp_move_pair:
        pgn_moves.append(f"{move_number}. " + " ".join(temp_move_pair))

    return " ".join(pgn_moves)
