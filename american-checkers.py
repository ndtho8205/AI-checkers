import math
import random
import sys
import time

BOARD_SIZE = 8
INFINITY = sys.maxsize
DEPTH_DEFAULT = 6
start = time.time()
timeout = False


class State:
    Empty = '.'
    Red = 'r'
    Black = 'b'
    Red_King = 'R'
    Black_King = 'B'


class PieceMovement:
    TOP_LEFT = (-1, -1)
    TOP_RIGHT = (-1, 1)
    BOTTOM_LEFT = (1, -1)
    BOTTOM_RIGHT = (1, 1)


class Value:
    LOSS = -INFINITY
    WIN = INFINITY
    PIECE_WEIGHT = 30
    KING_WEIGHT = 80
    MOVE_VALUE = 2
    JUMPS_VALUE = 6
    PIECE_ROW_ADVANCE = 1
    PIECE_MIDDLE_CENTER_SQUARES = 4
    PIECE_MIDDLE_SIDE_SQUARES = -2
    PIECE_CENTER_GOALIES = 10
    PIECE_SIDE_GOALIES = 8
    PIECE_DOUBLE_CORNER = 4
    IS_HOME_FREE = 15
    DIST_FACTOR = 5
    PIECE_POSITION = [[0, -1, 1, -1, 2, -1, 3, -1],
                      [-1, 4, -1, 5, -1, 6, -1, 7],
                      [8, -1, 9, -1, 10, -1, 11, -1],
                      [-1, 12, -1, 13, -1, 14, -1, 15],
                      [16, -1, 17, -1, 18, -1, 19, -1],
                      [-1, 20, -1, 21, -1, 22, -1, 23],
                      [24, -1, 25, -1, 26, -1, 27, -1],
                      [-1, 28, -1, 29, -1, 30, -1, 31],
                      ]


class TTEntry:
    LOWER_BOUND = -1
    EXACT_VALUE = 0
    UPPER_BOUND = 1

    def __init__(self, value, entry_type, depth, move):
        self.board_value = value
        self.entry_type = entry_type
        self.search_depth = depth
        self.move = move


class CheckerBoard:
    def __init__(self, current_player):
        self.board = None
        self.red_pieces = []
        self.black_pieces = []
        self.current_player = State.Red if current_player == State.Red else State.Black
        self.can_jump = False

    def new_game(self, state):
        self.board = state
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[i][j] in ('r', 'R'):
                    self.red_pieces.append((i, j))
                elif state[i][j] in ('b', 'B'):
                    self.black_pieces.append((i, j))

    def peek_move(self, move):
        B = self.copy()
        is_red = self.is_red(move[0])
        step = 0
        B.board[move[0][0]][move[0][1]] = State.Empty

        while move and step < len(move) - 1:
            if abs(move[step + 1][0] - move[step][0]) > 1:
                row = int(math.floor((move[step][0] + move[step + 1][0]) / 2))
                col = int(math.floor((move[step][1] + move[step + 1][1]) / 2))
                B.board[row][col] = State.Empty
                if is_red:
                    B.black_pieces.remove((row, col))
                else:
                    B.red_pieces.remove((row, col))
            step = step + 1

        B.board[move[-1][0]][move[-1][1]] = self.board[move[0][0]][move[0][1]]
        if is_red:
            B.red_pieces.remove(move[0])
            B.red_pieces.append(move[-1])
        else:
            B.black_pieces.remove(move[0])
            B.black_pieces.append(move[-1])

        if self.board[move[0][0]][move[0][1]] == State.Black and move[-1][0] == 7:
            B.board[move[-1][0]][move[-1][1]] = State.Black_King
        elif self.board[move[0][0]][move[0][1]] == State.Red and move[-1][0] == 0:
            B.board[move[-1][0]][move[-1][1]] = State.Red_King

        B.piece_jumping = []
        B.current_player = State.Black if B.current_player == State.Red else State.Red
        return B

    def get_moves(self):
        jumps = self.get_all_mandatory_jumps()
        if jumps:
            self.can_jump = True
            return jumps
        else:
            self.can_jump = False
            moves = self.get_all_simple_moves()
            return moves

    def moves_gen(self, state):
        pieces = self.red_pieces if state == State.Red else self.black_pieces
        jumps = self.get_all_jumps()
        if jumps:
            return True, jumps
        else:
            moves = self.get_all_simple_moves(pieces)
            return False, moves

    def get_all_simple_moves(self, pieces=None):
        if not pieces:
            pieces = self.red_pieces if self.current_player == State.Red else self.black_pieces
        all_simple_move = []
        for piece in pieces:
            moves = self.get_simple_moves(piece)
            if moves:
                all_simple_move.extend(moves)
        return all_simple_move if all_simple_move else []

    def get_all_mandatory_jumps(self, pieces=None):
        if not pieces:
            pieces = self.red_pieces if self.current_player == State.Red else self.black_pieces
        all_mandatory_jumps = []
        for piece in pieces:
            mandatory_jumps = self.get_mandatory_jumps(self, piece)
            if mandatory_jumps:
                all_mandatory_jumps.extend(mandatory_jumps)
        return all_mandatory_jumps if all_mandatory_jumps else []

    def get_all_jumps(self, pieces=None):
        if not pieces:
            pieces = self.red_pieces if self.current_player == State.Red else self.black_pieces
        all_jumps = []
        for piece in pieces:
            jumps = self.get_jumps(piece)
            if jumps:
                all_jumps.extend(jumps)
        return all_jumps if all_jumps else []

    def get_simple_moves(self, piece):
        tl = self.move(piece, PieceMovement.TOP_LEFT)
        tr = self.move(piece, PieceMovement.TOP_RIGHT)
        bl = self.move(piece, PieceMovement.BOTTOM_LEFT)
        br = self.move(piece, PieceMovement.BOTTOM_RIGHT)
        moves = []
        moves.append(tl) if tl else None
        moves.append(tr) if tr else None
        moves.append(bl) if bl else None
        moves.append(br) if br else None
        return moves if moves else []

    def get_mandatory_jumps(self, board, piece_jumping):
        jumps = board.get_jumps(piece_jumping)
        res = []
        for jump in jumps:
            B = board.peek_move(jump)
            next_jumps = self.get_mandatory_jumps(B, jump[1])
            if next_jumps:
                for next_jump in next_jumps:
                    res.append(jump + next_jump[1:])
            else:
                res.append(jump)
        return res

    def get_jumps(self, piece):
        tlj = self.jump(piece, PieceMovement.TOP_LEFT)
        trj = self.jump(piece, PieceMovement.TOP_RIGHT)
        blj = self.jump(piece, PieceMovement.BOTTOM_LEFT)
        brj = self.jump(piece, PieceMovement.BOTTOM_RIGHT)
        jumps = []
        jumps.append(tlj) if tlj else None
        jumps.append(trj) if trj else None
        jumps.append(blj) if blj else None
        jumps.append(brj) if brj else None
        return jumps if jumps else []

    def move(self, piece, direction):
        if not self.is_legal_move(piece, direction):
            return []

        target = (piece[0] + direction[0], piece[1] + direction[1])
        return [piece, target] \
            if self.is_valid_position(target) and self.board[target[0]][target[1]] == State.Empty \
            else []

    def jump(self, piece, direction):
        if not self.is_legal_move(piece, direction):
            return []
        target = (piece[0] + direction[0] * 2, piece[1] + direction[1] * 2)
        opponent = (piece[0] + direction[0], piece[1] + direction[1])
        if self.is_valid_position(target) \
                and self.board[target[0]][target[1]] == State.Empty \
                and self.board[opponent[0]][opponent[1]] != State.Empty \
                and (self.is_black(opponent) if self.is_red(piece) else self.is_red(opponent)):
            return [piece, target]
        else:
            return []

    def is_legal_move(self, piece, direction):
        state = self.board[piece[0]][piece[1]]
        if state == State.Red \
                and (direction == PieceMovement.BOTTOM_LEFT or direction == PieceMovement.BOTTOM_RIGHT):
            return False
        if state == State.Black \
                and (direction == PieceMovement.TOP_LEFT or direction == PieceMovement.TOP_RIGHT):
            return False
        return True

    def is_valid_position(self, piece):
        return 0 <= piece[0] < BOARD_SIZE and 0 <= piece[1] < BOARD_SIZE

    def is_red(self, piece):
        return piece in self.red_pieces

    def is_black(self, piece):
        return piece in self.black_pieces

    def is_over(self):
        pieces = self.red_pieces if self.current_player == State.Red else self.black_pieces
        for piece in pieces:
            moves = self.get_simple_moves(piece)
            if moves:
                return False
            jumps = self.get_jumps(piece)
            if jumps:
                return False
        return True

    def copy(self):
        B = CheckerBoard(self.current_player)
        B.board = [row[:] for row in self.board]
        B.red_pieces = self.red_pieces[:]
        B.black_pieces = self.black_pieces[:]
        return B

    def get_hash_value(self):
        hash_value = 0
        for piece in self.red_pieces:
            hash_value = hash_value * 31 + (piece[0] * 7 + piece[1]) * math.floor(
                    ord(self.board[piece[0]][piece[1]]) / 80)
        for piece in self.black_pieces:
            hash_value = hash_value * 31 + (piece[0] * 7 + piece[1]) * math.floor(
                    ord(self.board[piece[0]][piece[1]]) / 80)
        return hash_value

    def __str__(self):
        to_string = ''
        for i in [7, 6, 5, 4, 3, 2, 1, 0]:
            to_string += str(i) + ': '
            for j in range(8):
                to_string += self.board[i][j] + ' '
            to_string += '\n'
        to_string += '   0 1 2 3 4 5 6 7'
        to_string += '\n'
        return to_string


class AiPlayer:
    def __init__(self):
        self.tt_len = 0
        self.tt_exact_used = 0
        self.tt_lower_used = 0
        self.tt_upper_used = 0
        self.alpha_beta_cutoffs = 0

    def move_function(self, board, depth=DEPTH_DEFAULT):
        tt = {0: TTEntry(-INFINITY, TTEntry.EXACT_VALUE, -INFINITY, [])}
        best_value, best_move = self.negamax(board, depth, depth, -INFINITY, INFINITY, tt)
        self.tt_len = len(tt)
        return best_move

    def negamax(self, board, depth, orig_depth, alpha=-INFINITY, beta=INFINITY, tt=None):
        orig_alpha = alpha
        hash_value = board.get_hash_value()
        lookup = None if (tt is None) else (tt[hash_value] if hash_value in tt else None)
        if lookup:
            if lookup.search_depth >= depth:
                if lookup.entry_type == TTEntry.EXACT_VALUE:
                    self.tt_exact_used += 1
                    return lookup.board_value, lookup.move
                elif lookup.entry_type == TTEntry.LOWER_BOUND:
                    self.tt_lower_used += 1
                    alpha = max(alpha, lookup.board_value)
                elif lookup.entry_type == TTEntry.UPPER_BOUND:
                    self.tt_upper_used += 1
                    beta = min(beta, lookup.board_value)

                if alpha >= beta:
                    self.alpha_beta_cutoffs += 1
                    return lookup.board_value, lookup.move

        if depth <= 0 or board.is_over():
            value = self.evaluate_function(board)
            # if tt:
            #     tt[board.get_hash_value()] = \
            #         TTEntry(
            #                 value,
            #                 TTEntry.UPPER_BOUND if value <= alpha else (
            #                     TTEntry.LOWER_BOUND if value >= beta else TTEntry.EXACT_VALUE
            #                 ),
            #                 depth, [])
            return value, []

        elapsed = time.time() - start
        if elapsed > 2.965:
            global timeout
            # if not timeout:
            # print("===================================== TIMEOUT =====================================")
            timeout = True
            return self.evaluate_function(board), []

        moves = board.get_moves()
        # if board.can_jump:
        #     sorted(moves, key=lambda x: x[0], reverse=True)
        # else:
        #     moves = sorted(moves, key=lambda x: x[0], reverse=True)
        if lookup and (lookup.move in moves):
            moves.remove(lookup.move)
            moves = [lookup.move] + moves

        if depth == orig_depth and len(moves) == 1:
            return INFINITY, moves[0]

        best_value = -INFINITY
        best_moves = [] if not moves else random.choice(moves)

        for move in moves:
            B = board.peek_move(move)
            _value, _moves = \
                self.negamax(B, depth - 1, orig_depth, -beta, -alpha, tt)
            _value = -_value
            # if timeout:
            #     break
            if _value > best_value:
                best_value = _value
                best_moves = move

            alpha = max(alpha, _value)
            if alpha >= beta:
                break

            if alpha < _value:
                alpha = _value
                if alpha >= beta:
                    break

        if tt and not timeout:
            tt[board.get_hash_value()] = \
                TTEntry(
                        best_value,
                        TTEntry.UPPER_BOUND if best_value <= orig_alpha else (
                            TTEntry.LOWER_BOUND if best_value >= beta else TTEntry.EXACT_VALUE
                        ),
                        depth, best_moves)

        return best_value, best_moves

    def evaluate_function(self, B):
        score = 0
        red_kings = black_kings = red_pieces = black_pieces = 0
        red_material = black_material = 0
        pieces = B.red_pieces + B.black_pieces

        for piece in pieces:
            row = piece[0]
            col = piece[1]
            tscore = 0
            state = B.board[row][col]
            pos = Value.PIECE_POSITION[row][col]
            if state == State.Red or state == State.Black:
                if pos in (13, 14, 17, 18):
                    tscore += Value.PIECE_MIDDLE_CENTER_SQUARES
                elif pos in (12, 16, 15, 19):
                    tscore += Value.PIECE_MIDDLE_SIDE_SQUARES
                if state == State.Red:
                    red_pieces += 1
                    red_material += Value.PIECE_WEIGHT
                    if pos in (28, 31):
                        tscore += Value.PIECE_SIDE_GOALIES
                    elif pos in (29, 30):
                        tscore += Value.PIECE_CENTER_GOALIES
                    if pos in (27, 31):
                        tscore += Value.PIECE_DOUBLE_CORNER
                else:
                    black_pieces += 1
                    black_material += Value.PIECE_WEIGHT
                    if pos in (0, 3):
                        tscore += Value.PIECE_SIDE_GOALIES
                    elif pos in (1, 2):
                        tscore += Value.PIECE_CENTER_GOALIES
                    if pos in (0, 4):
                        tscore += Value.PIECE_DOUBLE_CORNER
            elif state == State.Red_King:
                red_kings += 1
                red_material += Value.KING_WEIGHT
            elif state == State.Black_King:
                black_kings += 1
                black_material += Value.KING_WEIGHT

            if B.is_red(piece):
                score += tscore
            else:
                score -= tscore

        max_material = max(red_material, black_material)
        min_material = min(red_material, black_material)
        if min_material == 0:
            min_material = 1
        score += int((red_material - black_material) * (max_material / min_material))

        (can_jump, red_moves) = B.moves_gen(State.Red)
        tscore = sum(len(move) for move in red_moves)
        if tscore == 0:
            tscore = Value.LOSS
        else:
            if can_jump:
                tscore *= Value.JUMPS_VALUE
            else:
                tscore *= Value.MOVE_VALUE
        score += tscore

        (can_jump, black_moves) = B.moves_gen(State.Black)
        tscore = -sum(len(move) for move in black_moves)
        if tscore == 0:
            tscore = Value.WIN
        else:
            if can_jump:
                tscore *= Value.JUMPS_VALUE
            else:
                tscore *= Value.MOVE_VALUE
        score += tscore

        tscore = 0
        for piece in pieces:
            row = piece[0]
            col = piece[1]
            state = B.board[row][col]
            if state == State.Red:
                tscore += (7 - row) * Value.PIECE_ROW_ADVANCE
            elif state == State.Black:
                tscore -= row * Value.PIECE_ROW_ADVANCE
        score += tscore

        score += self.home_free(B)

        outnumber = (red_kings - black_kings) * 1 if B.current_player == State.Red else -1
        if outnumber > 0 and (black_pieces if B.current_player == State.Red else red_pieces) < 5:
            score += self.end_game(B)

        return score if B.current_player == State.Red else -score

    def home_free(self, board):
        min_red_row = -1
        max_black_row = 8
        pieces = board.red_pieces + board.black_pieces
        for piece in pieces:
            row = piece[0]
            col = piece[1]
            state = board.board[row][col]
            if board.is_red(piece) and row > min_red_row:
                min_red_row = row
            if board.is_black(piece) and row < max_black_row:
                max_black_row = row
        tscore = 0
        for piece in pieces:
            row = piece[0]
            col = piece[1]
            state = board.board[row][col]
            if state == State.Black:
                if row >= min_red_row:
                    tscore -= Value.IS_HOME_FREE
            elif state == State.Red:
                if row <= max_black_row:
                    tscore += Value.IS_HOME_FREE
        return tscore

    def distance(self, a, b):
        x_dist = (a[0] - b[0]) ** 2
        y_dist = (a[1] - b[1]) ** 2
        return math.sqrt(x_dist + y_dist) if x_dist + y_dist > 0 else 0

    def end_game(self, board):
        dist = 0.0
        pieces = board.red_pieces + board.black_pieces
        opponent = State.Black if board.current_player == State.Red else State.Red
        for p1 in pieces:
            if board.board[p1[0]][p1[1]] == opponent:
                for p2 in pieces:
                    if board.board[p2[0]][p2[1]] == board.current_player \
                            and (board.board[p2[0]][p2[1]] == State.Red_King \
                                         or board.board[p2[0]][p2[1]] == State.Black_King):
                        dist += self.distance(p1, p2)
        score = int(dist)
        return -score * Value.DIST_FACTOR


class Player:
    def __init__(self, str_name):
        self.name = str_name

    def __str__(self):
        return self.name

    def find_move(self, board):
        ai_player = AiPlayer()
        global start
        start = time.time()
        timeout = False
        best_move = ai_player.move_function(board)
        return best_move

    def nextMove(self, state):
        board = CheckerBoard(self.name)
        board.new_game(state)
        return self.find_move(board)
