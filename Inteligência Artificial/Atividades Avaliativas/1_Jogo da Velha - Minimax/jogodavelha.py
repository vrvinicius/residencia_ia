import sys
import pygame
import numpy as np
import random
import math
import copy

# board configs
WIDTH = 500
HEIGHT = 500

ROWS = 3
COLUMNS = 3

SQUARE_SIZE = WIDTH // COLUMNS

LINE_WIDTH = 15
CIRC_WIDTH = 15
CROSS_WIDTH = 20

RADIUS = SQUARE_SIZE // 4

OFFSET = 50

BACKGROUND_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRC_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BACKGROUND_COLOR)
pygame.display.set_caption("JOGO DA VELHA - IA")


class Board:
    """
    class with the funcionalities of game board
    """

    def __init__(self):
        self.squares = np.zeros((ROWS, COLUMNS))
        self.empty_squares = self.squares  # list of empty squares
        self.marked_squares = 0

    def final_state(self, win=False):
        """
        @return 0 if there is no win yet
        @return 1 if player 1 wins
        @return 2 if player 2 wins
        """
        # vertical wins
        for col in range(COLUMNS):
            if (
                self.squares[0][col]
                == self.squares[1][col]
                == self.squares[2][col]
                != 0
            ):
                if win:
                    color = CIRC_COLOR if self.squares[0][col] == 2 else CROSS_COLOR
                    initPos = (col * SQUARE_SIZE + SQUARE_SIZE // 2, 20)
                    finPos = (col * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - 20)
                    pygame.draw.line(screen, color, initPos, finPos, LINE_WIDTH)
                return self.squares[0][col]

        # horizontal wins
        for row in range(ROWS):
            if (
                self.squares[row][0]
                == self.squares[row][1]
                == self.squares[row][2]
                != 0
            ):
                if win:
                    color = CIRC_COLOR if self.squares[row][0] == 2 else CROSS_COLOR
                    initPos = (20, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                    finPos = (WIDTH - 20, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                    pygame.draw.line(screen, color, initPos, finPos, LINE_WIDTH)
                return self.squares[row][0]

        # desc diagonal
        if self.squares[0][0] == self.squares[1][1] == self.squares[2][2] != 0:
            if win:
                color = CIRC_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                initPos = (20, 20)
                finPos = (WIDTH - 20, HEIGHT - 20)
                pygame.draw.line(screen, color, initPos, finPos, LINE_WIDTH)
            return self.squares[1][1]

        # asc diagonal
        if self.squares[2][0] == self.squares[1][1] == self.squares[0][2] != 0:
            if win:
                color = CIRC_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                initPos = (20, HEIGHT - 20)
                finPos = (WIDTH - 20, 20)
                pygame.draw.line(screen, color, initPos, finPos, LINE_WIDTH)
            return self.squares[1][1]

        # no win yet
        return 0

    def mark_square(self, row, col, player):
        self.squares[row][col] = player
        self.marked_squares += 1

    def empty_square(self, row, col):
        return self.squares[row][col] == 0

    def get_empty_squares(self):
        empty_squares = []
        for row in range(ROWS):
            for col in range(COLUMNS):
                if self.empty_square(row, col):
                    empty_squares.append((row, col))

        return empty_squares

    def is_full(self):
        return self.marked_squares == 9

    def is_empty(self):
        return self.marked_squares == 0


class Cleverness:
    """
    class to define the difficulty level of the game. Options are: 0 - Easy mode (random choice) or 1 - Hard mode (AI).
    """

    def __init__(self, level=1, player=2):  # ai is player 2
        self.level = level
        self.player = player

    def random_mode(self, board):
        empty_squares = board.get_empty_squares()
        index = random.randrange(0, len(empty_squares))

        return empty_squares[index]

    def minimax(self, board, is_maximizing):

        # terminal case
        case = board.final_state()

        # player 1 wins
        if case == 1:
            return 1, None

        # player 2 wins
        if case == 2:
            return -1, None  # ai minimazing

        elif board.is_full():
            return 0, None

        if is_maximizing:
            max_eval = -math.inf
            best_move = None
            empty_squares = board.get_empty_squares()

            for row, col in empty_squares:
                temp_board = copy.deepcopy(board)
                temp_board.mark_square(row, col, 1)  # player 1
                eval = self.minimax(temp_board, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = (row, col)

            return max_eval, best_move

        elif not is_maximizing:
            min_eval = math.inf
            best_move = None
            empty_squares = board.get_empty_squares()

            for row, col in empty_squares:
                temp_board = copy.deepcopy(board)
                temp_board.mark_square(row, col, self.player)
                eval = self.minimax(temp_board, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = (row, col)

            return min_eval, best_move

    def evaluation(self, main_board):
        if self.level == 0:
            # random choice
            eval = "random"
            move = self.random_mode(main_board)
        else:
            # minimax choice
            eval, move = self.minimax(main_board, False)  # ai is minimazing

        print(f"AI escolheu a posição {move} com uma avaliação de {eval}")

        return move  # row, col


class Game:
    """
    class with the game actions
    """

    def __init__(self):
        self.board = Board()
        self.ai = Cleverness()
        self.player = 1  # next player to move // 1-cross 2-circle
        self.gamemode = "ai"  # pvp or ai
        self.running = True
        self.show_lines()

    def make_move(self, row, col):
        self.board.mark_square(
            row, col, self.player
        )  # to know what was the players move
        self.draw_fig(row, col)
        self.next_turn()

    def show_lines(self):
        # bg for restart
        screen.fill(BACKGROUND_COLOR)
        # vertical lines
        pygame.draw.line(
            screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH
        )
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (WIDTH - SQUARE_SIZE, 0),
            (WIDTH - SQUARE_SIZE, HEIGHT),
            LINE_WIDTH,
        )

        # horizontal
        pygame.draw.line(
            screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH
        )
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (0, HEIGHT - SQUARE_SIZE),
            (WIDTH, HEIGHT - SQUARE_SIZE),
            LINE_WIDTH,
        )

    def draw_fig(self, row, col):
        if self.player == 1:
            # draw cross
            # desc line
            start_desc = (col * SQUARE_SIZE + OFFSET, row * SQUARE_SIZE + OFFSET)
            end_desc = (
                col * SQUARE_SIZE + SQUARE_SIZE - OFFSET,
                row * SQUARE_SIZE + SQUARE_SIZE - OFFSET,
            )
            pygame.draw.line(screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)

            # asc line
            start_asc = (
                col * SQUARE_SIZE + OFFSET,
                row * SQUARE_SIZE + SQUARE_SIZE - OFFSET,
            )
            end_asc = (
                col * SQUARE_SIZE + SQUARE_SIZE - OFFSET,
                row * SQUARE_SIZE + OFFSET,
            )
            pygame.draw.line(screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)

        elif self.player == 2:
            # draw circle
            center = (
                col * SQUARE_SIZE + SQUARE_SIZE // 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2,
            )
            pygame.draw.circle(screen, CIRC_COLOR, center, RADIUS, CIRC_WIDTH)

    def next_turn(self):
        self.player = self.player % 2 + 1

    def change_gamemode(self):
        self.gamemode = "ai" if self.gamemode == "pvp" else "pvp"

    def reset(self):
        self.__init__()

    def is_over(self):
        return self.board.final_state(win=True) != 0 or self.board.is_full()


def main():

    # object
    game = Game()
    board = game.board
    ai = game.ai

    # main loop
    while True:

        for (
            event
        ) in pygame.event.get():  # event is any move that can happen in the game

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

                if board.empty_square(row, col):
                    game.make_move(row, col)

            if event.type == pygame.KEYDOWN:
                # g-chance gamemode
                if event.key == pygame.K_g:
                    game.change_gamemode()

                # r-restart
                if event.key == pygame.K_r:
                    game.reset()
                    board = game.board
                    ai = game.ai

                # game level 0-random
                if event.key == pygame.K_0:
                    ai.level = 0

                # game level 1-ai
                if event.key == pygame.K_1:
                    ai.level = 1

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                row = pos[1] // SQUARE_SIZE
                col = pos[0] // SQUARE_SIZE

                if board.empty_square(row, col) and game.running:
                    game.make_move(row, col)

                    if game.is_over():
                        game.running = False

        if game.gamemode == "ai" and game.player == ai.player and game.running:
            pygame.display.update()

            # ai methods
            row, col = ai.evaluation(board)
            game.make_move(row, col)

            if game.is_over():
                game.running = False

        pygame.display.update()


main()
