import sys
import pygame
import numpy as np
from constants import *

# pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BACKGROUND_COLOR)
pygame.display.set_caption("JOGO DA VELHA IA")


class Board:
    def __init__(self):
        self.squares = np.zeros((ROWS, COLUMNS))
        print(self.squares)

    def mark_square(self, row, col, player):
        self.squares[row][col] = player

    def empty_square(self, row, col):
        return self.squares[row][col] == 0


class Game:

    def __init__(self):
        self.board = Board()
        self.player = 1  # next player to move // 1-cross 2-circle
        self.show_lines()

    def show_lines(self):
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


def main():

    # object
    game = Game()
    board = game.board

    # main loop
    while True:

        for (
            event
        ) in pygame.event.get():  # event is any move that can happen in the game

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                row = pos[1] // SQUARE_SIZE
                col = pos[0] // SQUARE_SIZE
                # print(row, col)

                if board.empty_square(row, col):
                    board.mark_square(
                        row, col, game.player
                    )  # to know what was the players move
                    game.draw_fig(row, col)
                    game.next_turn()
                    print(board.squares)

        pygame.display.update()


main()
