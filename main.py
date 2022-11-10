import pygame
import time
import random

#game window size 
window_x = 720
window_y = 480

#colors in RGB format
black = pygame.Color(0,0,0)
white = pygame.Color(255,255,255)
red = pygame.Color(255,0,0)
green = pygame.Color(255,0,0)
blue = pygame.Color(0,0,255)

#initialising pygame so that it creates a window using previous
# variables window_x, window_y

#initialising actual pygame
pygame.init()

#initialising game window with title (set_caption)
pygame.display.set_caption('Self-learning Snake')
# setting display size (window size, window_x, window_y)
game_window = pygame.display.set_mode((window_x, window_y))

#FPS controller 
fps = pygame.time.Clock()

# initial snake position and size 
snake_position = [100, 50]

# giving the snake 2 blocks for body
snake_body = [
    [100,50],
    [90,50]
]

#apple position
# apple position is randomly generated 
# // is division which then rounds the result down
# suppose that one 'block' is 10 pixels wide and tall, then to get the number of length of blocks that
# our window is, we have to divide the window_x by 10 and get the integer. We wont deal with fractional 
# blocks. 
# We then get a random number of blocks between 1 and the window size and then figure out the actual pixel distance
# after to get the actual coordinates 
apple_position = [
    random.randrange(1, (window_x//10)) * 10,
    random.randrange(1, (window_y//10)) * 10
]

# at some point we may stop spawning apples or we may want apples to only spawn sometimes
apple_spawn = True 

# default snake direction which will be towards the right side
default_direction = 'RIGHT'

# what we will use to compare the current direction 
# user will press a key which will be the current direction
current_position = default_direction

# initialising score
score = 0

# display scoring on screen and styling it
def display_score(color, font, size):
    # font styling
    score_font = pygame.font.SysFont(font, size)

    # create display surface and render
    # render creates new surface with specified text on it
    # pygame.font doesnt let you draw text directly on existing surface which is why we need
    # to create new surface then apply text (text can only be single line)
    score_surface = score_font.render('Score: ' + str(score), True, color)

    # creating and getting the rectangular area of the surface 
    score_area = score_surface.get_rect()

    # now display text
    # blit draws one image onto another (one surface onto another)
    # blit(source, destination, area, special flags)
    # blit is the actual drawing process
    game_window.blit(score_surface, score_area)

    # just to recap, we've styled the font to what we want
    # we've created a small surface containing the score which will be rendered
    # we've defined the area
    # we've then drawn the surface (score surface) onto the window (game_window) by giving it the coordinates (score area)

# how to lose in snake (game over)
def game_over():
    # the font we want to use and its size
    text_style = pygame.font.SysFont('arial', 40)

    # creating display surface and rendering
    game_over_surface = text_style.render('You scored: ' + str(score), True, red)

    # creating and getting the area
    game_over_area = game_over_surface.get_rect()

    # where to place the position of the text
    game_over_area.center = (window_x/2, window_y/2)

    # using blit to actually draw the text
    game_window.blit(game_over_surface, game_over_area)
    # now display by updating the entire screen
    pygame.display.flip()

    # setting a delay before we quit everything
    time.sleep(3)

    # deactivate the library
    pygame.quit()
    # if this fails then we should quit display before quitting library by doing
    # pygame.display.quit() before pygame.quit()

    # now actually quit the entire program
    quit()

# time to create actual game controls


