import pygame
import random
import pickle
import numpy as np
import time

# colors in RGB format
class color:
    def __init__(self):
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.red = (255,0,0)
        self.green = (0,255,0)
        self.blue = (0,0,255)

# the actual game
class environment:
    def __init__(self):

        # initialising pygame :D
        pygame.init()

        # initialise an empty variable
        self.epoch = None
        
        # game window size
        self.window_y = 720
        self.window_x = 480

        #block size/number of pixels per unit square
        self.block_size = 10

        #game speed
        self.game_speed = 20

        # giving the snake 1 block for body and setting its starting location
        # to be the center of the screen
        self.snake_body = [
            [self.window_y / 2,self.window_x / 2]
        ]

        # initial snake position and size 
        self.snake_position = [self.window_y / 2,self.window_x / 2]

        # initialising color :)
        self.color = color()

        # setting display size (window size, window_y, window_y)
        self.game_window = pygame.display.set_mode((self.window_y, self.window_x))

        #FPS controller 
        self.fps = pygame.time.Clock()

        # styling the font
        self.font = pygame.font.SysFont('Arial', 20)

        # current direction of snake
        self.current_direction = "RIGHT"


        # 
        self.payoff_matrix = np.zeros((self.window_y // self.block_size, self.window_x // self.block_size))

        self.payoff_matrix[int(self.snake_position[0] // self.block_size)][int(self.snake_position[1] // self.block_size)] = -1

        self.apple_position = self.spawn_apple()

        self.payoff_matrix[int(self.apple_position[0] // self.block_size)][int(self.apple_position[1] // self.block_size)] = 1

        self.snake_alive = True

        self.uneventful_move = 0

        self.score = 1

        self.number_of_rounds = 0



        




    def spawn_apple(self):
        # apple position
        # apple position is randomly generated 
        # // is division which then rounds the result down
        # suppose that one 'block' is 10 pixels wide and tall, then to get the number of length of blocks that
        # our window is, we have to divide the window_y by 10 and get the integer. We wont deal with fractional 
        # blocks. 
        # We then get a random number of blocks between 1 and the window size and then figure out the actual pixel distance
        # after to get the actual coordinates 

        apple_position = [
            random.randrange(1, (self.window_y // self.block_size)) * self.block_size,
            random.randrange(1, (self.window_x // self.block_size)) * self.block_size
        ]

        # if the apple spawns in an area that is not possible, or that is inside the snakes body,
        # retry until it isnt
        # 0's on the matrix are empty, anything else is not empty
        if self.payoff_matrix[apple_position[0] // self.block_size][apple_position[1] // self.block_size] != 0:
            self.spawn_apple()

        return apple_position


    def display_apple(self, apple_position):
        # drawing the fruit
        pygame.draw.rect(
            # surface
            self.game_window,

            # color
            self.color.red,

            # again Rect(left, top, width, height)
            pygame.Rect(apple_position[0], apple_position[1], 10, 10)
        )


    def display_snake(self, snake_body):
        # drawing the snake :)
        for position in snake_body:
            pygame.draw.rect(
                # the surface we draw on
                self.game_window,

                # the color of the snake body
                self.color.green,

                # Rect (left, top, width, height)
                # x and y axis, block height and width
                # drawing the actual blocks on the screen based off of the coordinates of snake body
                pygame.Rect(position[0], position[1], 10, 10),
            )


    # display scoring on screen and styling it
    def display_score(self):

        # create display surface and render
        # render creates new surface with specified text on it
        # pygame.font doesnt let you draw text directly on existing surface which is why we need
        # to create new surface then apply text (text can only be single line)
        score_surface = self.font.render('Score: ' + str(self.score), True, self.color.white)

        # creating and getting the rectangular area of the surface 
        score_area = score_surface.get_rect()

        # now display text
        # blit draws one image onto another (one surface onto another)
        # blit(source, destination, area, special flags)
        # blit is the actual drawing process
        self.game_window.blit(score_surface, score_area)

        # just to recap, we've styled the font to what we want
        # we've created a small surface containing the score which will be rendered
        # we've defined the area
        # we've then drawn the surface (score surface) onto the window (game_window) by giving it the coordinates (score area)


    def display_epoch(self):

        epoch_surface = self.font.render('Epoch: ' + str(self.epoch), True, self.color.white)

        self.game_window.blit(epoch_surface, [100, 0])

        # same as before but we have a different coordinate for where we want the epoch value to show up


    def is_possible(self, x, y):
        # is it true that the object x value (in terms of the game coordinates using block size 10)
        # is between 0 and the payoff matrix's x value
        # similiarly for object y value and matrix y value
        # if not then it is not a valid/possible coordinate
        # len(matrix) returns x value
        # len(matrix[0]) and len(matrix[1]) returns y
        return 0 <= (x // self.block_size) < len(self.payoff_matrix) and 0 <= (y // self.block_size) < len(self.payoff_matrix[1]) 
        # returns true or false


    # we have 12 pieces of information which we need to know at all times
    # we store these as part of q-learning
    def get_state(self):
        # start with an empty list
        state = []

        # we get the direction in which the snake is travelling
        state.append(int(self.current_direction == "UP"))
        state.append(int(self.current_direction == "DOWN"))
        state.append(int(self.current_direction == "LEFT"))
        state.append(int(self.current_direction == "RIGHT"))

        # we check where the apple is relative to the snake's head
        state.append(int((self.apple_position[1] // self.block_size) < (self.snake_position[1] // self.block_size)))
        state.append(int((self.apple_position[1] // self.block_size) > (self.snake_position[1] // self.block_size)))
        state.append(int((self.apple_position[0] // self.block_size) < (self.snake_position[0] // self.block_size)))
        state.append(int((self.apple_position[0] // self.block_size) > (self.snake_position[0] // self.block_size)))

        # we check if there is any danger nearby and store the states
        # check for all directions from the snake's head
        state.append(int(self.is_dangerous(self.snake_position[0], self.snake_position[1] + 10)))
        state.append(int(self.is_dangerous(self.snake_position[0], self.snake_position[1] - 10)))
        state.append(int(self.is_dangerous(self.snake_position[0] + 10, self.snake_position[1])))
        state.append(int(self.is_dangerous(self.snake_position[0] - 10, self.snake_position[1])))


        # return the state 
        return tuple(state)


    # is the block we're looking at dangerous?
    def is_dangerous(self, x, y):

        if self.is_possible(x, y):
            
            if self.payoff_matrix[int(x // self.block_size)][int(y // self.block_size)] == -1:
                # if the block we're looking at has a payoff of -1 its dangeous 
                return True

                #if not then its not dangerous
            return False
        
        # if the block we're looking at is not possible then its dangerous
        # in other words, its probably the border 
        else: 
            return True



    def step(self, action="None"):
        # if None move, then you can move at random
        # here we redefine the action, which is a number/index, as the next direction
        if action == "None":
            
            action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        else: 
            action = ["UP", "DOWN", "LEFT", "RIGHT"][action]
        
        # if the user presses anything else, carry on like nothing happened
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.game_over_screen()
            else:
                pass


        # now we have to handle simultaneous button presses 
        # we want to move in the direction that was pressed first, not both at the same time
        if action == 'UP' and (self.current_direction != 'DOWN' or self.score == 1):
            self.current_direction = 'UP'
            # this if you are pressing up and any key other than down, change to up
            # basically priority of key presses
        if action == 'DOWN' and (self.current_direction != 'UP' or self.score == 1):
            self.current_direction = 'DOWN'
        if action == 'LEFT' and (self.current_direction != 'RIGHT' or self.score == 1):
            self.current_direction = 'LEFT'
        if action == 'RIGHT' and (self.current_direction != 'LEFT' or self.score == 1):
            self.current_direction = 'RIGHT'

        # handling actual movement :)
        if self.current_direction == 'UP':
            # snake position is (x, y) so moving up is changing y value by -10
            # 10 because thats what we previously defined as 1 block, or 10 pixels
            self.snake_position[1] -= self.block_size
        if self.current_direction == 'DOWN':
            self.snake_position[1] += self.block_size
        if self.current_direction == 'LEFT':
            # changing the x value of the body
            self.snake_position[0] -= self.block_size
        if self.current_direction == 'RIGHT':
            self.snake_position[0] += self.block_size


        # increasing length of snake and score by 10 per apple

        # we've previously changed the snake head position on key press
        # now we have to add that to the body
        # for example, we could go from body coordinates of (100, 50), (90,50)
        # to new position of (100, 40) , (100, 50), (90, 50)
        # ie, we update the body 
        # then later we will remove the tail of snake to get final update
        # (100, 40), (100, 50)
        self.snake_body.insert(0, list(self.snake_position))

        # at some point we may stop spawning apples or we may want apples to only spawn sometimes
        apple_spawn = True 

        reward = 0

        # if snake head position same as apple position,
        # ie, x and y coordinates are the same then add score +1
        # reset apple coordinates 
        if self.snake_position[0] == self.apple_position[0] and self.snake_position[1] == self.apple_position[1]:
            # if snake eats the body then make new snake head permanent, ie body grows because we dont
            # remove the tail of the snake
            self.score += 1
            reward = 1
            apple_spawn = False
        else:
            # if snake isnt eating the apple then on the next move, remove the tail so that body size
            # stays the same 
            self.snake_body.pop()
            # if it didnt eat the apple, then its an uneventful move
            self.uneventful_move += 1

        # if spawn is off, generate new apple
        if not apple_spawn:
            self.apple_position = self.spawn_apple()
        
            # set apple spawn to true again because we just generated a new one
            apple_spawn = True


        # losing conditions which includes touching the outside of the window
        # or the snake head eating its own body

        # touching window border (we only care about snake head position)
        # if snake head touches the left border or right border
        if self.snake_position[0] < 0 or self.snake_position[0] >= (self.window_y - self.block_size):
            self.snake_alive = False
            self.score -= 1
            reward = -10
        # if snake head touches top border or bottom border respectively
        if self.snake_position[1] <= 0 or self.snake_position[1] > (self.window_x - self.block_size):
            self.snake_alive = False
            self.score -= 1
            reward = -10
        
        # snake head eating its own body
        # get all coordinates/blocks of body and exclude the snake head
        for block in self.snake_body[1:]:
            # if snake head position eating one of its body
            # we compare x coordinates and y coordinates
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                self.snake_alive = False
                self.score -= 1
                reward = -10
        
         # draw and update everything

        # update background every frame
        # we'll have to rerender screen and we'll set background to black
        self.game_window.fill(self.color.black)
        # continously updating the scoreboard
        self.display_score()
        # update epoch (in practice we only need to load this once but oh well)
        self.display_epoch()
        # continously updating food
        self.display_apple(self.apple_position)
        # continously updating snake
        self.display_snake(self.snake_body)
        
        # refresh entire screen with all its updates
        pygame.display.update()

        # update number of rounds survived
        self.number_of_rounds += 1

        return self.get_state(), reward, not self.snake_alive

    # how to lose in snake (game over)
    def game_over_screen(self):

        # creating display surface and rendering
        game_over_surface = self.font.render('You scored: ' + str(self.score), True, self.color.red)

        # creating and getting the area
        game_over_area = game_over_surface.get_rect()

        # where to place the position of the text
        game_over_area.center = (self.window_y / 2, self.window_x / 2)

        # using blit to actually draw the text
        self.game_window.blit(game_over_surface, game_over_area)
        # now display by updating the entire screen
        pygame.display.flip()

        # setting a delay before we quit everything
        time.sleep(10)

        # deactivate the library
        pygame.quit()
        # if this fails then we should quit display before quitting library by doing
        # pygame.display.quit() before pygame.quit()


    # playing a certain game back
    def play_back_game(self, epoch):
        # set epoch to itself 
        self.epoch = epoch
        # print it out...
        self.display_epoch()
        # update everything
        pygame.display.update()

        # reset, making the algorithm try to find loops
        self.uneventful_move = 0
        
        # select the file 
        filename = f"/home/tai/Documents/Snake/q_learning_snake/pickle/{epoch}.pickle"
        # open the file
        with open(filename, 'rb') as file:
            # pass through the pickle file containing the q table
            table = pickle.load(file)
        time.sleep(2)

        # actually carrying out the check
        while self.snake_alive:
        
            print("old position: ", self.snake_position, "apple position", self.apple_position)
            # get the state so that we can pick the best action
            state = self.get_state()
            # do the best action
            action = np.argmax(table[state])
            
            if self.uneventful_move == 1000:
                print("stuck in loop")
                break
            
            # take the action
            self.step(action)
            # set/reset the fps
            self.fps.tick(self.game_speed)
        
        # death/gameover screen
        if self.snake_alive == False:
            # fill it up in black
            #self.game_window.fill(self.color.black)
            # or see how death occured

            # display stats
            self.display_epoch()
            self.display_score()
            # update everything
            pygame.display.update()
            # display final game over screen
            self.game_over_screen()

        # return the stat that actually matters
        return self.score