import pygame, random, time, signal, sys, pickle
from pygame.locals import *

'''
SELF-LEARNING MODEL FOR FLAPPY BIRD
'''
# VARIABLES
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'

pygame.mixer.init()

TRAINING_MODE = False # if set to False, the model will not try random actions and will always choose the best action
RENDER_GUI = True  # set to false if you want to train without rendering the game, this will speed up training


class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        if RENDER_GUI:
            self.images = [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                           pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                           pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]
            self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        else:
            self.images = [pygame.Surface((34, 24)), pygame.Surface((34, 24)), pygame.Surface((34, 24))]
            self.image = pygame.Surface((34, 24))

        self.speed = SPEED

        self.current_image = 0
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY

        # UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def reset(self):
        self.rect[1] = SCREEN_HEIGHT / 2
        self.speed = SPEED


class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        if RENDER_GUI:
            self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        else:
            self.image = pygame.Surface((PIPE_WIDTH, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            if RENDER_GUI:
                self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):

    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        if RENDER_GUI:
            self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        else:
            self.image = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


def reset_game(bird, pipe_group, ground_group):
    bird.reset()

    for pipe in pipe_group:
        pipe.kill()

    for i in range(2):
        pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    for ground in ground_group:
        ground.kill()

    for i in range(2):
        ground = Ground(GROUND_WIDTH * i)
        ground_group.add(ground)


pygame.init()
if RENDER_GUI:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird')

if RENDER_GUI:
    BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
    BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
    BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()
    font = pygame.font.SysFont("Arial", 32)
else:
    BACKGROUND = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = None

bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)

ground_group = pygame.sprite.Group()

for i in range(2):
    ground = Ground(GROUND_WIDTH * i)
    ground_group.add(ground)

pipe_group = pygame.sprite.Group()
for i in range(2):
    pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

clock = pygame.time.Clock()



# parameters - these can be adjusted as needed
alpha = 0.1  # learning rate, how drastically we update the Q-values
gamma = 0.99  # discount, higher this is the more we prioritize future rewards over immediate rewards
epsilon = 0.3  # exploration rate, higher value means more exploration (i.e. random action)
epsilon_decay = 0.99  # decay the epsilon value, so we reduce exploration as time progresses
min_epsilon = 0.01  # minimum epsilon value, to make sure we always take at least some random actions
if not TRAINING_MODE:
    # if we're not training, we don't want the model to take random actions
    epsilon = 0

########################  Q_Table saving/loading code   ########################
# load q_table from file
# set this to a different file if you want to train a new model
# q_values_15k is a pre-trained model with 15k episodes
# q_values_100k is a pre-trained model with 100k episodes (best performance)
Q = {}
Q_FILE = "q_values_100k.pkl"


def load_q_table():
    global Q
    try:
        with open(Q_FILE, "rb") as f:
            Q = pickle.load(f)
            print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Q-table file not found. Starting fresh.")
        Q = {}


def save_q_table():
    with open(Q_FILE, "wb") as f:
        pickle.dump(Q, f)
        print("Q-table saved successfully.")


# on exit we'll save the new q_values to a file
def signal_handler(sig, frame):
    save_q_table()
    pygame.quit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


#########################################################################

'''
State representation:
- bird_y: vertical position of the bird
- bird_speed: vertical speed of the bird
- pipe_x: horizontal position of the next pipe
- gap_y: vertical position of the middle of the gap in the next pipe
'''
def get_state(bird, pipe_group):
    pipes = pipe_group.sprites()  # Get the list of sprites in the group
    pipe_x = int(pipes[0].rect[0] // 10)
    pipe_y = int(pipes[0].rect[1] // 10)
    bird_y = int(bird.rect[1] // 10)
    bird_speed = int(bird.speed)
    gap_y = (pipe_y + PIPE_HEIGHT // 10 // 2)  # Middle of the pipe gap
    return (bird_y, bird_speed, pipe_x, gap_y)

'''
Action space:
- flap: move the bird up
- do_nothing: do nothing
'''
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(["flap", "do_nothing"])
    else:
        return "flap" if Q.get((state, "flap"), 0) > Q.get((state, "do_nothing"), 0) else "do_nothing"

'''
update Q-values using the standard Q-learning update rule:
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a)) 
'''
def update_q_table(state, action, reward, next_state):
    best_next_action = "flap" if Q.get((next_state, "flap"), 0) > Q.get((next_state, "do_nothing"), 0) else "do_nothing"
    td_target = reward + gamma * Q.get((next_state, best_next_action), 0)
    td_delta = td_target - Q.get((state, action), 0)
    Q[(state, action)] = Q.get((state, action), 0) + alpha * td_delta


# load Q-table if available
load_q_table()

# main loop, this will continuously run the game and train until the user closes the window
episode = 0
score = 0
scores = []

while True:
    state = get_state(bird, pipe_group)
    action = choose_action(state)

    if RENDER_GUI:
        clock.tick(20)

    if action == "flap":
        bird.bump()

    # run one step of the game
    for event in pygame.event.get():
        if event.type == QUIT:
            save_q_table()
            pygame.quit()
            sys.exit()

    if RENDER_GUI:
        screen.blit(BACKGROUND, (0, 0))

    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])
        new_ground = Ground(GROUND_WIDTH - 20)
        ground_group.add(new_ground)

    if is_off_screen(pipe_group.sprites()[0]):
        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])
        pipes = get_random_pipes(SCREEN_WIDTH * 2)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

        # increase score when a pipe is passed
        score += 1

    bird_group.update()
    ground_group.update()
    pipe_group.update()

    if RENDER_GUI:
        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)

        # display the score and decision choice for each step
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        decision_text = font.render(f"Decision: {action}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        screen.blit(decision_text, (10, 40))

        pygame.display.update()

    next_state = get_state(bird, pipe_group)


    #### REWARDS ####
    # reward for passing a pipe, penalized heavily for hitting the ground or a pipe
    reward = 1 if not (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                       pygame.sprite.groupcollide(bird_group, pipe_group, False, False,
                                                  pygame.sprite.collide_mask)) else -10000

    # penalize for being too close to the ground or too high
    # this helps the bird learn to stay in the middle of the screen when 'coasting'
    if bird.rect[1] < SCREEN_HEIGHT / 6:
        reward -= 5  # penalty for being too high
    elif bird.rect[1] > 5 * SCREEN_HEIGHT / 6:
        reward -= 10  # penalty for being too low

    update_q_table(state, action, reward, next_state)

    # if the reward is less than -1000, we probably hit a pipe or the ground
    # so we'll end the episode and reset the game
    if reward < -1000:
        episode += 1
        scores.append(score)
        avg_score = sum(scores[-100:]) / min(len(scores), 100)  # running average of the last 100 scores
        print(f"Episode {episode} finished with score {score}, average score: {avg_score:.2f}")
        print(f"Epsilon: {epsilon:.4f}")
        score = 0
        reset_game(bird, pipe_group, ground_group)

    if episode % 10 == 0:
        # decay the epsilon value so we explore less as we learn
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
