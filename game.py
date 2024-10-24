'''import neat
import pygame
import random
import os
import time
pygame.font.init()

WIN_WIDTH=600
WIN_HEIGHT=800

BIRD_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join('imgs' , 'bird1.png'))),pygame.transform.scale2x(pygame.image.load(os.path.join('imgs' , 'bird2.png'))),pygame.transform.scale2x(pygame.image.load(os.path.join('imgs' , 'bird3.png')))
BASE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join('imgs' , 'base.png')))
PIPE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join('imgs' , 'pipe.png')))
BG_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join('imgs' , 'bg.png')))
Stat_Font=pygame.font.SysFont("comicsans",50)
class Bird:
    IMG=BIRD_IMG
    Max_Rotation=25
    ROT_velocity=20
    Animation_time=5

    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.vel=0
        self.height=self.y
        self.img_count=0
        self.img = self.IMG[0]

    def jump(self):
        self.vel=-10.5
        self.tick_count=0
        self.height=self.y

    def move(self):
        self.tick_count=1

        d=self.vel * self.tick_count + 0.5* self.tick_count**2
        if d>=16:
            d=16

        if d<0:
            d-=2

        self.y =self.y+d
        if d<0 or self.y < self.height +50:
            if self.tilt < self.Max_Rotation:
                self.tilt=self.Max_Rotation
        else:
            if self.tilt>-90:
                self.tilt-=self.ROT_velocity



    def draw(self,win):
        self.img_count+=1
        if self.img_count < self.Animation_time:
            self.img = self.IMG[0]
        elif self.img_count < self.Animation_time *2:
            self.img=self.IMG[1]
        elif self.img_count < self.Animation_time * 3:
            self.img = self.IMG[2]
        elif self.img_count < self.Animation_time * 4:
            self.img = self.IMG[1]
        elif self.img_count < self.Animation_time * 4 +1:
            self.img = self.IMG[0]

        if self.tilt <=-80:
            self.img=self.IMG[1]
            self.img_count= self.Animation_time*2

        rotated_img=pygame.transform.rotate(self.img,self.tilt)
        new_rect = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
        win.blit(rotated_img,new_rect.topleft)


    def get_mask(self):
        return pygame.mask.from_surface(self.img)



class Pipe:
    GAP=200
    VEL=5

    def __init__(self,x):
        self.x=x
        self.height=0
        self.top=0
        self.bottom=0
        self.Pipe_Top=pygame.transform.flip(PIPE_IMG,False,True)
        self.Pipe_Bottom = PIPE_IMG
        self.passed=False
        self.set_height()

    def set_height(self):
        self.height=random.randrange(50,450)
        self.top=self.height - self.Pipe_Top.get_height()
        self.bottom=self.height+self.GAP


    def move(self):
        self.x-=self.VEL

    def draw(self,win):
        win.blit(self.Pipe_Top,(self.x,self.top))
        win.blit(self.Pipe_Bottom,(self.x,self.bottom))


    def collision(self,bird):
        bird_mask = bird.get_mask()
        top_mask= pygame.mask.from_surface(self.Pipe_Top)
        bottom_mask = pygame.mask.from_surface(self.Pipe_Bottom)

        top_offset=(self.x  - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x -  bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask,bottom_offset)
        t_point= bird_mask.overlap(top_mask,top_offset)

        if t_point or b_point:
            return True

        return False


class Base:
    Vel =5
    width=BASE_IMG.get_width()
    IMG=BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = 0

    def move(self):
        self.x1-=self.Vel
        self.x2 -= self.Vel

        if self.x1 + self.width < 0:
            self.x1 = self.x2+self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw( self, win):
        win.blit(self.IMG,(self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
def draw_window(win,birds,pipe,base,score):
    win.blit(BG_IMG,(0,0))
    for pi in pipe:
        pi.draw(win)

    text = Stat_Font.render("Score"+ str(score) ,1 , (255,255,255))
    win.blit(text,(WIN_WIDTH-10-text.get_width(),10))
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes,config):
    nets = []
    birds = []
    ge = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(g)
        g.fitness=0
    #bird = Bird(230,350)
    base = Base(730)
    pipe = [Pipe(600)]
    score = 0
    wind=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))

    clock =pygame.time.Clock()
    run=True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run=False
                pygame.quit()
                quit()
                break
        #bird.move()

        pipe_ind=0
        if len(birds)>0:
            if len(pipe) > 1 and birds[0].x > pipe[0].x + pipe[0].Pipe_Top.get_width():
                pipe_ind=1
        else:
            run=False
            break

        for x,bird in enumerate(birds):
            ge[x].fitness+=0.1
            bird.move()

            output = nets[x].activate((bird.y,abs(bird.y - pipe[pipe_ind].height) , abs(bird.y - pipe[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()


        base.move()
        add_pipe=False
        rem=[]
        for pip in pipe:
            pip.move()
            for x,bird in enumerate(birds):
                if pip.collision(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

            if pip.x + pip.Pipe_Top.get_width() < 0:
                rem.append(pip)

            if not pip.passed and pip.x < bird.x:
                pip.passed=True
                add_pipe=True




        if add_pipe:
            score+=1
            for g in ge:
                g.fitness+=5
            pipe.append(Pipe(600))

        for r in rem:
           pipe.remove(r)

        for i,bird in enumerate(birds):
            if bird.y + bird.height > 730 or bird.y <-50 :
                birds.pop(i)
                nets.pop(i)
                ge.pop(i)

        draw_window(wind,birds,pipe,base,score)




def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StatisticsReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner= p.run(main,50)


if __name__== '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,'config-feedforward.txt')
    run(config_path)'''


"""
The classic game of flappy bird. Make with python
and pygame. Features pixel perfect collision using masks :o

Date Modified:  Jul 30, 2019
Author: Tech With Tim
Estimated Work Time: 5 hours (1 just for that damn collision)
"""


import pygame
import random
import os
import time
import neat

import pickle
pygame.font.init()  # init fontl

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

gen = 0

class Bird:
    """
    Bird class representing the flappy bird
    """
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        self.img_count += 1

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2


        # tilt the bird
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """
        gets the mask for the current image of the bird
        :return: None
        """
        return pygame.mask.from_surface(self.img)


class Pipe():
    """
    represents a pipe object
    """
    GAP = 200
    VEL = 5

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))


    def collide(self, bird, win):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False

class Base:
    """
    Represnts the moving floor of the game
    """
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    """
    Rotate a surface and blit it to the window
    :param surf: the surface to blit to
    :param image: the image surface to rotate
    :param topLeft: the top left position of the image
    :param angle: a float value for angle
    :return: None
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)

def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    """
    draws the windows for the main game loop
    :param win: pygame window surface
    :param bird: a Bird object
    :param pipes: List of pipes
    :param score: score of the game (int)
    :param gen: current generation
    :param pipe_ind: index of closest pipe
    :return: None
    """
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # score
    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # generations
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    global WIN, gen
    win = WIN
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                pipe_ind = 1                                                                 # pipe on the screen for neural network input

        for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move()

            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            for bird in birds:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)

        # break if score gets large enough
        if score > 20:
            pickle.dump(nets[0],open("best.pickle", "wb"))
            break


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)