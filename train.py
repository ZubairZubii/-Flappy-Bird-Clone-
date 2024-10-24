import neat
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
    run(config_path)'
