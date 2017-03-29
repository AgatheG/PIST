"REMARQUES GENERALES : VARIABLES"
# la matrice coord == le corps du Snake
# la matrice spots == la plateau de jeu

"Importation modules utiles"
from collections import deque, namedtuple
import random
import pygame
import socket
import select
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from pathlib import Path

"Creation fichier enregistrement"
my_file = Path("model.json")
if my_file.is_file():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")

else:
    "creation nouveau neural network"
    model = Sequential()
    model.add(Dense(input_dim=5, units=3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')

'DEF DES CARACTERISTIQUES DU JEU'
# vitesse du Snake
speed = 7500
BOARD_LENGTH = 32
OFFSET = 16
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
EXP = []
EPS = [0.8, 0.0025]
FOUND=[0]
LOST=[0]
COMPTEUR = [0]

ALPHA=0.6
GAMMA=0.9
DIRECTIONS = namedtuple('DIRECTIONS',
                        ['Up', 'Down', 'Left', 'Right'])(0, 1, 2, 3)

"INITIALISATION MATRICE Q"
# Q = dict()
##lettre = ["v", "t", "p"] = [0, 1, 2]
# for i in range(-1*BOARD_LENGTH, BOARD_LENGTH+1):
#    for j in range(-1*BOARD_LENGTH, BOARD_LENGTH+1):
#        for m in range(3):
#            for n in range(3):
#                for o in range(3):
#                    Q[str(i)+"_"+str(j)+str(m)+str(n)+str(o)] = [5*random.random(), 5*random.random(), 5*random.random()]


"COLORATION ALEATOIRE SUPER SWAG DU SNAKE"


def rand_color():
    return (random.randrange(254) | 64, random.randrange(254) | 64, random.randrange(254) | 64)


"ASSOCIE A TOUT ETAT POSSIBLE UN CODE UNIQUE (A PARTIR DES LA POSITION + ETAT VOISINS"

# "['case qui tue','case vide','case pomme']  = [0,1,2]
def code_etat(position, voisins, food, spots):
    s = ""

    # ecart selon les x
    s += str(position[0] - food[0]) + "_"

    # ecart selon les y
    s += str(position[1] - food[1])

    # obtention etats cases voisines
    for i in range(3):
        v = voisins[i]
        if (v[0] < 0 or v[0] >= BOARD_LENGTH or v[1] < 0 or v[1] >= BOARD_LENGTH):
            s += "0"
        else:
            if spots[v[0]][v[1]] == 0:
                s += "1"
            elif spots[v[0]][v[1]] == 1:
                s += "0"
            else:
                s += "2"
    return s

"'CREATION CLASSE SNAKE"


# demarre initialement au milieu du board ie aux coordonnees (16,16), en se dirigeant vers la droite
# remarque : deque == type d'objet (en gros une liste chainee)

class Snake(object):
    def __init__(self, direction=DIRECTIONS.Right, point=(16, 16, rand_color()), color=None):
        # taille max sachant nb popommes mangees
        self.tailmax = 4

        # dir acuelle
        self.direction = direction

        # coord points du serpent (le dernier elem == tete)
        # ie head == self.deque[self.deque.__len__-1][self.deque.__len__-1]
        self.deque = deque()
        self.deque.append(point)

        # couleur (obviously)
        self.color = color

        # prochaine direction (je sais pas pourquoi il faut un deque et pas juste unelement ici en vrai)
        self.nextDir = deque()

        # etat Snake
        self.state = ""

        # mat recompense
        #self.rewardMatrix = [[] for i in range(BOARD_LENGTH + 2)]

        # exp replay
        self.experience = EXP

        # politique decision
        self.Q = np.array([[0, 0, 0]])

    def get_color(self):
        if self.color is None:
            return rand_color()
        else:
            return self.color

    "RETOURNE LA LISTE [voisin gauche, voisin devant, voisin droite]"

    def voisins(self):
        i = self.deque[self.deque.__len__() - 1][0]
        j = self.deque[self.deque.__len__() - 1][1]
        V = []

        if (self.direction == DIRECTIONS.Up):
            V.append([i, j - 1])
            V.append([i - 1, j])
            V.append([i, j + 1])

        elif (self.direction == DIRECTIONS.Right):
            V.append([i - 1, j])
            V.append([i, j + 1])
            V.append([i + 1, j])

        elif (self.direction == DIRECTIONS.Down):
            V.append([i, j + 1])
            V.append([i + 1, j])
            V.append([i, j - 1])

        elif (self.direction == DIRECTIONS.Left):
            V.append([i + 1, j])
            V.append([i, j - 1])
            V.append([i - 1, j])
        return V

    "donne actions possibles (directions relatives pour Snake) a partir de directions absolues (par rapport au plateau)"

    def trad_direction(self, nv_dir):
        if (self.direction == DIRECTIONS.Up):
            if nv_dir == 0:
                return DIRECTIONS.Left
            if nv_dir == 1:
                return DIRECTIONS.Up
            else:
                return DIRECTIONS.Right

        elif (self.direction == DIRECTIONS.Right):
            if nv_dir == 0:
                return DIRECTIONS.Up
            if nv_dir == 1:
                return DIRECTIONS.Right
            else:
                return DIRECTIONS.Down

        elif (self.direction == DIRECTIONS.Down):
            if nv_dir == 0:
                return DIRECTIONS.Right
            if nv_dir == 1:
                return DIRECTIONS.Down
            else:
                return DIRECTIONS.Left

        elif (self.direction == DIRECTIONS.Left):
            if nv_dir == 0:
                return DIRECTIONS.Down
            if nv_dir == 1:
                return DIRECTIONS.Left
            else:
                return DIRECTIONS.Up

    # choix prochaine direction
    def populate_nextDir(self, events, identifier):
        "Code pour direction automatique du serpent"
        aleat = random.randrange(0, 3)
        tirage = random.random()

        "Cas ou le serpent explore --> direction aleat"

        if tirage < EPS[0]:
            self.nextDir.appendleft(self.trad_direction(aleat))
            if EPS[0] > EPS[1]:
                EPS[0] -=0.0000055

            return aleat

        # Cas ou le serpent exploite la politique de decision d action du QLearning
        else:

            if self.Q.argmax() == 0:
                self.nextDir.appendleft(self.trad_direction(0))
                return 0
            elif self.Q.argmax() == 1:
                self.nextDir.appendleft(self.trad_direction(1))
                # print(self.trad_direction(1))
                return 1
            else:
                self.nextDir.appendleft(self.trad_direction(2))
                # print(self.trad_direction(2))
                return 2

                #        "Code pour controle via fleches"
                #        if (identifier == "arrows"):
                #            for event in events:
                #                if event.type == pygame.KEYDOWN:
                #                    if event.key == pygame.K_UP:
                #                        self.nextDir.appendleft(DIRECTIONS.Up)
                #                    elif event.key == pygame.K_DOWN:
                #                        self.nextDir.appendleft(DIRECTIONS.Down)
                #                    elif event.key == pygame.K_RIGHT:
                #                        self.nextDir.appendleft(DIRECTIONS.Right)
                #                    elif event.key == pygame.K_LEFT:
                #                        self.nextDir.appendleft(DIRECTIONS.Left)

        "Code pour controle via clavier (si 2 joueurs)"

    #        if (identifier == "wasd"):
    #            for event in events:
    #                if event.type == pygame.KEYDOWN:
    #                    if event.key == pygame.K_z:
    #                        self.nextDir.appendleft(DIRECTIONS.Up)
    #                    elif event.key == pygame.K_s:
    #                        self.nextDir.appendleft(DIRECTIONS.Down)
    #                    elif event.key == pygame.K_d:
    #                        self.nextDir.appendleft(DIRECTIONS.Right)
    #                    elif event.key == pygame.K_q:
    #                        self.nextDir.appendleft(DIRECTIONS.Left)

#    def initializeRewardMatrix(self, food):
#        for i in range(BOARD_LENGTH + 2):
#            for j in range(BOARD_LENGTH + 2):
#                if i == 0 or i == BOARD_LENGTH + 1 or j == 0 or j == BOARD_LENGTH + 1:
#                    self.rewardMatrix[i].append(-20)
#                else:
#                    self.rewardMatrix[i].append(-1)
#
#        self.rewardMatrix[food[0]][food[1]] = 50
#        for coord in self.deque:
#            self.rewardMatrix[coord[0]][coord[1]] = -20
#        return self.rewardMatrix


'FONCTION PLACANT LA POPOMME'

def find_food(spots):
    while True:
        food = random.randrange(BOARD_LENGTH), random.randrange(BOARD_LENGTH)
        if (not (spots[food[0]][food[1]] == 1 or
                         spots[food[0]][food[1]] == 2)):
            break
    return food


'FONCTION TESTANT LES DEUX CONDITIONS DE GAMEOVER'

def end_condition(board, coord):
    # teste si le Snake vient de se manger salement le mur
    if (coord[0] < 0 or coord[0] >= BOARD_LENGTH or coord[1] < 0 or
                coord[1] >= BOARD_LENGTH):
        return True
    # teste si le Snake vient de se manger stupidement la queue
    if (board[coord[0]][coord[1]] == 1):
        return True
    return False

"FONCTION CREANT L AIRE DE JEU"

# l aire de jeu est representee par la liste de liste spots
def make_board():
    spots = [[] for i in range(BOARD_LENGTH)]
    for row in spots:
        for i in range(BOARD_LENGTH):
            row.append(0)
    return spots

"FONCTION RETOURNANT LA RECOMPENSE POUR UNE ACTION"

def get_reward(old_state, directionRelative):
    tete = int(old_state[len(old_state)-1-(2-directionRelative)])
    if(tete<32 and tete>=0 and tete<32 and tete>=0):
        #print(tete[0])
        #print(tete[1])
        if tete == 0:
            return -20
        elif tete== 1:
            return -1
        elif tete ==2:
            return 50
    else:
        return -20

"MAJ DU TABLEAU DE JEU"

def update_board(screen, snakes, food):
    rect = pygame.Rect(0, 0, OFFSET, OFFSET)
  #  snakes[0].rewardMatrix = snakes[0].initializeRewardMatrix(food)

    # redef du tableau de jeu case par case
    spots = [[] for i in range(BOARD_LENGTH)]
    num1 = 0
    num2 = 0
    for row in spots:
        for i in range(BOARD_LENGTH):
            row.append(0)
            temprect = rect.move(num1 * OFFSET, num2 * OFFSET)
            pygame.draw.rect(screen, BLACK, temprect)
            num2 += 1
        num1 += 1

    # ca place la popomme
    spots[food[0]][food[1]] = 2

    temprect = rect.move(food[1] * OFFSET, food[0] * OFFSET)
    pygame.draw.rect(screen, rand_color(), temprect)

    # ca renseigne ou qu il est le snake
    for snake in snakes:
        for coord in snake.deque:
            spots[coord[0]][coord[1]] = 1
            temprect = rect.move(coord[1] * OFFSET, coord[0] * OFFSET)
            pygame.draw.rect(screen, coord[2], temprect)
    return spots


def get_color(s):
    if s == "bk":
        return BLACK
    elif s == "wh":
        return WHITE
    elif s == "rd":
        return RED
    elif s == "bl":
        return BLUE
    elif s == "fo":
        return rand_color()
    else:
        print("WHAT", s)
        return BLUE


"I DON T KNOW REALLY LOL"


def update_board_delta(screen, deltas):
    # accepts a queue of deltas in the form
    # [("d", 13, 30), ("a", 4, 6, "rd")]
    # valid colors: re, wh, bk, bl
    rect = pygame.Rect(0, 0, OFFSET, OFFSET)
    change_list = []
    delqueue = deque()
    addqueue = deque()
    while len(deltas) != 0:
        d = deltas.pop()
        change_list.append(pygame.Rect(d[1], d[2], OFFSET, OFFSET))
        if d[0] == "d":
            delqueue.append((d[1], d[2]))
        elif d[0] == "a":
            addqueue.append((d[1], d[2], get_color(d[3])))

    for d_coord in delqueue:
        temprect = rect.move(d_coord[1] * OFFSET, d_coord[0] * OFFSET)
        # TODO generalize background color
        pygame.draw.rect(screen, BLACK, temprect)

    for a_coord in addqueue:
        temprect = rect.move(a_coord[1] * OFFSET, a_coord[0] * OFFSET)
        pygame.draw.rect(screen, a_coord[2], temprect)

    return change_list


"DEF DU MENU DU SNAKE"


# Return 0 to exit the program, 1 for a one-player game
def menu(screen):
    font = pygame.font.Font(None, 30)
    menu_message1 = font.render("Press enter for one-player, t for two-player", True, WHITE)
    menu_message2 = font.render("C'est le PIST de l'ambiance", True, WHITE)

    screen.fill(BLACK)
    screen.blit(menu_message1, (32, 32))
    screen.blit(menu_message2, (32, 64))
    pygame.display.update()
    while True:
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return 1
                if event.key == pygame.K_t:
                    return 2
                if event.key == pygame.K_l:
                    return 3
                if event.key == pygame.K_n:
                    return 4
        if done:
            break
    if done:
        pygame.quit()
        return 0


def quit(screen):
    return False


"FAIT SE DEPLACER LE SNAKE SELON LE DEPLACEMENT INDIQUE DANS LA DEQUE NEXTDIR + SA DIRCTION ACTUELLE"


def move(snake):
    if len(snake.nextDir) != 0:
        next_dir = snake.nextDir.pop()
    else:
        next_dir = snake.direction
    # direct = snake.direction
    head = snake.deque.pop()
    snake.deque.append(head)
    next_move = head

    if (next_dir == DIRECTIONS.Up):
        if snake.direction != DIRECTIONS.Down:
            next_move = (head[0] - 1, head[1], snake.get_color())
            snake.direction = next_dir
        else:
            next_move = (head[0] + 1, head[1], snake.get_color())
    elif (next_dir == DIRECTIONS.Down):
        if snake.direction != DIRECTIONS.Up:
            next_move = (head[0] + 1, head[1], snake.get_color())
            snake.direction = next_dir
        else:
            next_move = (head[0] - 1, head[1], snake.get_color())
    elif (next_dir == DIRECTIONS.Left):
        if snake.direction != DIRECTIONS.Right:
            next_move = (head[0], head[1] - 1, snake.get_color())
            snake.direction = next_dir
        else:
            next_move = (head[0], head[1] + 1, snake.get_color())
    elif (next_dir == DIRECTIONS.Right):
        if snake.direction != DIRECTIONS.Left:
            next_move = (head[0], head[1] + 1, snake.get_color())
            snake.direction = next_dir
        else:
            next_move = (head[0], head[1] - 1, snake.get_color())
    return next_move


"INDIQUE SI CASE == POPOMME"


def is_food(board, point):
    return board[point[0]][point[1]] == 2


"EXP REPLAY"


def end_cond(etat, action):
    if etat[len(etat) - 3] == '0' and action == '0':
        return False
    if etat[len(etat) - 2] == '0' and action == '1':
        return False
    if etat[len(etat) - 1] == '0' and action == '2':
        return False
    return True


def enregistrement(m):
    if COMPTEUR[0]==1000:
        model_json = m.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        m.save_weights("model.h5")
        COMPTEUR[0]=0
        print("Victoire : "
              + str(FOUND[0])
              + " Défaites : " + str(LOST[0]) + " Ratio : " + str(FOUND[0] / (LOST[0] + 1))
              + " EPS : "
              + str(EPS[0]))
        LOST[0] = 0
        FOUND[0] = 0

    else: COMPTEUR[0]+=1


"VERSION UN JOUEUR"


# Return false to quit program, true to go to
# gameover screen
def one_player(screen):
    clock = pygame.time.Clock()
    spots = make_board()

    # Board set up
    spots[0][0] = 1
    food = find_food(spots)
    snake = Snake()
    snake.state = code_etat(snake.deque[snake.deque.__len__() - 1], snake.voisins(), food, spots)
    #snake.rewardMatrix = snake.initializeRewardMatrix(food)

    while True:
        clock.tick(speed)
        # Event processing
        done = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                print("Quit given")
                done = True
                break
        if done:
            return False

        # print(snake.trad_direction(snake.direction))

        "Game logic"
        # head = snake.deque[snake.deque.__len__() - 1]
        old_state = snake.state
        lenOldState=len(old_state)
        oldStateSplit = old_state.split("_")
        temp=model.predict(np.array([[oldStateSplit[0], oldStateSplit[1], old_state[lenOldState - 3],
                                      old_state[lenOldState - 2], old_state[lenOldState - 1]]]))
        snake.Q=np.array([temp[0][0],temp[0][1],temp[0][2]])

        next_head = move(snake)
        snake.populate_nextDir(events, "arrows")
        snake.state = code_etat(next_head, snake.voisins(), food, spots)

        #new_state = snake.state

        "EXP REPLAY"
        directionRelative = snake.trad_direction(snake.direction)
        recomp = get_reward(old_state, directionRelative)

        snake.experience.append(
            [old_state, directionRelative, recomp, snake.state])

        lenExpMax = 30000
        if (len(snake.experience) > lenExpMax):
            del(snake.experience[random.randrange(0, lenExpMax)])
        batch = 32

        lenExp = len(snake.experience)
        if(lenExp>=batch):
            x_train = []
            y_train = []

            for i in range(batch):
                sample = snake.experience[random.randrange(0, lenExp)]
                sample0 = sample[0]
                sample1 = sample[1]
                sample2 = sample[2]
                sample3 = sample[3]
                lenSample0 = len(sample0)
                lenSample3 = len(sample3)

                sample0split = sample[0].split("_")
                sample3split = sample[3].split("_")
                oldState4Keras=[sample0split[0], sample0split[1],
                                                  sample0[lenSample0 - 3], sample0[lenSample0 - 2],
                                                  sample0[lenSample0 - 1]]
                Qmodif = model.predict(np.array([oldState4Keras]))
                temp2 = model.predict(np.array([[sample3split[0], sample3split[1],
                                                 sample3[lenSample3 - 3], sample3[lenSample3 - 2],
                                                 sample3[lenSample3 - 1]]]))
               # print(Qmodif)
                if end_cond(sample0, sample1):
                    Qmodif[0][sample1] = np.array([[ALPHA * sample2]])
                    # Q[sample0][sample1] = Q[sample0][sample1] + 0.6*sample2
                    # model.fit(np.array([[sample0split[0], sample0split[1], sample0[lenSample0-3], sample0[lenSample0-2], sample0[lenSample0-1]]]), Qmodif, verbose = 0)

                else:
                    Qmodif[0][sample1] = np.array(
                        [[Qmodif[0][sample1] + ALPHA * sample2 + GAMMA * (temp2.max() - Qmodif[0][sample1])]])
                    # Q[sample0][sample1] = Q[sample0][sample1] + 0.6*(sample2 + 0.9*max(Q[sample3]) - Q[sample0][sample1])
                x_train.append(oldState4Keras)
                y_train.append([Qmodif[0][0], Qmodif[0][1], Qmodif[0][2]])
#                print("L'état est " + str(sample0))
#                print("L'état suivant est " + str(sample3))
#                print("La direction choisie est " + str(sample1))
#                print("La récompense est " + str(sample2))
#                print(Qmodif)

            model.fit(np.array(x_train), np.array(y_train), verbose=0)
        "PRISE DE DECISION"
        if (end_condition(spots, next_head)):
            LOST[0]+=1
            return snake.tailmax


        enregistrement(model)

        if is_food(spots, next_head):
            FOUND[0]+=1
            snake.tailmax += 4
            food = find_food(spots)

        snake.deque.append(next_head)

        if len(snake.deque) > snake.tailmax:
            snake.deque.popleft()

        # Draw code
        screen.fill(BLACK)  # makes screen black

        spots = update_board(screen, [snake], food)

        pygame.display.update()


"VERSIO DEUX JOUEURS (pour l instant osef)"


def two_player(screen):
    clock = pygame.time.Clock()
    spots = make_board()

    snakes = [Snake(DIRECTIONS.Right, (0, 0, RED), RED), Snake(DIRECTIONS.Right, (5, 5, BLUE), BLUE)]
    for snake in snakes:
        point = snake.deque.pop()
        spots[point[0]][point[1]] = 1
        snake.deque.append(point)
    food = find_food(spots)

    while True:
        clock.tick(speed)
        done = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
                break
        if done:
            return False
        snakes[0].populate_nextDir(events, "arrows")
        snakes[1].populate_nextDir(events, "wasd")

        for snake in snakes:
            next_head = move(snake)[0]
            if (end_condition(spots, next_head)):
                return snake.tailmax

            if is_food(spots, next_head):
                snake.tailmax += 4
                food = find_food(spots)

            snake.deque.append(next_head)

            if len(snake.deque) > snake.tailmax:
                snake.deque.popleft()

        screen.fill(BLACK)

        spots = update_board(screen, snakes, food)

        pygame.display.update()


"ALORS LA..."


def network_nextDir(events, net_id):
    # assume "arrows"
    enc_dir = ""
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                enc_dir += net_id + "u"
            elif event.key == pygame.K_DOWN:
                enc_dir += net_id + "d"
            elif event.key == pygame.K_RIGHT:
                enc_dir += net_id + "r"
            elif event.key == pygame.K_LEFT:
                enc_dir += net_id + "l"
    return enc_dir


"... CA FAIT DES TRUCS"


def encode_deltas(delta_str):
    # delta_str is in the form
    # "(15 23 bk)(22 12 fo)(10 11 rm)"
    deltas = deque()
    state = "open"
    while len(delta_str) != 0:
        if state == "open":
            encoded_delta = ["fx", 0, 0, "fx"]
            delta_str = delta_str[1:]
            on_num = 1
            store_val = ""
            state = "num"
        if state == "num":
            if delta_str[0] == " ":
                delta_str = delta_str[1:]
                encoded_delta[on_num] = int(store_val)
                store_val = ""
                on_num += 1
                if on_num > 2:
                    state = "color"
            else:
                store_val += delta_str[0]
                delta_str = delta_str[1:]
        if state == "color":
            if delta_str[0] == ")":
                if store_val == "rm":
                    encoded_delta[0] = "d"
                elif store_val == "fo":
                    encoded_delta[0] = "a"
                    encoded_delta[3] = "fo"
                else:
                    encoded_delta[0] = "a"
                    encoded_delta[3] = store_val
                delta_str = delta_str[1:]
                state = "open"
                deltas.appendleft(encoded_delta)
            else:
                store_val += delta_str[0]
                delta_str = delta_str[1:]
    return deltas


"AUCUNE FUCKING IDEE DE CE QUE CA FAIT (CA TRADUIT LES ACTIONS DU JOUEUR EN GROS, OSEF JE CROIS)"


def client(screen):
    HOST, PORT = "samertm.com", 9999
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.connect((HOST, PORT))
    net_id = s.recv(1024)
    net_id = net_id.decode("utf-8")
    fake_snake = Snake()
    screen.fill(BLACK)
    pygame.display.update()

    while True:
        done = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
        if done:
            return False
        send_data = network_nextDir(events, net_id)
        if send_data != "":
            s.sendall(send_data.encode("utf-8"))

        read, _write, _except = select.select([s], [], [])
        recv_data = ""

        if len(read) != 0:
            recv_data = read[0].recv(1024)
            recv_data = recv_data.decode("utf-8")
            if recv_data == "":
                break
            deltas = encode_deltas(recv_data)
            change_list = update_board_delta(screen, deltas)
            pygame.display.update()


"DEINITION TABLEAU GAMEOVER"


def game_over(screen, eaten):
    message1 = "You ate %d foods" % eaten
    message2 = "Press enter to play again, esc to quit."
    game_over_message1 = pygame.font.Font(None, 30).render(message1, True, BLACK)
    game_over_message2 = pygame.font.Font(None, 30).render(message2, True, BLACK)

    overlay = pygame.Surface((BOARD_LENGTH * OFFSET, BOARD_LENGTH * OFFSET))
    overlay.fill((84, 84, 84))
    overlay.set_alpha(150)
    screen.blit(overlay, (0, 0))

    screen.blit(game_over_message1, (35, 35))
    screen.blit(game_over_message2, (65, 65))
    game_over_message1 = pygame.font.Font(None, 30).render(message1, True, WHITE)
    game_over_message2 = pygame.font.Font(None, 30).render(message2, True, WHITE)
    screen.blit(game_over_message1, (32, 32))
    screen.blit(game_over_message2, (62, 62))

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_RETURN:
                    return True


def leaderboard(screen):
    font = pygame.font.Font(None, 30)
    screen.fill(BLACK)
    try:
        with open("leaderboard.txt") as f:
            lines = f.readlines()
            titlemessage = font.render("Leaderboard", True, WHITE)
            screen.blit(titlemessage, (32, 32))
            dist = 64
            for line in lines:
                delimited = line.split(",")
                delimited[1] = delimited[1].strip()
                message = "{0[0]:.<10}{0[1]:.>10}".format(delimited)
                rendered_message = font.render(message, True, WHITE)
                screen.blit(rendered_message, (32, dist))
                dist += 32
    except IOError:
        message = "Nothing on the leaderboard yet."
        rendered_message = font.render(message, True, WHITE)
        screen.blit(rendered_message, (32, 32))

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_RETURN:
                    return True


def main():
    pygame.init()
    screen = pygame.display.set_mode([BOARD_LENGTH * OFFSET,
                                      BOARD_LENGTH * OFFSET])
    pygame.display.set_caption("Snaake")
    thing = pygame.Rect(10, 10, 50, 50)
    pygame.draw.rect(screen, pygame.Color(255, 255, 255, 255), pygame.Rect(50, 50, 10, 10))
    first = True
    playing = True
    while playing:
        if first or pick == 3:
            pick = menu(screen)

        options = {0: quit,
                   1: one_player,
                   2: two_player,
                   3: leaderboard,
                   4: client}
        now = options[pick](screen)
        if now == False:
            break
        elif pick == 1 or pick == 2:
            eaten = now / 4 - 1
            "DECOMMENTER LA LIGNE D EN DESSOUS == OBTENIR DES ECRANS DE GAMEOVER ENTRE CHAQUE MORT"
            # playing = game_over(screen, eaten)
            first = False

    pygame.quit()


if __name__ == "__main__":
    main()