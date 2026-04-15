import random
from collections import deque

import numpy as np

from constants import DIRECTIONS

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
VELOCITY = [(0,-1), (1,0), (0,1), (-1,0)]


class Snake:
    """
    Snake environment tailored for RL + GA experiments.

    Constructor args
      width, height          : grid size in cells (ints)
      init_length            : starting snake length
      lifespan               : maximum steps per episode (agent's lifetime)
      seed                   : random seed (optional)
      state_includes_sensory : include local sensory info (neighbors)
      render_mode            : None or 'pygame' (pygame required)
    """

    def __init__(
        self,
        width=20,
        height=20,
        init_length=3,
        lifespan=5000,
        state_includes_sensory=True,
        render_mode=None,
        seed=None,
        hunger_cap=80,  #make it starve if it doesn't bring joy
        frame_rate = 20,
    ):
        self.width = width
        self.height = height
        self.init_length = init_length
        self.lifespan = lifespan
        self.state_includes_sensory = state_includes_sensory
        self.render_mode = render_mode if (render_mode == "pygame" and PYGAME_AVAILABLE) else None
        self.hunger_cap = hunger_cap
        self.hunger = hunger_cap
        self.color = 200
        self.framerate = frame_rate

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.snake = deque()
        self.direction = 3  
        self.food = None
        self.steps = 0
        self.done = False
        self.score = 0
        self.first_move = True

        if self.render_mode == 'pygame':
            self._init_renderer(cell_size=20)

    def _inside(self, pos):
        x,y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _place_food(self):
        all_cells = {(x,y) for x in range(self.width) for y in range(self.height)}
        available = list(all_cells - set(self.snake))
        
        if not available:
            self.food = None
            return
        
        self.food = random.choice(available)

    def _init_snake(self):
        self.snake.clear()
        midx = self.width // 2
        midy = self.height // 2

        # horizontal initial snake pointing left
        for i in range(self.init_length):
            self.snake.appendleft((midx - i, midy))

        self.direction = 3
        self.first_move = True

    def reset(self):
        self._init_snake()
        self._place_food()
        self.steps = 0
        self.done = False
        self.score = 0
        self.hunger = self.hunger_cap
        return self._get_state()
    
    def change_direction(self, action):
        #changes direction
        if action == 0:
            self.direction = DIRECTIONS[self.direction-1]
        elif action == 2:
            self.direction = DIRECTIONS[(self.direction+1)%len(DIRECTIONS)]


    def step(self, action):
        """
        action: integer mapping to action choice
        returns: state(dict), reward(float), done(bool), info(dict)
        """
        self.hunger -= 1
        if self.done:
            raise RuntimeError("step() called on done env; call reset() first.")
        
        if self.hunger  <= 0:
            reward = -5.0
            self.done = True
            return self._get_state(), reward, self.done, {"reason":"starved :("}


        self.change_direction(action)
        dx, dy = VELOCITY[self.direction]
        headx, heady = self.snake[0]
        new_head = (headx + dx, heady + dy)

        reward = -0.001  # small step penalty to encourage shorter solutions
        self.steps += 1

        # collision with wall
        if not self._inside(new_head):
            reward = -5.0
            self.done = True
            return self._get_state(), reward, self.done, {"reason":"wall"}

        # collision with self (skip on first move)
        if new_head in self.snake and not self.first_move:
            reward = -5.0
            self.done = True
            return self._get_state(), reward, self.done, {"reason":"self"}

        # move snake
        self.snake.appendleft(new_head)
        self.first_move = False  # after first move

        #eat
        ate = False
        if self.food and new_head == self.food:
            ate = True
            reward = +1.0
            self.score += 1
            self.hunger=self.hunger_cap
            self._place_food()
        else:
            # pop tail (normal move)
            self.snake.pop()

        # lifespan end
        if self.lifespan is not None and self.steps >= self.lifespan:
            self.done = True
            return self._get_state(), reward, self.done, {"reason":"lifespan_end"}

        return self._get_state(), reward, self.done, {"ate": ate}

    def render(self, scale=20, epsilon = None):
        
        # pygame render
        if not PYGAME_AVAILABLE:
            return
        
        self._draw(epsilon=epsilon)

    def close(self):
        if self.render_mode == 'pygame' and PYGAME_AVAILABLE:
            pygame.quit()

    # ---------- State representation ----------
    def _get_state(self):
        headx, heady = self.snake[0]
        
        #get current direction
        dir_x, dir_y = VELOCITY[self.direction]
        # 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        #VELOCITY = [(0,-1), (1,0), (0,1), (-1,0)]
        
        #rotate vector 90 degrees right
        right_x,right_y = -dir_y, dir_x
        # so (0,-1) UP becomes 1,0 Right, Right beceoms down, and down becomes left

        
        
        
        state = {}

        # food direction & normalized distance
        state["food_front/back_norm"] = state["food_left/right_norm"] = 0.0
        if self.food is None:
            self._place_food()
        if self.food:
            fx, fy = self.food

            distance_foodx = fx - headx
            distance_foody = fy - heady
            
            #getting the sign of direction, smaller state space for testing every solution
            state["food_front/back_norm"] = np.sign((distance_foodx * dir_x + distance_foody * dir_y) / max(self.height,self.width))
            state["food_left/right_norm"] = np.sign((distance_foodx * right_x + distance_foody * right_y) /max(self.height,self.width))

        # sensory: walls/body -> obstacle flags
        rel_dirs = {
            "obstacle_front": (dir_x, dir_y),
            "obstacle_right": (right_x, right_y),
            "obstacle_left":  (-right_x, -right_y)
        }

        for key, (off_x, off_y) in rel_dirs.items():
            check_pos = (headx + off_x, heady + off_y)
            is_collision = (
                not self._inside(check_pos) or
                (check_pos in self.snake)
            )
            state[key] = 1.0 if is_collision else 0.0

        return state

    # ---------- Renderer internals ----------
    def _init_renderer(self, cell_size=20):
        pygame.init()
        self.cell_size = cell_size
        self.win_w = self.width * cell_size
        self.win_h = self.height * cell_size
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)

    def _draw(self, epsilon = None):
        # draws current grid
        self.screen.fill((30,30,30))
        # draw grid cells
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                                    self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (50,50,50), rect, 1)

        # draw snake
        snake_list = list(self.snake)
        for i, (x,y) in enumerate(snake_list):
            rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                                self.cell_size, self.cell_size)
            if i == 0:
                # head
                pygame.draw.rect(self.screen, (0,100,0), rect)
            else:
                # body
                pygame.draw.rect(self.screen, (0,self.color%255,0), rect)
                self.color += 1

        # draw food
        if self.food:
            fx,fy = self.food
            rect = pygame.Rect(fx*self.cell_size, fy*self.cell_size,
                                self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (200,0,0), rect)
            
        # draw score and other notable details
        score_surf = self.font.render(f"Score: {self.score}", True, (255,255,255))
        self.screen.blit(score_surf, (5,5))
        if epsilon:
            epsilon_surf = self.font.render(f"epsilon: {epsilon:.2f}", True, (255,255,255))
            self.screen.blit(epsilon_surf, (5 + score_surf.get_width() + 20, 5))
        
        pygame.display.flip()
        self.clock.tick(self.framerate)  # limit FPS
