import random
from collections import deque

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

# 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
ACTIONS = [(0,-1), (1,0), (0,1), (-1,0)]

class Snake:
    """
    Snake environment tailored for RL + GA experiments.

    Constructor args
      width, height          : grid size in cells (ints)
      init_length            : starting snake length
      lifespan               : maximum steps per episode (agent's lifetime)
      seed                   : random seed (optional)
      state_includes_location: include agent absolute location in state dict
      state_includes_sensory : include local sensory info (neighbors)
      render_mode            : None or 'pygame' (pygame required)
    """

    def __init__(
        self,
        width=20,
        height=20,
        init_length=3,
        lifespan=5000,
        state_includes_location=True,
        state_includes_sensory=True,
        render_mode=None,
        seed=None,
        hunger_cap=80,
        frame_rate = 20,
    ):
        self.width = width
        self.height = height
        self.init_length = init_length
        self.lifespan = lifespan
        self.state_includes_location = state_includes_location
        self.state_includes_sensory = state_includes_sensory
        self.render_mode = render_mode if (render_mode == "pygame" and PYGAME_AVAILABLE) else None
        self.hunger_cap = hunger_cap
        self.hunger = hunger_cap
        self.color = 200
        self.framerate = frame_rate

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # environment state
        self.snake = deque()
        self.direction = 3  # start moving left
        self.food = None
        self.steps = 0
        self.done = False
        self.score = 0
        self.first_move = True  # safe first move flag

        # Rendering
        if self.render_mode == 'pygame':
            self._init_renderer(cell_size=20)

    # ---------- Helpers ----------
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

    # ---------- Public API ----------
    def reset(self):
        self._init_snake()
        self._place_food()
        self.steps = 0
        self.done = False
        self.score = 0
        self.hunger = self.hunger_cap
        return self._get_state()

    def step(self, action):
        """
        action: integer 0..3 mapping to up/right/down/left
        returns: state(dict), reward(float), done(bool), info(dict)
        """
        self.hunger -= 1
        if self.done:
            raise RuntimeError("step() called on done env; call reset() first.")
        
        if self.hunger  <= 0:
            reward = -5.0
            self.done = True
            return self._get_state(), reward, self.done, {"reason":"starved :("}

        # prevent direct reversal
        if abs(action - self.direction) == 2:
            action = self.direction  # ignore reverse

        self.direction = action
        dx, dy = ACTIONS[action]
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

    def render(self, scale=20):
        if self.render_mode != 'pygame':
            # simple ASCII render if pygame not available
            grid = [['.' for _ in range(self.width)] for __ in range(self.height)]
            for (x,y) in self.snake:
                grid[y][x] = 'S'
            if self.food:
                fx,fy = self.food
                grid[fy][fx] = 'F'
            print('\n'.join(''.join(row) for row in grid))
            print(f"Score: {self.score} Steps: {self.steps}")
            return

        # pygame render
        if not PYGAME_AVAILABLE:
            return
        
        self._draw()

    def close(self):
        if self.render_mode == 'pygame' and PYGAME_AVAILABLE:
            pygame.quit()

    # ---------- State representation ----------
    def _get_state(self):
        headx, heady = self.snake[0]
        state = {}

        if self.state_includes_location:
            state["x"], state["y"] = headx, heady

        # Food direction & normalized distance
        state["food_dx_sign"] = state["food_dy_sign"] = 0.0
        if self.food is None:
            self._place_food()
        if self.food:
            fx, fy = self.food
            dx, dy = fx - headx, fy - heady
            state["food_dx_sign"] = float(np.sign(dx))
            state["food_dy_sign"] = float(np.sign(dy))
            max_dist = (self.width - 1) + (self.height - 1)

        # Sensory: walls/body -> obstacle flags
        if self.state_includes_sensory:
            for name, (dx, dy) in zip(["up", "right", "down", "left"], ACTIONS):
                nx, ny = headx + dx, heady + dy
                is_wall = not self._inside((nx, ny))
                is_body = (nx, ny) in self.snake
                state[f"obstacle_{name}"] = 1 if (is_wall or is_body) else 0
                # NOTE: no more food_{name} flags – redundant with dx/dy + dist

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

    def _draw(self):
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
            
        # draw score
        score_surf = self.font.render(f"Score: {self.score}", True, (255,255,255))
        self.screen.blit(score_surf, (5,5))
        pygame.display.flip()
        self.clock.tick(self.framerate)  # limit FPS
