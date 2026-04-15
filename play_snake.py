import pygame

from snake import Snake

env = Snake(
    width=20,
    height=20,
    init_length=3,
    lifespan=10_000,
    hunger_cap=200,
    render_mode="pygame",
    frame_rate=10,
)
 
KEY_ACTION = {

    pygame.K_a:     0,

    pygame.K_d:     2,
}
 
def main():
    env.reset()
    action = 1  
 
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key in KEY_ACTION:
                    action = KEY_ACTION[event.key]
 
        _, _, done, info = env.step(action)
        action = 1 
 
        env.render()
 
        if done:
            print(f"score: {env.score}")
            env.reset()
 
    env.close()
 
if __name__ == "__main__":
    main()