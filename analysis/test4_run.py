from render_explanation import RenderExplaination
import pygame

import random

def main():

    num_bars = 14
    num_bar_seg = 4
    shap_values = [[round(random.uniform(0.00, 1.50),2) for _ in range(num_bars)] for _ in range(num_bar_seg)]
    vectors = [
            (0.5, -0),
            (1, 45),
            (1.5, 273),
            (2, -17),
        ]

    pygame.init()
    render_explaination = RenderExplaination()
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
    
        clock.tick(30)
        render_explaination.render_frame(shap_values, vectors)
 

if __name__ == "__main__":
    main()