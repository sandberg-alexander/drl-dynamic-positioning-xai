# Below is a Python script using pygame.
# The goal is to create a main game window that can contain multiple sub-windows or scenes,
# each with its own class. We'll show an example with n subwindows, each represented by a class.
# The subwindows can be toggled or updated as needed.

import pygame
import sys

pygame.init()

class SubWindow:
    def __init__(self, name, rect, color):
        self.name = name
        self.rect = rect  # (x, y, width, height)
        self.color = color

    def update(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        # Add your subwindow-specific logic here

class SubWindowA(SubWindow):
    def __init__(self, name, rect, color):
        super().__init__(name, rect, color)
    
    def update(self, surface):
        super().update(surface)
        # Additional drawing or logic specific to SubWindowA
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(f"{self.name}", True, (255, 255, 255))
        surface.blit(text_surface, (self.rect[0] + 5, self.rect[1] + 5))

class SubWindowB(SubWindow):
    def __init__(self, name, rect, color):
        super().__init__(name, rect, color)
    
    def update(self, surface):
        super().update(surface)
        # Additional drawing or logic specific to SubWindowB
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(f"{self.name}", True, (0, 0, 0))
        surface.blit(text_surface, (self.rect[0] + 5, self.rect[1] + 5))

class MainGame:
    def __init__(self, screen_width=800, screen_height=600, title="Multiple SubWindows"):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        # We can store any number of subwindows in a list
        self.subwindows = []

        # Example usage: create 2 different subwindow types
        self.add_subwindow(SubWindowA("SubWindow A", (50, 50, 300, 200), (255, 0, 0)))
        self.add_subwindow(SubWindowB("SubWindow B", (400, 50, 300, 200), (0, 255, 0)))

    def add_subwindow(self, subwindow):
        self.subwindows.append(subwindow)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill((30, 30, 30))

            # Update all subwindows
            for sub in self.subwindows:
                sub.update(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    # Create an instance of our MainGame with default parameters
    game = MainGame()
    # Now run the game loop
    game.run()
















import pygame
import sys

pygame.init()

class SubWindow:
    def __init__(self, name, rect, color):
        self.name = name
        self.rect = rect  # (x, y, width, height)
        self.color = color

    def update(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        # Add your subwindow-specific logic here

class SubWindowA(SubWindow):
    def __init__(self, name, rect, color):
        super().__init__(name, rect, color)

    def update(self, surface):
        super().update(surface)
        # Additional drawing or logic specific to SubWindowA
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(f"{self.name}", True, (255, 255, 255))
        surface.blit(text_surface, (self.rect[0] + 5, self.rect[1] + 5))

class SubWindowB(SubWindow):
    def __init__(self, name, rect, color):
        super().__init__(name, rect, color)

    def update(self, surface):
        super().update(surface)
        # Additional drawing or logic specific to SubWindowB
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(f"{self.name}", True, (0, 0, 0))
        surface.blit(text_surface, (self.rect[0] + 5, self.rect[1] + 5))

class MainGame:
    def __init__(self, screen_width=800, screen_height=600, title="Multiple SubWindows"):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        # We can store any number of subwindows in a list
        self.subwindows = []

        # Create 4 subwindows with padding in each corner
        # Top-left
        self.add_subwindow(SubWindowA("Top Left", (20, 20, 360, 260), (255, 0, 0)))
        # Bottom-left
        self.add_subwindow(SubWindowB("Bottom Left", (20, 320, 360, 260), (0, 255, 0)))
        # Top-right
        self.add_subwindow(SubWindowA("Top Right", (420, 20, 360, 260), (0, 0, 255)))
        # Bottom-right
        self.add_subwindow(SubWindowB("Bottom Right", (420, 320, 360, 260), (255, 255, 0)))

    def add_subwindow(self, subwindow):
        self.subwindows.append(subwindow)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill((30, 30, 30))

            # Update all subwindows
            for sub in self.subwindows:
                sub.update(self.screen)

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    game = MainGame()
    game.run()