import pygame

# Initialize Pygame
pygame.init()

# Set up the main screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Render Explanations")

# Define Window base class
class Window:
    def __init__(self, title="Window", width=450, height=450, x=0, y=0):
        self.title = title
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.surface = pygame.Surface((self.width, self.height))

    def render(self, screen):
        # Draw a simple rectangle to represent the window
        pygame.draw.rect(
            screen,
            (150, 150, 150),
            (self.x, self.y, self.width, self.height)
        )
        # Could do more rendering here, e.g. text, shapes, etc.

# Define specialized windows
class VesselRender(Window):
    def __init__(self, title, x, y):
        super().__init__(title=title, x=x, y=y)

class BodyRender(VesselRender):
    def __init__(self, x, y, title="BodyRender"):
        super().__init__(title, x, y)

class NedRender(VesselRender):
    def __init__(self, x, y, title="NedRender",):
        super().__init__(title, x, y)

class ShapRender(Window):
    def __init__(self, x, y, title="ShapRender",):
        super().__init__(title=title, x=x, y=y)

# Create instances of our windows
body_window = BodyRender(x=25, y=25)
ned_window = NedRender(x=25, y=525)
shap_window_top = ShapRender(title="ShapRender_top", x=525, y=25)
shap_window_bottom = ShapRender(title="ShapRender_bottom", x=525, y=525)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the main screen
    screen.fill((0, 0, 0))

    # Render each window
    body_window.render(screen)
    ned_window.render(screen)
    shap_window_top.render(screen)
    shap_window_bottom.render(screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
