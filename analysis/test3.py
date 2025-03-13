import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 600, 600  # Increased height to fit more space below the X-axis
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamic Barplot")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 122, 255)
BLACK = (0, 0, 0)

# Settings
num_bars = 14
axis_padding = 100
max_value = 1.0  # Change this value to adjust the range of the X-axis
increment = 0.2   # Increment value for ticks

# Bar settings
bar_height = (HEIGHT - 200) // num_bars  # Adjust for padding
bar_gap = 5
bar_values = [round(random.uniform(0.00, max_value), 2) for _ in range(num_bars)]  # Random values between 0.00 and max_value
max_bar_length = WIDTH - 200  # Maximum bar length

# Font for labels
font = pygame.font.SysFont(None, 24)

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Draw the bars and labels
    for i in range(num_bars):
        bar_x = axis_padding
        bar_y = 50 + i * (bar_height + bar_gap)
        bar_length = int((bar_values[i] / max_value) * max_bar_length)  # Scale bar length
        pygame.draw.rect(screen, BLUE, (bar_x, bar_y, bar_length, bar_height))

        # Draw the label aligned from the left
        label_surface = font.render(f"Bar {i+1}", True, BLACK)
        label_rect = label_surface.get_rect()
        label_rect.topleft = (axis_padding - label_rect.width - 10, bar_y + (bar_height - label_rect.height) // 2)
        screen.blit(label_surface, label_rect)

    # Draw the X and Y axes connected at the bottom
    x_axis_y = 50 + num_bars * (bar_height + bar_gap) + 30
    pygame.draw.line(screen, BLACK, (axis_padding, 50), (axis_padding, x_axis_y), 2)  # Y-axis
    pygame.draw.line(screen, BLACK, (axis_padding, x_axis_y), (WIDTH - 20, x_axis_y), 2)  # X-axis

    # Draw X-axis ticks and grid lines
    num_ticks = int(max_value / increment) + 1  # Calculate number of ticks
    for tick in range(num_ticks):
        tick_value = tick * increment
        tick_x = axis_padding + int((tick_value / max_value) * max_bar_length)
        pygame.draw.line(screen, BLACK, (tick_x, x_axis_y - 5), (tick_x, x_axis_y + 5), 2)
        tick_label = font.render(f"{tick_value:.2f}", True, BLACK)
        screen.blit(tick_label, (tick_x - 10, x_axis_y + 10))

    # Draw text at the bottom
    bottom_text = "Sum of |SHAP value|"
    bottom_text_surface = font.render(bottom_text, True, BLACK)
    bottom_text_rect = bottom_text_surface.get_rect()
    bottom_text_rect.center = (WIDTH // 2, x_axis_y + 40)
    screen.blit(bottom_text_surface, bottom_text_rect)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the display
    pygame.display.flip()
    pygame.time.Clock().tick(30)

# Quit Pygame
pygame.quit()
sys.exit()
