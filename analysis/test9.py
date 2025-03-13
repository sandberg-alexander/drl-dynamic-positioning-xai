import pygame
import math

def draw_arrow(surface, color, start, end, arrow_head_length=10, arrow_head_angle=math.radians(30), width=2):
    """
    Draws an arrow from start to end on the given surface.
    
    Parameters:
        surface: The Pygame surface to draw on.
        color: Color of the arrow (e.g., (0, 0, 0) for black).
        start: Starting point as a tuple (x, y).
        end: Ending point as a tuple (x, y).
        arrow_head_length: Length of the arrowhead lines.
        arrow_head_angle: Angle (in radians) between the arrow line and each arrowhead line.
        width: Thickness of the lines.
    """
    # Draw the main line of the arrow
    pygame.draw.line(surface, color, start, end, width)
    
    # Compute the angle of the main line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    
    # Calculate endpoints for the arrowhead lines
    left_angle = angle + arrow_head_angle
    right_angle = angle - arrow_head_angle
    
    left_end = (end[0] - arrow_head_length * math.cos(left_angle),
                end[1] - arrow_head_length * math.sin(left_angle))
    right_end = (end[0] - arrow_head_length * math.cos(right_angle),
                 end[1] - arrow_head_length * math.sin(right_angle))
    
    pygame.draw.line(surface, color, end, left_end, width)
    pygame.draw.line(surface, color, end, right_end, width)

def draw_gradient_arrows(screen, angle, magnitude, color=(0, 0, 0)):
    """
    Draws a grid of arrows over the entire screen with no overlap.
    The spacing between arrow origins is based on the absolute value of the arrow magnitude,
    but with a minimum spacing of 40 pixels. Negative magnitudes result in arrows drawn in
    the opposite direction.
    
    Parameters:
        screen: The Pygame display surface.
        angle: The angle (in radians) for all arrows.
        magnitude: The length of each arrow. If negative, arrows will point opposite to the angle.
        color: Color of the arrows.
    """
    width, height = screen.get_size()
    # Use the absolute value for spacing to ensure no overlap, with a minimum of 40 pixels.
    spacing = max(60, int(abs(magnitude)))
    
    for x in range(0, width, spacing):
        for y in range(0, height, spacing):
            start = (x, y)
            end = (x + magnitude * math.cos(angle), y + magnitude * math.sin(angle))
            draw_arrow(screen, color, start, end)

# --- Example usage ---

def main():
    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Gradient of Arrows with Negative Magnitude Support")
    
    clock = pygame.time.Clock()
    running = True

    # Define arrow parameters:
    arrow_angle = math.radians(45-90)  # 45 degrees (converted to radians)
    arrow_magnitude = -8*20           # Negative arrow length; arrows point opposite to the angle
    arrow_color = (196, 196, 196)         # Black arrows

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255, 255, 255))  # Clear screen with white
        
        # Draw the grid of arrows
        draw_gradient_arrows(screen, arrow_angle, arrow_magnitude, color=arrow_color)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
