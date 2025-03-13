import pygame

def draw_ned_axes_box(surface, box_top_left=(70, 70), axis_length=50, color=(0, 0, 0)):
    """
    Draws N (North) and E (East) axes (with arrowheads and labels) inside a drawing box.
    The drawing box’s top-left corner is at `box_top_left` (e.g. (70,70)).
    
    The drawing is laid out using local margins so that the entire NED drawing fits inside the box.
    
    Parameters:
        surface (pygame.Surface): The canvas on which to draw.
        box_top_left (tuple): The (x,y) position where the top-left of the drawing box is placed.
        axis_length (int): The full length from the axis origin to the arrow tip.
        color (tuple): The color used for drawing (lines, arrowheads, and text).
    """
    # Parameters for the arrow heads
    arrow_head_length = 10  # length along the axis
    arrow_head_width = 5    # half-width of the arrow head
    
    # Define margins (in local coordinates within the drawing box)
    margin_left = 20   # space from left edge of box to the axes origin
    margin_top = 20    # space from top edge of box to the north arrow tip area
    # (We don't explicitly set right/bottom margins here; they result from axis_length and margins.)
    
    # Determine the local origin (where the two axes cross) relative to the drawing box.
    # For the north axis to fit, we must leave room above the origin equal to the axis_length.
    local_origin = (box_top_left[0] + margin_left,
                    box_top_left[1] + axis_length + margin_top)
    
    # --- Draw North Axis (pointing upward) ---
    # The line stops where the arrow head begins.
    n_line_end = (local_origin[0], local_origin[1] - (axis_length - arrow_head_length))
    pygame.draw.line(surface, color, local_origin, n_line_end, 2)
    
    # Arrow head for the N axis:
    n_tip = (local_origin[0], local_origin[1] - axis_length)
    # The base of the arrow head is the end of the line (n_line_end)
    n_left = (local_origin[0] - arrow_head_width, n_line_end[1])
    n_right = (local_origin[0] + arrow_head_width, n_line_end[1])
    pygame.draw.polygon(surface, color, [n_tip, n_left, n_right])
    
    # --- Draw East Axis (pointing right) ---
    # The line stops where the arrow head begins.
    e_line_end = (local_origin[0] + (axis_length - arrow_head_length), local_origin[1])
    pygame.draw.line(surface, color, local_origin, e_line_end, 2)
    
    # Arrow head for the E axis:
    e_tip = (local_origin[0] + axis_length, local_origin[1])
    e_top = (local_origin[0] + axis_length - arrow_head_length, local_origin[1] - arrow_head_width)
    e_bottom = (local_origin[0] + axis_length - arrow_head_length, local_origin[1] + arrow_head_width)
    pygame.draw.polygon(surface, color, [e_tip, e_top, e_bottom])
    
    # --- Draw Labels ---
    # Initialize font (adjust size or font type as needed)
    font = pygame.font.SysFont(None, 24)
    
    # Place the "N" label near the north arrow tip.
    # Here we place it to the right of the tip so it doesn’t run off the top edge.
    n_label = font.render("N", True, color)
    n_label_rect = n_label.get_rect(midleft=(n_tip[0] + 5, n_tip[1]))
    surface.blit(n_label, n_label_rect)
    
    # Place the "E" label near the east arrow tip.
    # Here we place it below the tip.
    e_label = font.render("E", True, color)
    e_label_rect = e_label.get_rect(midtop=(e_tip[0], e_tip[1] + 5))
    surface.blit(e_label, e_label_rect)

    
# -------------------------------
# Example usage in a simple Pygame program:
# -------------------------------
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("NED Axes in a Drawing Box")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Fill the entire window with white
        screen.fill((255, 255, 255))
        
        # Draw the NED axes inside a drawing box whose top-left is at (70,70)
        draw_ned_axes_box(screen, box_top_left=(20, 20), axis_length=50, color=(0, 0, 0))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
