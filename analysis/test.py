import pygame
import math

# Constants
WIDTH, HEIGHT = 600, 600
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
SCALE = 50  # Scale to convert meters to pixels
FPS = 30

# Vessel dimensions in meters
VESSEL_LENGTH = 5.06  # Length of the vessel in meters
VESSEL_WIDTH = 2.86   # Width of the vessel in meters

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (30, 144, 255)
RED = (255, 69, 0)
SHAP_RED = (255, 0, 93)
YELLOW = (255, 255, 0)


def draw_rectangle(surface, x, y, heading, color, width, height):
    """Draw a rectangle representing a vessel or target."""
    angle_rad = math.radians(- heading + 90)  # Adjust heading to define 0 as -90 degrees
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rectangle corners relative to its center
    half_width = width / 2
    half_height = height / 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    # Rotate and translate corners
    rotated_corners = [
        (
            x + corner[0] * cos_a - corner[1] * sin_a,
            y + corner[0] * sin_a + corner[1] * cos_a
        ) for corner in corners
    ]

    # Draw the rectangle
    pygame.draw.polygon(surface, color, rotated_corners)

def draw_agent(screen, position, orientation, rect_width, rect_height):
    target_rect = pygame.Rect(0, 0, rect_width, rect_height)
    target_rect.center = position
    target_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
    target_surface.fill((0, 0, 255, 128))  # Blue color with transparency
    arrow_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
    pygame.draw.polygon(arrow_surface, (0, 0, 0), [(rect_width // 2, 0), (rect_width // 2 - 10, 20), (rect_width // 2 + 10, 20)])  # Arrow pointing up
    target_surface.blit(arrow_surface, (0, 0))
    rotated_surface = pygame.transform.rotate(target_surface, -orientation)
    new_rect = rotated_surface.get_rect(center=position)
    screen.blit(rotated_surface, new_rect.topleft)
    # Draw a small dot in the middle of the blue rectangle
    pygame.draw.circle(screen, (0, 0, 0), position, 3)


def draw_octagonal_rect(surface, cx, cy, w, h, cut, color):
    """
    Draws a centered rectangle with cut-off corners at 45 degrees,
    with a black triangle at the top to indicate orientation.

    :param surface: Pygame surface to draw on
    :param cx: Center x-coordinate
    :param cy: Center y-coordinate
    :param w: Width of the rectangle
    :param h: Height of the rectangle
    :param cut: The amount of corner cut
    :param color: Color of the shape
    """
    # Calculate top-left corner
    x = cx - w // 2
    y = cy - h // 2

    # Define the 8 corner points
    points = [
        (x + cut, y),  # Top-left cut
        (x + w - cut, y),  # Top-right cut
        (x + w, y + cut),  # Right-top cut
        (x + w, y + h - cut),  # Right-bottom cut
        (x + w - cut, y + h),  # Bottom-right cut
        (x + cut, y + h),  # Bottom-left cut
        (x, y + h - cut),  # Left-bottom cut
        (x, y + cut)  # Left-top cut
    ]

    # Draw the main shape
    pygame.draw.polygon(surface, color, points)

    # Define the orientation triangle (small black triangle at the top center)
    triangle_size = cut * 0.8  # Make the triangle proportional to the cut size
    triangle_points = [
        (cx, y),  # Top point
        (cx - triangle_size // 2, y + triangle_size),  # Bottom-left point
        (cx + triangle_size // 2, y + triangle_size)  # Bottom-right point
    ]

    # Draw the triangle
    pygame.draw.polygon(surface, BLACK, triangle_points)

def rotate_point(cx, cy, px, py, angle):
    """
    Rotates a point (px, py) around center (cx, cy) by angle (degrees).
    """
    rad = math.radians(angle)
    dx, dy = px - cx, py - cy
    new_x = cx + (dx * math.cos(rad) - dy * math.sin(rad))
    new_y = cy + (dx * math.sin(rad) + dy * math.cos(rad))
    return new_x, new_y

def draw_dashed_octagonal_rect(surface, cx, cy, angle, w, h, cut, color, dash_length=10, gap_length=5):
    """
    Draws a dashed outline of a rotated octagonal rectangle.

    :param surface: Pygame surface to draw on
    :param cx: Center x-coordinate
    :param cy: Center y-coordinate
    :param w: Width of the rectangle
    :param h: Height of the rectangle
    :param cut: The amount of corner cut
    :param color: Outline color
    :param angle: Rotation angle in degrees
    :param dash_length: Length of dashes
    :param gap_length: Length of gaps
    """
    x = cx - w // 2
    y = cy - h // 2
    angle = - angle

    # Define the 8 corner points (before rotation)
    points = [
        (x + cut, y), (x + w - cut, y),
        (x + w, y + cut), (x + w, y + h - cut),
        (x + w - cut, y + h), (x + cut, y + h),
        (x, y + h - cut), (x, y + cut)
    ]

    # Rotate all points around the center
    rotated_points = [rotate_point(cx, cy, px, py, angle) for px, py in points]

    # Function to draw dashed lines between two points
    def draw_dashed_line(surface, color, start, end, dash_length=10, gap_length=5):
        x1, y1 = start
        x2, y2 = end
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        dashes = int(length // (dash_length + gap_length))
        for i in range(dashes):
            t1 = i / dashes
            t2 = (i + 0.5) / dashes
            segment_start = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
            segment_end = (x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2)
            pygame.draw.line(surface, color, segment_start, segment_end, 2)

    # Draw dashed outline
    for i in range(len(rotated_points)):
        draw_dashed_line(surface, color, rotated_points[i], rotated_points[(i + 1) % len(rotated_points)], dash_length, gap_length)


def draw_dashed_rectangle(surface, x, y, heading, color, width, height):
    """Draw a dashed rectangle representing a target."""
    angle_rad = math.radians(- heading + 90)  # Adjust heading to define 0 as -90 degrees
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rectangle corners relative to its center
    half_width = width / 2
    half_height = height / 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]

    # Rotate and translate corners
    rotated_corners = [
        (
            x + corner[0] * cos_a - corner[1] * sin_a,
            y + corner[0] * sin_a + corner[1] * cos_a
        ) for corner in corners
    ]

    # Draw dashed lines for each side of the rectangle
    for i in range(4):
        start = rotated_corners[i]
        end = rotated_corners[(i + 1) % 4]
        draw_dashed_line(surface, start, end, color)

def draw_dashed_line(surface, start, end, color, dash_length=10):
    """Draw a dashed line between two points."""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    num_dashes = int(length // dash_length)

    for i in range(num_dashes):
        t1 = i / num_dashes
        t2 = (i + 0.5) / num_dashes
        segment_start = (x1 + t1 * dx, y1 + t1 * dy)
        segment_end = (x1 + t2 * dx, y1 + t2 * dy)
        pygame.draw.line(surface, color, segment_start, segment_end, 2)

def draw_arrow(surface, x, y, heading, arrow_length, color, font=None):
    """Draw an arrow originating from the vessel and optionally display its length."""
    angle_rad = math.radians(- heading + 90)  # Adjust heading to define 0 as -90 degrees
    arrow_width = 10   # Width of the arrow head in pixels

    end_x = x + arrow_length * math.cos(angle_rad)
    end_y = y - arrow_length * math.sin(angle_rad)

    # Calculate arrowhead points
    left_x = end_x - arrow_width * math.cos(angle_rad - math.pi / 6)
    left_y = end_y + arrow_width * math.sin(angle_rad - math.pi / 6)
    right_x = end_x - arrow_width * math.cos(angle_rad + math.pi / 6)
    right_y = end_y + arrow_width * math.sin(angle_rad + math.pi / 6)

    # Draw arrow line
    pygame.draw.line(surface, color, (x, y), (end_x, end_y), 4)
    # Draw arrowhead
    pygame.draw.polygon(surface, color, [(end_x, end_y), (left_x, left_y), (right_x, right_y)])

    # Display arrow length if font is provided
    if font:
        arrow_length_m = arrow_length / SCALE  # Convert to meters
        text = font.render(f"{arrow_length_m:.2f} m", True, color)
        surface.blit(text, ((x + end_x) // 2 - text.get_width() // 2, (y + end_y) // 2 - text.get_height() // 2))

def draw_longest_element(surface, font, arrow_length, target_heading, target_x, target_y):
    """Determine and draw the longest element (arrow, X-line, or Y-line)."""
    arrow_length_m = arrow_length / SCALE
    x_length = abs(target_x)
    y_length = abs(target_y)

    if arrow_length_m >= x_length and arrow_length_m >= y_length:
        draw_arrow(surface, CENTER_X, CENTER_Y, target_heading, arrow_length, SHAP_RED, font)
    elif x_length >= y_length:
        target_screen_x = CENTER_X + target_x * SCALE
        pygame.draw.line(surface, SHAP_RED, (CENTER_X, CENTER_Y), (target_screen_x, CENTER_Y), 2)
        pygame.draw.line(surface, SHAP_RED, (target_screen_x, CENTER_Y-5), (target_screen_x, CENTER_Y+5), 2)
        text = font.render(f"{x_length:.2f} m", True, SHAP_RED)
        surface.blit(text, ((CENTER_X + target_screen_x) // 2 - text.get_width() // 2, CENTER_Y + 10))
    else:
        target_screen_y = CENTER_Y - target_y * SCALE
        pygame.draw.line(surface, SHAP_RED, (CENTER_X, CENTER_Y), (CENTER_X, target_screen_y), 2)
        pygame.draw.line(surface, SHAP_RED, (CENTER_X-5, target_screen_y), (CENTER_X+5, target_screen_y), 2)
        text = font.render(f"{y_length:.2f} m", True, SHAP_RED)
        surface.blit(text, (CENTER_X + 10, (CENTER_Y + target_screen_y) // 2 - text.get_height() // 2))

def plot_vessel_simulation(target_x=0, target_y=0, target_heading=0, velocity=0):
    """
    Render a 10x10 box with a vessel at the center and a target pose relative to the body frame in real time using Pygame.

    Parameters:
        target_x (float): X-coordinate of the target in body-frame units (meters).
        target_y (float): Y-coordinate of the target in body-frame units (meters).
        target_heading (float): Heading of the target in degrees.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Vessel and Target Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # Font for displaying lengths

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(WHITE)

        # Draw the vessel (fixed at the center, dimensions converted to pixels)
        #draw_rectangle(screen, CENTER_X, CENTER_Y, 0, BLUE, VESSEL_LENGTH * SCALE, VESSEL_WIDTH * SCALE)
        #draw_agent(screen, (CENTER_X,CENTER_Y), 0, VESSEL_WIDTH * SCALE, VESSEL_LENGTH * SCALE)
        draw_octagonal_rect(screen, CENTER_X, CENTER_Y, VESSEL_WIDTH * SCALE, VESSEL_LENGTH * SCALE, 30, BLUE)
        
        # Draw a black dot at the center of the vessel
        pygame.draw.circle(screen, BLACK, (CENTER_X, CENTER_Y), 3)

        # Draw the yellow arrow pointing from the vessel
        draw_arrow(screen, CENTER_X, CENTER_Y, -45, 60, YELLOW, False)
        
        # Calculate target screen position
        target_screen_x = CENTER_X + target_x * SCALE
        target_screen_y = CENTER_Y - target_y * SCALE

        draw_dashed_octagonal_rect(screen, target_screen_x, target_screen_y, target_heading, VESSEL_WIDTH * SCALE, VESSEL_LENGTH * SCALE, 30, YELLOW)
        # Draw the target as a dashed rectangle
        #draw_dashed_rectangle(screen, target_screen_x, target_screen_y, target_heading, YELLOW, VESSEL_LENGTH * SCALE, VESSEL_WIDTH * SCALE)

        # Draw a black dot at the center of the target
        pygame.draw.circle(screen, BLACK, (int(target_screen_x), int(target_screen_y)), 3)
        
        # Determine and draw the longest element
        draw_longest_element(screen, font, velocity * SCALE, target_heading, target_x, target_y)

        # Draw the bounding box (10x10 meters converted to pixels)
        pygame.draw.rect(screen, BLACK, (CENTER_X - 5 * SCALE, CENTER_Y - 5 * SCALE, 10 * SCALE, 10 * SCALE), 2)

        # Update display
        pygame.display.flip()

        # Example update for simulation (replace with actual logic if needed)
        velocity = 4 * math.cos(pygame.time.get_ticks() / 800)  # Update target position
        target_x = 3 * math.cos(pygame.time.get_ticks() / 500)  # Update target position
        target_y = 3 * math.sin(pygame.time.get_ticks() / 500)
        target_heading = (pygame.time.get_ticks() / 10) % 360  # Update target heading

        # Cap the frame rate
        clock.tick(FPS)

    pygame.quit()

# Example usage
plot_vessel_simulation(target_x=2, target_y=3, target_heading=-45, velocity=4)
