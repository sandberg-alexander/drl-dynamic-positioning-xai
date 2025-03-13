import pygame
import math

def draw_curved_arrow(surface, center, r, angle, color,
                        thickness=2, arrowhead_length=15, arrowhead_width=10, num_points=30):
    """
    Draws a curved arrow along a circular arc.

    The arc is drawn along a circle of radius r centered at 'center'. The arrow
    always starts at the point directly above the center (i.e. at 270°) and
    sweeps an arc by the given 'angle' in degrees. For example, angle=90 will
    draw one-quarter of a circle. The arrowhead is drawn at the tip (end of the
    arc) and is oriented tangentially to the circle. Negative angles are also
    supported, causing the arc (and arrowhead) to be drawn in the opposite direction.

    Parameters:
      surface          - Pygame surface to draw on.
      center           - Tuple (x, y) for the center of the circle.
      r                - Radius of the circle.
      angle            - Arc angle in degrees (max is 360). The arrow starts at the
                         point (center[0], center[1]-r) and extends by this angle.
                         Can be negative.
      color            - Color for the arrow (arc and arrowhead).
      thickness        - Line thickness for the arc.
      arrowhead_length - Distance from the arrow tip back to the base of the arrowhead.
      arrowhead_width  - Width of the base of the arrowhead.
      num_points       - Number of points used to approximate the arc.
    """
    # Do nothing if no arc is specified.
    if angle == 0:
        return

    # Define starting angle: 270° (i.e. "up" from the center).
    start_deg = 270
    end_deg = start_deg + angle

    # Convert degrees to radians.
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)

    # Create a list of points along the arc.
    arc_points = []
    for i in range(num_points):
        # Linear interpolation from start to end angle.
        t = i / (num_points - 1)
        current_rad = start_rad + t * (end_rad - start_rad)
        x = center[0] + r * math.cos(current_rad)
        y = center[1] + r * math.sin(current_rad)
        arc_points.append((x, y))

    # Draw the arc as connected lines.
    pygame.draw.lines(surface, color, False, arc_points, thickness)

    # Calculate the tip of the arrow (end of the arc).
    tip = arc_points[-1]

    # Determine the tangent vector at the tip.
    # For a circle parameterized by theta, the derivative is (-sin(theta), cos(theta)).
    # However, the actual direction along the arc depends on whether angle is positive or negative.
    direction = 1 if angle > 0 else -1  # positive: one direction; negative: reverse.
    tangent = (direction * -math.sin(end_rad), direction * math.cos(end_rad))
    # (tangent is already a unit vector, but we normalize just in case)
    mag = math.hypot(tangent[0], tangent[1])
    if mag:
        tangent = (tangent[0] / mag, tangent[1] / mag)
    else:
        tangent = (0, 0)

    # Find the base center of the arrowhead by stepping back from the tip along the tangent.
    base_center = (tip[0] - arrowhead_length * tangent[0],
                   tip[1] - arrowhead_length * tangent[1])

    # Compute a perpendicular vector to the tangent (for arrowhead width).
    perp = (tangent[1], -tangent[0])

    # Determine the two base corners of the arrowhead.
    left_corner = (base_center[0] + (arrowhead_width / 2) * perp[0],
                   base_center[1] + (arrowhead_width / 2) * perp[1])
    right_corner = (base_center[0] - (arrowhead_width / 2) * perp[0],
                    base_center[1] - (arrowhead_width / 2) * perp[1])

    # Draw the arrowhead as a filled polygon.
    pygame.draw.polygon(surface, color, [tip, left_corner, right_corner])


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()

    # Define some parameters.
    center = (250, 250)
    radius = 30
    arc_angle = -180     # Try positive 90 or negative -90 or even 360.
    arrow_color = (255, 0, 0)  # Red

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # White background

        # Draw the curved arrow.
        draw_curved_arrow(screen, center, radius, arc_angle, arrow_color,
                          thickness=3, arrowhead_length=10, arrowhead_width=15)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
