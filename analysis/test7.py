import pygame
import math

def draw_arrow(surface, start, angle, arrow_length, color,
               arrowhead_length=10, arrowhead_width=10, line_width=2):
    """
    Draws an arrow on a Pygame surface.
    
    Negative arrow_length is supported:
      If arrow_length is negative, the arrow will be drawn in the opposite direction.
    
    Parameters:
        surface (pygame.Surface): Surface to draw the arrow on.
        start (tuple): (x, y) starting point of the arrow.
        angle (float): Angle (in radians) the arrow points.
        arrow_length (float): Total arrow length (line + arrowhead). Can be negative.
        color (tuple): Color of the arrow (e.g., (255, 0, 0) for red).
        arrowhead_length (float): Fixed length for the arrowhead (if possible).
        arrowhead_width (float): Base width of the arrowhead triangle.
        line_width (int): Width of the arrow line.
    """
    angle = math.radians(angle-90)

    # Support negative arrow_length by flipping the arrow direction.
    if arrow_length < 0:
        arrow_length = -arrow_length
        angle += math.pi  # Flip the direction by 180 degrees.
    
    # Determine how much length is allocated for the line and arrowhead.
    if arrow_length > arrowhead_length:
        line_length = arrow_length - arrowhead_length
        current_arrowhead_length = arrowhead_length
    else:
        # If the total length is too short, use it all for the arrowhead.
        line_length = 0
        current_arrowhead_length = arrow_length
        # Scale the arrowhead's base width to maintain its angles.
        arrowhead_width *= (arrow_length / arrowhead_length)

    # Calculate the end point of the line (where the arrowhead begins).
    line_end = (
        start[0] + line_length * math.cos(angle),
        start[1] + line_length * math.sin(angle)
    )

    # Calculate the tip of the arrow (end of the arrowhead).
    tip = (
        line_end[0] + current_arrowhead_length * math.cos(angle),
        line_end[1] + current_arrowhead_length * math.sin(angle)
    )

    # Calculate a perpendicular vector to determine the arrowhead base.
    perp = (-math.sin(angle), math.cos(angle))

    # Determine the two base corners of the arrowhead triangle.
    left_base = (
        line_end[0] + (arrowhead_width / 2) * perp[0],
        line_end[1] + (arrowhead_width / 2) * perp[1]
    )
    right_base = (
        line_end[0] - (arrowhead_width / 2) * perp[0],
        line_end[1] - (arrowhead_width / 2) * perp[1]
    )

    # Draw the line (only if its length > 0).
    if line_length > 0:
        pygame.draw.line(surface, color, start, line_end, line_width)

    # Draw the arrowhead as a filled triangle.
    pygame.draw.polygon(surface, color, [tip, left_base, right_base])


def draw_arched_arrow2(surface, start, angle, total_length, radius, color,
                        arrowhead_length=0.2, arrowhead_width=10, line_width=2):
    """
    Draws an arched arrow along a circular arc with fixed radius.

    The arrow is drawn along a circle determined by a center computed from the
    starting point and the tangent direction 'angle'. The arc’s sweep (total_length)
    is measured in radians (max 2π – any longer value is clamped). A full arrow (line +
    arrowhead) is drawn along the arc. The arrowhead is a triangle with a fixed sweep 
    (arrowhead_length in radians) and a base width in pixels. If total_length is shorter 
    than arrowhead_length, then the arrowhead shrinks proportionally, and no separate line 
    is drawn.

    Negative total_length values are supported – they flip the arrow along the arc (i.e.
    they draw the arc on the opposite side of the tangent).

    Parameters:
      surface (pygame.Surface): Surface on which to draw.
      start (tuple): (x, y) starting point on the arc (which lies on the circle).
      angle (float): Tangent angle (in radians) at the starting point.
      total_length (float): Total arc sweep (in radians) for the arrow (line + arrowhead).
                            Maximum allowed is 2π. Can be negative.
      radius (float): Radius (in pixels) of the circular arc.
      color (tuple): Color of the arrow.
      arrowhead_length (float): Fixed arc sweep (in radians) for the arrowhead.
                                If total_length is less than this, the arrowhead is scaled down.
      arrowhead_width (float): Base width (in pixels) of the arrowhead triangle.
      line_width (int): Width of the arc “line” (in pixels).
    """
    #total_length = math.radians(total_length-90)

    # Handle negative total_length by storing a sign and flipping the arc direction.
    sign = 1
    if total_length < 0:
        sign = -1
        total_length = -total_length
    total_length = min(total_length, 2 * math.pi)  # Clamp to a full circle.

    # Determine the portions for the line and the arrowhead.
    if total_length > arrowhead_length:
        line_length = total_length - arrowhead_length
        current_arrowhead_length = arrowhead_length
    else:
        line_length = 0
        current_arrowhead_length = total_length
        # Scale the arrowhead width to maintain the angles.
        arrowhead_width *= (total_length / arrowhead_length)
    
    # Compute the circle center.
    # For a straight arrow, 'angle' is the tangent direction at 'start'.
    # Here we define the center to be to the left of the tangent when sign==1,
    # or to the right when sign==-1.
    cx = start[0] + sign * radius * (-math.sin(angle))
    cy = start[1] + sign * radius * ( math.cos(angle))
    center = (cx, cy)
    
    # Compute the polar angle (theta) at the start.
    theta_start = math.atan2(start[1] - cy, start[0] - cx)
    
    # Compute the angles for the end of the line and the arrow tip.
    theta_line_end = theta_start + sign * line_length
    theta_tip = theta_start + sign * total_length

    # Determine the positions on the circle.
    line_end = (cx + radius * math.cos(theta_line_end),
                cy + radius * math.sin(theta_line_end))
    tip = (cx + radius * math.cos(theta_tip),
           cy + radius * math.sin(theta_tip))
    
    # Draw the arc line if there is any line portion.
    if line_length > 0:
        # Choose a number of segments for a smooth arc.
        segments = max(2, int(abs(line_length) / (2 * math.pi) * 100))
        arc_points = []
        for i in range(segments + 1):
            theta = theta_start + sign * (line_length * i / segments)
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            arc_points.append((x, y))
        pygame.draw.lines(surface, color, False, arc_points, line_width)
    
    # Compute the tangent at the tip.
    # The derivative of (cos(theta), sin(theta)) is (-sin(theta), cos(theta)).
    # Multiply by sign so that the tangent points in the drawn direction.
    tangent = (sign * -math.sin(theta_tip), sign * math.cos(theta_tip))
    
    # Compute a perpendicular vector to the tangent.
    perp = (-tangent[1], tangent[0])
    
    # Determine the base points of the arrowhead triangle.
    # The base of the arrowhead is anchored at line_end.
    left_base = (line_end[0] + (arrowhead_width / 2) * perp[0],
                 line_end[1] + (arrowhead_width / 2) * perp[1])
    right_base = (line_end[0] - (arrowhead_width / 2) * perp[0],
                  line_end[1] - (arrowhead_width / 2) * perp[1])
    
    # Draw the arrowhead as a filled triangle.
    pygame.draw.polygon(surface, color, [tip, left_base, right_base])



def draw_curved_arrow(surface, center, r, angle, color,
                        thickness=2, arrowhead_length=10, arrowhead_width=10, num_points=30):
    """
    Draws a curved arrow along a circular arc.

    The arc is drawn along a circle of radius r centered at 'center'. The arrow always
    starts at the point directly above the center (i.e. at 270°) and sweeps an arc by the
    given 'angle' in degrees. For example, angle=90 will draw one-quarter of a circle.
    The arrowhead is drawn at the tip (end of the arc) and is oriented tangentially to
    the circle. Negative angles are also supported, causing the arc (and arrowhead) to be
    drawn in the opposite direction.

    The arrow is composed of two portions:
      1. The arc “line” portion, whose length is dynamic.
      2. The arrowhead (drawn as a triangle) with a fixed length along the arc, unless the
         total arc length is too short—in that case the arrowhead shrinks proportionally
         (and no line is drawn).

    Parameters:
      surface          - Pygame surface to draw on.
      center           - Tuple (x, y) for the center of the circle.
      r                - Radius of the circle.
      angle            - Total arc angle in degrees (max is 360). The arrow starts at the
                         point (center[0], center[1]-r) and extends by this angle.
                         Can be negative.
      color            - Color for the arrow (arc and arrowhead).
      thickness        - Line thickness for the arc.
      arrowhead_length - Fixed length (in pixels along the arc) for the arrowhead.
      arrowhead_width  - Width (in pixels) of the base of the arrowhead.
      num_points       - Number of points used to approximate the arc line.
    """
    # Do nothing if no arc is specified.
    if angle == 0:
        return

    # Convert total angle (in degrees) to radians.
    total_angle_rad = math.radians(angle)
    sign = 1 if angle > 0 else -1

    # Total arc length in pixels along the circle.
    total_arc_length = r * abs(total_angle_rad)

    # Decide how to split the arc into line and arrowhead.
    if total_arc_length > arrowhead_length:
        line_arc_length = total_arc_length - arrowhead_length
        current_arrowhead_length = arrowhead_length
    else:
        # The arrow is too short: no separate line and the arrowhead shrinks.
        line_arc_length = 0
        current_arrowhead_length = total_arc_length
        # Scale arrowhead width to maintain the arrowhead’s angles.
        arrowhead_width *= (total_arc_length / arrowhead_length)

    # Convert the line and arrowhead lengths (in pixels) back into arc angles (radians).
    line_arc_angle = line_arc_length / r
    arrowhead_arc_angle = current_arrowhead_length / r

    # The arc always starts at 270° (i.e. "up" from the center).
    start_angle = math.radians(270)
    # Angle at the end of the line (if any) and at the tip of the arrow.
    line_end_angle = start_angle + sign * line_arc_angle
    tip_angle = start_angle + sign * (line_arc_angle + arrowhead_arc_angle)

    # If there is a line portion, generate and draw the arc points.
    if line_arc_length > 0:
        arc_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            theta = start_angle + sign * (line_arc_angle * t)
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            arc_points.append((x, y))
        pygame.draw.lines(surface, color, False, arc_points, thickness)

    # Compute the tip of the arrow.
    tip = (center[0] + r * math.cos(tip_angle),
           center[1] + r * math.sin(tip_angle))

    # Determine the tangent at the tip.
    # For a circle x = center_x + r*cos(theta), y = center_y + r*sin(theta),
    # the derivative is (-r*sin(theta), r*cos(theta)).
    tangent = (sign * -math.sin(tip_angle), sign * math.cos(tip_angle))
    # (tangent is unit length because sin^2+cos^2 = 1)

    # Find the base center of the arrowhead by stepping back from the tip along the tangent.
    base_center = (tip[0] - current_arrowhead_length * tangent[0],
                   tip[1] - current_arrowhead_length * tangent[1])

    # Compute a perpendicular vector to the tangent (to define the arrowhead’s base width).
    perp = (tangent[1], -tangent[0])

    # Determine the two base corners of the arrowhead.
    left_corner = (base_center[0] + (arrowhead_width / 2) * perp[0],
                   base_center[1] + (arrowhead_width / 2) * perp[1])
    right_corner = (base_center[0] - (arrowhead_width / 2) * perp[0],
                    base_center[1] - (arrowhead_width / 2) * perp[1])

    # Draw the arrowhead as a filled polygon.
    pygame.draw.polygon(surface, color, [tip, left_corner, right_corner])



def draw_legend_box(surface, legend_items, font,
                    box_color=(50, 50, 50, 180), text_color=(255, 255, 255),
                    marker_size=12, spacing=5, padding=10, margin=10):
    """
    Draws a legends box at the bottom right corner of the given Pygame surface.

    Parameters:
      surface:         Pygame surface to draw on.
      legend_items:    List of tuples (marker_color, label) where marker_color is a
                       (R, G, B) tuple and label is a string.
      font:            A pygame.font.Font instance to render the text.
      box_color:       Background color of the legend box. Can include an alpha value.
      text_color:      Color for the text.
      marker_size:     Size (width and height in pixels) of the colored marker.
      spacing:         Horizontal spacing between the marker and the text.
      padding:         Internal padding within the legend box.
      margin:          Margin between the legend box and the screen edges.
    """
    # Render each legend item's text.
    rendered_items = []
    for marker_color, label in legend_items:
        text_surface = font.render(label, True, text_color)
        rendered_items.append((marker_color, text_surface))
    
    # Calculate the width and height of the legend box.
    max_text_width = max(text.get_width() for _, text in rendered_items)
    line_height = max(text.get_height() for _, text in rendered_items)
    vertical_spacing = 5  # spacing between each legend item

    # Total height is the sum of all item heights plus spacing, with extra padding.
    total_height = len(rendered_items) * line_height + (len(rendered_items) - 1) * vertical_spacing
    box_width = padding * 2 + marker_size + spacing + max_text_width
    box_height = padding * 2 + total_height

    # Determine the box position at bottom right.
    surface_rect = surface.get_rect()
    box_rect = pygame.Rect(0, 0, box_width, box_height)
    box_rect.bottomright = (surface_rect.width - margin, surface_rect.height - margin)

    # Create a temporary surface for the legend box with per-pixel alpha for transparency.
    legend_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
    legend_surface.fill(box_color)

    # Blit each legend item (marker + text) onto the legend_surface.
    y_offset = padding
    for marker_color, text_surface in rendered_items:
        # Draw the marker (a filled rectangle)
        marker_rect = pygame.Rect(padding, y_offset + (line_height - marker_size) // 2,
                                  marker_size, marker_size)
        pygame.draw.rect(legend_surface, marker_color, marker_rect)

        # Draw the text next to the marker.
        text_pos = (padding + marker_size + spacing, y_offset)
        legend_surface.blit(text_surface, text_pos)

        y_offset += line_height + vertical_spacing

    # Finally, blit the legend_surface onto the main surface.
    surface.blit(legend_surface, box_rect.topleft)

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
    arc_angle = 90     # Try positive 90 or negative -90 or even 360.
    arrow_color = (255, 0, 0)  # Red
    # Create a font for the legend text.
    font = pygame.font.SysFont('DejaVu Sans', 12) 

    # Define your legend items: (marker_color, label)
    legend_items = [
        ((255, 102, 0), "Desired- pose/ total thrust force/ total trust moment"),
        ((255, 205, 128), "Desired thrust force"),
        ((0, 255, 123), "Surge/ sway/ angular velocity"),
    ]


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # White background

        draw_arrow(screen, center, -45, -50, (250,0,0))
        draw_arched_arrow2(screen,center,math.pi,arc_angle/180*math.pi,radius,(250,0,0))
        draw_curved_arrow(screen,center,radius,arc_angle,arrow_color)
        draw_legend_box(screen, legend_items, font)

        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
