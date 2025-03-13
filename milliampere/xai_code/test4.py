import pygame
from enum import Enum
import random
import math
import numpy as np

class Colors(Enum):
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    GRAY = (150,150,150)
    BLUE = (0, 122, 255)
    YELLOW = (255, 102, 0)
    SHAP_RED = (0, 255, 123)
    LIGHT_BLUE = (209, 237, 255)
    LIGHT_GRAY = (240, 240, 240)
    LIGHT_YELLOW = (255, 205, 128)


class Window:
    def __init__(self, screen, title="Window", width=450, height=450, x=0, y=0):
        self.screen = screen
        self.title = title
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont('DejaVu Sans', 11) 

    def render_window(self):
        self.surface.fill(Colors.WHITE.value)
        
    def render_window_titel(self):
        text_surface = self.font.render(self.title, True, Colors.GRAY.value)
        text_rect = text_surface.get_rect()
        text_rect.topright = (self.x + self.width - 5, self.y + 5)
        self.screen.blit(text_surface, text_rect)
    

class VesselRender(Window):
    
    # Render constants
    #SCALE = 50/2
    #FRAME = 10 # meters

    VESSEL_LENGTH = 5.06 # meters
    VESSEL_WIDTH = 2.86 # meters
    VESSEL_CORNER = 0.5 # meters

    
      

    def __init__(self, screen, title, x, y, scale):
        super().__init__(screen, title=title, x=x, y=y)
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.SCALE = scale

            # Agent
        self.AGENT_W = self.VESSEL_WIDTH * self.SCALE
        self.AGENT_H = self.VESSEL_LENGTH * self.SCALE
        self.AGENT_C = self.VESSEL_CORNER * self.SCALE
        self.AGENT_TRIANGLE_SIZE = 0.4 * self.SCALE
        self.AGENT_CIRCLE_RADIUS = 3  

    def draw_arrow2(self, angle, arrow_length, color, x=0, y=0, arrow_width=10, arrowhead_angle=math.pi/6, pct_explained=None, width=4):
        angle_rad = math.radians(-angle+90)

        if arrow_length < 0:
            arrow_length = abs(arrow_length)
            angle_rad += math.pi

        end_x = self.center_x + x + arrow_length * math.cos(angle_rad)
        end_y = self.center_y + y - arrow_length * math.sin(angle_rad)

        left_x = end_x - arrow_width * math.cos(angle_rad - arrowhead_angle)
        left_y = end_y + arrow_width * math.sin(angle_rad - arrowhead_angle)
        right_x = end_x - arrow_width * math.cos(angle_rad + arrowhead_angle)
        right_y = end_y + arrow_width * math.sin(angle_rad + arrowhead_angle)

        line_end_x = self.center_x + x + (arrow_length - arrow_width/2) * math.cos(angle_rad)
        line_end_y = self.center_y + y - (arrow_length - arrow_width/2) * math.sin(angle_rad)

        pygame.draw.line(self.surface, color, (self.center_x + x, self.center_y + y), (line_end_x, line_end_y), width=width)
        pygame.draw.polygon(self.surface, color, [(end_x, end_y), (left_x, left_y), (right_x, right_y)])

        if pct_explained:
            antialias = True
            color = Colors.SHAP_RED.value
            text = self.font.render(f"{pct_explained:.2f} %", antialias, color)
            self.surface.blit(text, ((self.center_x + x + end_x) // 2 - text.get_width() // 2, (self.center_y + y + end_y) // 2 - text.get_height() // 2))

    def draw_arrow(self, angle, arrow_length, color, x=0, y=0, arrowhead_length=10, arrowhead_width=10, line_width=2, pct_explained=None):
        x = self.center_x + x
        y = self.center_y + y
        angle = math.radians(angle-90)
        
        if arrow_length < 0:
            arrow_length = - arrow_length
            angle += math.pi

        if arrow_length > arrowhead_length:
            line_length = arrow_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_length = 0
            current_arrowhead_length = arrow_length
            arrowhead_width *= (arrow_length / arrowhead_length) 
        
        line_end = (
            x + line_length * math.cos(angle),
            y + line_length * math.sin(angle)
        )

        tip = (
            line_end[0] + current_arrowhead_length * math.cos(angle),
            line_end[1] + current_arrowhead_length * math.sin(angle)
        )

        perp = (-math.sin(angle), math.cos(angle))

        left_base = (
            line_end[0] + (arrowhead_width / 2) * perp[0],
            line_end[1] + (arrowhead_width / 2) * perp[1]
        )

        right_base = (
            line_end[0] - (arrowhead_width / 2) * perp[0],
            line_end[1] - (arrowhead_width / 2) * perp[1]
        )

        if line_length > 0:
            pygame.draw.line(self.surface, color, (x, y), line_end, line_width)

        pygame.draw.polygon(self.surface, color, [tip, left_base, right_base])



    def draw_explanation(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, pct_explained, explain, color=Colors.SHAP_RED.value, line_end_len=5, text_offset=10):
        if explain == 4 or explain == 5:
            self.draw_arrow(vel_angle, vel_magnitude * self.SCALE, color, pct_explained=pct_explained)
        elif explain == 1:
            target_screen_x = self.center_x + x_tilde * self.SCALE
            pygame.draw.line(self.surface, Colors.SHAP_RED.value, (self.center_x, self.center_y), (target_screen_x, self.center_y), width=2)
            pygame.draw.line(self.surface, Colors.SHAP_RED.value, (target_screen_x, self.center_y - line_end_len), (target_screen_x, self.center_y + line_end_len), width=2)
            antialias = True
            text = self.font.render(f"{pct_explained:.2f} %", antialias, color)
            self.surface.blit(text, ((self.center_x + target_screen_x) // 2 - text.get_width() // 2, self.center_y + text_offset))
        elif explain == 2:
            target_screen_y = self.center_y - y_tilde * self.SCALE
            pygame.draw.line(self.surface, Colors.SHAP_RED.value, (self.center_x, self.center_y), (self.center_x, target_screen_y), width=2)
            pygame.draw.line(self.surface, Colors.SHAP_RED.value, (self.center_x - line_end_len, target_screen_y), (self.center_x + line_end_len, target_screen_y), width=2)
            antialias = True
            text = self.font.render(f"{pct_explained:.2f} %", antialias, color)
            self.surface.blit(text, (self.center_x + text_offset, (self.center_y + target_screen_y) // 2 - text.get_width() // 2))
        else:
            print(f"Explain does not exist!")


    def draw_action(self, n_d, alpha_d):
        self.draw_arrow(alpha_d, n_d * self.SCALE, Colors.YELLOW.value)
    
    def draw_vessel(self, center_x, center_y, heading, color=Colors.YELLOW.value, agent=True, ned=False):
        if ned:
            center_x, center_y = self.rotate_body2ned(heading, center_x, center_y)
        target_screen_x = self.center_x + center_x * self.SCALE
        target_screen_y = self.center_y - center_y * self.SCALE

        points = self.make_vessel_shape(target_screen_x, target_screen_y)
        triangle_points = self.make_triangle(target_screen_x, target_screen_y)
        
        if heading != 0:
            points = [self.rotate_points(target_screen_x, target_screen_y, px, py, heading) for px,py in points]
            triangle_points = [self.rotate_points(target_screen_x, target_screen_y, px, py, heading) for px,py in triangle_points]

        if agent:
            pygame.draw.polygon(self.surface, Colors.BLUE.value, points)
            pygame.draw.polygon(self.surface, Colors.BLACK.value, triangle_points)
            pygame.draw.circle(self.surface, Colors.BLACK.value, (target_screen_x, target_screen_y), self.AGENT_CIRCLE_RADIUS)
        else:
            for i in range(len(points)):
                self.draw_dashed_line(points[i], points[(i + 1) % len(points)], color)
            
            for i in range(len(triangle_points)):
                self.draw_dashed_line(triangle_points[i], triangle_points[(i + 1) % len(triangle_points)], color)
            
            pygame.draw.circle(self.surface, color, (target_screen_x, target_screen_y), self.AGENT_CIRCLE_RADIUS)

    def rotate_body2ned(self, psi, x, y):
        psi = math.radians(psi)
        new_x = x * math.cos(psi) + y * math.sin(psi)
        new_y = - x * math.sin(psi) + y * math.cos(psi)
        return new_x, new_y

    def make_vessel_shape(self, center_x, center_y):
        x = center_x - self.AGENT_W // 2
        y = center_y - self.AGENT_H // 2
        
        points = [
            (x + self.AGENT_C, y),
            (x + self.AGENT_W - self.AGENT_C, y),
            (x + self.AGENT_W, y + self.AGENT_C),
            (x + self.AGENT_W, y + self.AGENT_H - self.AGENT_C),
            (x + self.AGENT_W - self.AGENT_C, y + self.AGENT_H),
            (x + self.AGENT_C, y + self.AGENT_H),
            (x, y + self.AGENT_H - self.AGENT_C),
            (x, y + self.AGENT_C)
        ]
        return points
    
    def make_triangle(self, center_x, center_y):
        y = center_y - self.AGENT_H // 2
        triangle_points = [
            (center_x, y),
            (center_x - self.AGENT_TRIANGLE_SIZE // 2, y + self.AGENT_TRIANGLE_SIZE),
            (center_x + self.AGENT_TRIANGLE_SIZE // 2, y + self.AGENT_TRIANGLE_SIZE)
        ]
        return triangle_points
    
    def draw_dashed_line(self, start, end, color, dash_length=10, gap_length=5):
        x1, y1 = start
        x2, y2 = end
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        dashes = int(length // (dash_length + gap_length))
        for i in range(dashes):
            t1 = i / dashes
            t2 = (i + 0.5) / dashes
            segment_start = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
            segment_end = (x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2)
            pygame.draw.line(self.surface, color, segment_start, segment_end, 2)

    def rotate_points(self, center_x, center_y, px, py, heading):
        rad = math.radians(heading)
        dx, dy = px - center_x, py - center_y
        new_x = center_x + (dx * math.cos(rad) - dy * math.sin(rad))
        new_y = center_y + (dx * math.sin(rad) + dy * math.cos(rad))
        return new_x, new_y
    
class BodyRender(VesselRender):
    def __init__(self, screen, x, y, title="BodyRender"):
        super().__init__(screen, title, x, y, scale=50)
        Lx = 0.8
        Ly = 1.8
        self.thruster_positions = np.array([[-Lx, -Ly],
                                    [Lx, -Ly],
                                    [Lx, Ly],
                                    [-Lx, Ly]])
        
        self.legend_items = [
            ((255, 102, 0), "Desired {'total thrust force', 'total trust moment', 'pose'}"),
            ((255, 205, 128), "Desired thrust force"),
            ((0, 255, 123), "{'surge', 'sway', 'angular'} velocity"),
        ]

    
    def calculate_total_moment(self, vectors):
        """
        Calculates the total moment about the origin (0,0) given thruster positions
        and corresponding force vectors.

        Parameters:
            thruster_positions: An array of shape (4,2) with positions [[x1, y1], ...].
            vectors: A list of 4 tuples, where each tuple is (n, theta) with:
                    - n: force magnitude
                    - theta: force angle in radians

        Returns:
            total_moment: The sum of moments (torques) about (0,0).
        """
        total_moment = 0.0
        i = 0
        for (x, y), (n, theta) in zip(self.thruster_positions, vectors):
            # Compute force components
            F_x = n * math.cos(theta)
            F_y = n * math.sin(theta)

            # Moment (torque) is the cross product r x F in 2D:
            moment = - x*self.SCALE * F_x - y*self.SCALE * F_y
            total_moment += moment
        return total_moment

    def _draw_vel(self, u_hat, v_hat):
        self.draw_arrow(0,u_hat*self.SCALE,Colors.SHAP_RED.value)
        self.draw_arrow(90,v_hat*self.SCALE,Colors.SHAP_RED.value)

    
    def draw_curved_arrow2(self, surface, center, r, angle, color,
                            thickness=2, arrowhead_length=10, arrowhead_width=15, num_points=30):
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

    def draw_curved_arrow(self, surface, center, r, angle, color,
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
        

    def draw_legend_box(self, surface, legend_items, font,
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

    def render(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain, u_hat, v_hat,r_hat, vectors):
        self.render_window()
        self.surface.fill(Colors.LIGHT_BLUE.value)
        
        self.draw_vessel(x_tilde, y_tilde, psi_tilde, agent=False)
        self.draw_vessel(0, 0, 0)
        #self.draw_explanation(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, pct_explained, explain)
        for i,(x,y) in enumerate(self.thruster_positions):
            #self.draw_arrow(-135,3.5*self.SCALE,color=Colors.GRAY.value, x=x*self.SCALE,y=y*self.SCALE)
            self.draw_arrow(180*vectors[i][1]/math.pi,vectors[i][0]*self.SCALE,color=Colors.LIGHT_YELLOW.value, x=x*self.SCALE,y=y*self.SCALE, line_width=2)
            pygame.draw.line(self.surface, Colors.LIGHT_YELLOW.value, (self.center_x+x*self.SCALE-10*math.cos(vectors[i][1]), self.center_y+y*self.SCALE-10*math.sin(vectors[i][1])),(self.center_x+x*self.SCALE+10*math.cos(vectors[i][1]), self.center_y+y*self.SCALE+10*math.sin(vectors[i][1])), width=2)
            pygame.draw.circle(self.surface, Colors.BLACK.value, (self.center_x+x*self.SCALE, self.center_y+y*self.SCALE), self.AGENT_CIRCLE_RADIUS)
        total_moment = self.calculate_total_moment(vectors)
        self.draw_curved_arrow(self.surface,(self.center_x,self.center_y),140/5,total_moment,Colors.YELLOW.value)
        self.draw_action(n_d, alpha_d)
        self._draw_vel(u_hat, v_hat)
        self.draw_curved_arrow(self.surface,(self.center_x,self.center_y),140/5,r_hat,Colors.SHAP_RED.value)
        pygame.draw.line(self.surface, Colors.BLACK.value, (self.center_x, self.center_y), (self.center_x, self.center_y-140/5),width=1)
        pygame.draw.circle(self.surface, Colors.BLACK.value, (self.center_x, self.center_y), self.AGENT_CIRCLE_RADIUS)
        self.draw_legend_box(self.surface, self.legend_items, self.font)
        self.screen.blit(self.surface, (self.x, self.y))
        self.render_window_titel()

class NedRender(VesselRender):
    def __init__(self, screen, x, y, title="NedRender",):
        super().__init__(screen, title, x, y, scale=50/2)

    def draw_ned_axes_box(self, surface, box_top_left=(20, 20), axis_length=50, color=(74,74,74)):
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
        arrow_head_length = 8  # length along the axis
        arrow_head_width = 4    # half-width of the arrow head
        
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
        
    def render(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain):
        self.render_window()
        self.surface.fill(Colors.LIGHT_BLUE.value)

        
        self.draw_vessel(0, 0, 45, agent=False)
        self.draw_vessel(-x_tilde, -y_tilde, 45-psi_tilde, ned=True)
        self.draw_ned_axes_box(self.surface)
    
        self.screen.blit(self.surface, (self.x, self.y))
        self.render_window_titel()

class ShapRender(Window):
    def __init__(self, screen, x, y, title="ShapRender"):
        super().__init__(screen, title=title, x=x, y=y, width=475)

        # Settings
        self.widow_padding = 200
        self.axis_padding = 125
        self.num_bars = 14
        self.max_value = 1.0   # Maximum value for each segment
        self.increment = 1.0

        self.bar_height = (self.height - self.widow_padding) // self.num_bars
        self.bar_gap = 5
        # Placeholder (not used now since shap_values is provided externally as 4 lists)
        self.bar_values = [round(random.uniform(0.00, self.max_value), 2) for _ in range(self.num_bars)]
        self.max_bar_length = self.width - self.widow_padding

        self.feature_names = [
            'x̃ᵇₜ [m]',  # x̃ᵇₜ
            'ỹᵇₜ [m]',  # ỹᵇₜ
            'ϕₜ [°]',    # ϕₜ
            'ûₜ [m/s]',  # ûₜ
            'v̂ₜ [m/s]',  # v̂ₜ
            'r̂ₜ [°/s]', # r̂ₜ
            'n_{d₁,ₜ₋₁} [RPM]',
            'n_{d₂,ₜ₋₁} [RPM]',
            'n_{d₃,ₜ₋₁} [RPM]',
            'n_{d₄,ₜ₋₁} [RPM]',
            'α_{d₁,ₜ₋₁} [°]',
            'α_{d₂,ₜ₋₁} [°]',
            'α_{d₃,ₜ₋₁} [°]',
            'α_{d₄,ₜ₋₁} [°]'
        ]

        self.legend_items = [
            ((87, 120, 164), "n_{d₁,ₜ} [RPM]"),
            ((228, 148, 68), "n_{d₂,ₜ} [RPM]"),
            ((106, 159, 88), "n_{d₃,ₜ} [RPM]"),
            ((209, 97, 93), "n_{d₄,ₜ} [RPM]")
        ]

    def _draw_bars(self, shap_values):
        """
        Draws bars based on shap_values, which is expected to be a list of 4 lists.
        Each sub-list should contain self.num_bars elements corresponding to the values
        for the blue, red, green, and pink bars, respectively.
        """
        # Each segment gets an equal share of the available horizontal length.
        segment_max_length = self.max_bar_length / 4

        # Define colors for each segment in order: blue, red, green, pink.
        colors = [
            (87, 120, 164),
            (228, 148, 68),      # Red
            (106, 159, 88),      # Green
            (209, 97, 93)   # Pink
        ]

        for i in range(self.num_bars):
            bar_y = 50 + i * (self.bar_height + self.bar_gap)
            current_x = self.axis_padding

            # Loop through each segment and draw its bar.
            for seg in range(4):
                # Get the value for this bar and segment.
                val = abs(shap_values[seg][i])
                seg_length = int((val / self.max_value) * segment_max_length)
                pygame.draw.rect(self.surface, colors[seg],
                                 (current_x, bar_y, seg_length, self.bar_height))
                current_x += seg_length

            # Draw the feature label to the left of the bars.
            label_surface = self.font.render(self.feature_names[i], True, Colors.BLACK.value)
            label_rect = label_surface.get_rect()
            label_rect.topleft = (self.axis_padding - label_rect.width - 10,
                                  bar_y + (self.bar_height - label_rect.height) // 2)
            self.surface.blit(label_surface, label_rect)

    def _draw_axis(self):
        """
        Draws the Y- and X-axes along with tick marks.
        The X-axis now spans a total value of 4 * self.max_value.
        """
        # Total maximum value across all four segments.
        total_max_value = 4 * self.max_value
        axis = 50 + self.num_bars * (self.bar_height + self.bar_gap) + 30

        pygame.draw.line(self.surface, Colors.BLACK.value,
                         (self.axis_padding, 50), (self.axis_padding, axis), 2)  # Y-axis
        pygame.draw.line(self.surface, Colors.BLACK.value,
                         (self.axis_padding, axis),
                         (self.axis_padding + self.max_bar_length, axis), 2)  # X-axis

        num_ticks = int(total_max_value / self.increment) + 1
        for tick in range(num_ticks):
            tick_value = tick * self.increment
            tick_x = self.axis_padding + int((tick_value / total_max_value) * self.max_bar_length)
            pygame.draw.line(self.surface, Colors.BLACK.value,
                             (tick_x, axis - 5), (tick_x, axis + 5), 2)
            tick_label = self.font.render(f"{tick_value:.2f}", True, Colors.BLACK.value)
            self.surface.blit(tick_label, (tick_x - 10, axis + 10))

        bottom_text = "Sum of |SHAP values|"
        bottom_text_surface = self.font.render(bottom_text, True, Colors.BLACK.value)
        bottom_text_rect = bottom_text_surface.get_rect()
        bottom_text_rect.center = (self.width // 2, axis + 40)
        self.surface.blit(bottom_text_surface, bottom_text_rect)

    def draw_legend_box(self, surface, legend_items, font,
                        box_color=(200, 200, 200, 180), text_color=(0, 0, 0),
                        marker_size=12, spacing=5, padding=10, margin=10):
        """
        Draws a legends box at the bottom right corner of the given Pygame surface.

        Parameters:
          - surface: Pygame surface to draw on.
          - legend_items: List of tuples (marker_color, label) where marker_color is a (R, G, B) tuple.
          - font: A pygame.font.Font instance for rendering text.
          - box_color: Background color of the legend box (can include alpha).
          - text_color: Color for the text.
          - marker_size: Size (width and height in pixels) of the marker.
          - spacing: Horizontal spacing between marker and text.
          - padding: Internal padding within the legend box.
          - margin: Margin between the legend box and the screen edges.
        """
        # Render each legend item's text.
        rendered_items = []
        for marker_color, label in legend_items:
            text_surface = font.render(label, True, text_color)
            rendered_items.append((marker_color, text_surface))
        
        # Calculate dimensions for the legend box.
        max_text_width = max(text.get_width() for _, text in rendered_items)
        line_height = max(text.get_height() for _, text in rendered_items)
        vertical_spacing = 5

        total_height = len(rendered_items) * line_height + (len(rendered_items) - 1) * vertical_spacing
        box_width = padding * 2 + marker_size + spacing + max_text_width
        box_height = padding * 2 + total_height

        surface_rect = surface.get_rect()
        box_rect = pygame.Rect(0, 0, box_width, box_height)
        box_rect.bottomright = (surface_rect.width - margin, surface_rect.height - margin - 60)

        # Create a temporary surface with per-pixel alpha.
        legend_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        legend_surface.fill(box_color)

        # Blit each legend item onto the legend_surface.
        y_offset = padding
        for marker_color, text_surface in rendered_items:
            marker_rect = pygame.Rect(padding, y_offset + (line_height - marker_size) // 2,
                                      marker_size, marker_size)
            pygame.draw.rect(legend_surface, marker_color, marker_rect)

            text_pos = (padding + marker_size + spacing, y_offset)
            legend_surface.blit(text_surface, text_pos)

            y_offset += line_height + vertical_spacing

        surface.blit(legend_surface, box_rect.topleft)

    def render(self, shap_values,legend_items):
        """
        Expects shap_values to be a list of four lists (blue, red, green, pink),
        each containing self.num_bars values.
        """
        self.render_window()
        
        self._draw_bars(shap_values)
        self._draw_axis()
        self.draw_legend_box(self.surface, legend_items, self.font)

        self.screen.blit(self.surface, (self.x, self.y))
        self.render_window_titel()




class RenderExplaination():
    def __init__(self, screen_width=1000, screen_height=975, title="Explainations"):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        
        self.body_window = BodyRender(self.screen, x=25, y=25,title="BODY-frame")
        self.ned_window = NedRender(self.screen, x=25, y=500, title="NED-frame")
        self.shap_window_top = ShapRender(self.screen, title="SHAP values RPM", x=500, y=25)
        self.shap_window_bottom = ShapRender(self.screen, title="SHAP values azimuth angles", x=500, y=500)
        self.legend_items1 = [
            ((87, 120, 164), "n_{d₁,ₜ} [RPM]"),
            ((228, 148, 68), "n_{d₂,ₜ} [RPM]"),
            ((106, 159, 88), "n_{d₃,ₜ} [RPM]"),
            ((209, 97, 93), "n_{d₄,ₜ} [RPM]")
        ]
        self.legend_items2 = [
            ((87, 120, 164), "α_{d₁,ₜ} [°]"),
            ((228, 148, 68), "α_{d₂,ₜ} [°]"),
            ((106, 159, 88), "α_{d₃,ₜ} [°]"),
            ((209, 97, 93), "α_{d₄,ₜ} [°]")
        ]

    def render_frame(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain, shap_values, shap_values2, u_hat, v_hat, r_hat, vectors):
        # Fill the main screen
        self.screen.fill(Colors.LIGHT_GRAY.value)

        # Render each window
        self.body_window.render(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain, u_hat, v_hat, r_hat, vectors)
        self.ned_window.render(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain)
        self.shap_window_top.render(shap_values, self.legend_items1)
        self.shap_window_bottom.render(shap_values2, self.legend_items2)

        # Update the display
        pygame.display.flip()   



# SCALE = 50
# center_x, center_y = 300

# R = np.array([0, SCALE],
#              [-SCALE, 0])

# T = np.array([center_y],[center_x])

# def trans_cord_2_px(cord):
#     return R @ cord + T


# pygame.draw.circle(surface,color,trans_cord_2_px((x,y)),3)