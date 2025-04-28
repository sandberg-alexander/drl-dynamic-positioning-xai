import pygame
import numpy as np
from enum import Enum
import math

###########################################################################################################


class Color(Enum):
    WHITE = (255,255,255)
    GRAY = (150,150,150)
    BLACK = (0,0,0)
    GREEN = (0,255,0)
    PURPLE = (255,0,255)
    RED = (255,0,0)
    ORANGE = (255,128,0)
    
    SCREEN_COLOR = (240,240,240)

    TABLEAU_BLUE = (87,120,164)
    TABLEAU_ORANGE = (228,148,68)
    TABLEAU_GREEN = (106,159,88)
    TABLEAU_RED = (209,97,93)
    
    LEGEND_BOX = (200,200,200,180)
    
    AGENT_BLACK = (0,0,0,0)
    AGENT_BLUE = (0,122,255,220)
    DESIRED_YELLOW = (255,102,0)
    DESIRED_LIGHT_YELLOW = (255,205,128)
    OCEAN_BLUE = (209,237,255)
    VELOCITY_GREEN = (0,255,123)

    LIGHT_RED = (255, 171, 171)


###########################################################################################################


class Window():
    def __init__(self, screen, window_pos, title, window_width, window_height):
        # private attributes
        self._screen = screen
        self._window_pos = window_pos
        self._title = title
        self._title_offset = 5 # pixels
        
        # public attributes
        self.window_width = window_width
        self.window_height = window_height
        self.window_font = pygame.font.SysFont('DejaVu Sans', 11)
        self.antialias = True
        self.surface = pygame.Surface((self.window_width, self.window_height))
    
    def render_surface(self):
        self.surface.fill(Color.WHITE.value)
    
    def render_surface_title(self):
        text_surface = self.window_font.render(self._title, self.antialias, Color.GRAY.value)
        text_rect = text_surface.get_rect()
        text_rect.topright = (self.window_width - self._title_offset, self._title_offset)
        self.surface.blit(text_surface, text_rect)

    def render_window(self):
        self._screen.blit(self.surface, self._window_pos)


    def draw_arrow(self, angle, arrow_length, color, start_pos=np.zeros(2), arrowhead_length=10, arrowhead_width=10, line_width=2):
        
        if arrow_length < 0:
            arrow_length = - arrow_length
            angle += np.pi
        
        if arrow_length > arrowhead_length:
            line_length = arrow_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_length = 0
            current_arrowhead_length = arrow_length
            arrowhead_width *= (arrow_length / arrowhead_length)

        rotate = np.array([np.cos(angle), np.sin(angle)])

        line_end = start_pos + line_length * rotate
        arrow_tip = line_end + current_arrowhead_length * rotate

        left_base = line_end + arrowhead_width/2 * np.array([-rotate[1], rotate[0]])
        right_base = line_end - arrowhead_width/2 * np.array([-rotate[1], rotate[0]])

        if line_length > 0:
            pygame.draw.line(self.surface, color, start_pos, line_end, line_width)
        pygame.draw.polygon(self.surface, color, [arrow_tip, left_base, right_base])

    
    def create_legend(self, legend_items, font,
                    box_color=Color.LEGEND_BOX.value, text_color=Color.BLACK.value,
                    marker_size=12, spacing=5, padding=10, margin=10, padding_bottom=60):

        rendered_items = []
        for marker_color, label in legend_items:
            text_surface = font.render(label, self.antialias, text_color)
            rendered_items.append((marker_color, text_surface))
        
        max_text_width = max(text.get_width() for _, text in rendered_items)
        line_height = max(text.get_height() for _, text in rendered_items)
        
        total_height = len(rendered_items) * line_height + (len(rendered_items) - 1) * spacing
        box_width = padding * 2 + marker_size + spacing + max_text_width
        box_height = padding * 2 + total_height

        surface_rect = self.surface.get_rect()
        box_rect = pygame.Rect(0,0,box_width,box_height)
        box_rect.bottomright = surface_rect.width - margin, surface_rect.height - margin - padding_bottom

        legend_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        legend_surface.fill(box_color)

        y_offset = padding
        for marker_color, text_surface, in rendered_items:
            marker_rect = pygame.Rect(padding, y_offset + (line_height - marker_size) // 2,
                                      marker_size, marker_size)
            pygame.draw.rect(legend_surface, marker_color, marker_rect)

            text_pos = (padding + marker_size + spacing, y_offset)
            legend_surface.blit(text_surface, text_pos)

            y_offset += line_height + spacing

        return legend_surface, box_rect.topleft

    def draw_legend(self, legend_surface, box_pos):
        self.surface.blit(legend_surface, box_pos)
    

###########################################################################################################


class Utilities():
    def __init__(self):
        pass
    
    def _draw_arrow(self, angle, arrow_length, color, start_pos=np.zeros(2), arrowhead_length=10, arrowhead_width=10, line_width=2,
                    show_measurement=False,label=None,font=None):
        
        if arrow_length < 0:
            arrow_length = - arrow_length
            angle += np.pi
        
        if arrow_length > arrowhead_length:
            line_length = arrow_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_length = 0
            current_arrowhead_length = arrow_length
            arrowhead_width *= (arrow_length / arrowhead_length)

        rotate = np.array([np.cos(angle), np.sin(angle)])

        line_end = start_pos + line_length * rotate
        arrow_tip = line_end + current_arrowhead_length * rotate

        left_base = line_end + arrowhead_width/2 * np.array([-rotate[1], rotate[0]])
        right_base = line_end - arrowhead_width/2 * np.array([-rotate[1], rotate[0]])

        if line_length > 0:
            pygame.draw.line(self.surface, color, start_pos, line_end, line_width)
        pygame.draw.polygon(self.surface, color, [arrow_tip, left_base, right_base])

        # Display measurement text
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont('Arial', 18)
            
            text = label
            text_surface = font.render(text, True, color)
            
            # Position text above the middle of the line
            text_pos_x = (start_pos[0] + arrow_tip[0]) / 2 - text_surface.get_width() / 2
            text_pos_y = (start_pos[1] + arrow_tip[1]) / 2 - text_surface.get_height() - 5
            
            # Adjust text position if line is vertical
            if np.pi/2 - np.pi/16 < abs(angle) < np.pi/2 + np.pi/16:
                text_pos_x = (start_pos[0] + arrow_tip[0]) / 2 + 10
                text_pos_y = (start_pos[1] + arrow_tip[1]) / 2 - text_surface.get_height() / 2
            
            self.surface.blit(text_surface, (text_pos_x, text_pos_y))


    def _draw_curved_arrow(self, angle, center, color, radius=10,
                           line_width=2, arrowhead_length=10, arrowhead_widht=10, num_points=30,
                           show_measurement=False, font=None, label=None, ):
        
        if angle == 0:
            return
        
        sign = 1 if angle > 0 else -1

        total_arc_length = radius * abs(angle)

        if total_arc_length > arrowhead_length:
            line_arc_length = total_arc_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_arc_length = 0
            current_arrowhead_length = total_arc_length
            arrowhead_widht *= (total_arc_length / arrowhead_length)

        line_arc_angle = line_arc_length / radius
        arrowhead_arc_angle = current_arrowhead_length / radius

        start_angle = 3*np.pi/2
        tip_angle = start_angle + sign * (line_arc_angle + arrowhead_arc_angle)

        if line_arc_length > 0:
            arc_points = [None] * num_points
            for i in range(num_points):
                t = i / (num_points -1)
                theta = start_angle + sign * (line_arc_angle * t)
                pos = center + radius * np.array([np.cos(theta), np.sin(theta)])
                arc_points[i] = pos
            pygame.draw.lines(self.surface, color, False, arc_points, line_width)
        
        rotate = np.array([np.cos(tip_angle), np.sin(tip_angle)])
        tip = center + radius * rotate
        base_center = tip - current_arrowhead_length * sign * np.array([-rotate[1], rotate[0]])
        left_corner = base_center + arrowhead_widht/2 * sign * rotate
        right_corner = base_center - arrowhead_widht/2 * sign * rotate
        
        pygame.draw.polygon(self.surface, color, [tip, left_corner, right_corner])


        # Display measurement text if needed
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont('Arial', 18)
            
            # Create the label text
            if label is None:
                angle_degrees = np.degrees(abs(angle))
                label = f"{angle_degrees:.1f}°"
            
            text_surface = font.render(label, True, color)
            
            # Position text in the middle of the arc
            mid_angle = start_angle + sign * (angle/2)
            if sign == 1:
                mid_vector = np.array([np.cos(mid_angle), np.sin(mid_angle)])
            else:
                mid_vector = np.array([-np.cos(mid_angle), np.sin(mid_angle)])
            text_pos = center + radius * 1.2 * mid_vector  # Position slightly outside the arc

            # Offset to center the text
            text_rect = text_surface.get_rect()
            text_rect.center = text_pos
            
            self.surface.blit(text_surface, text_rect)
        
    def _draw_distance(self, surface, start_pos, length, angle_rad, color=(255, 255, 255), 
                      line_width=2, perpendicular_length=10, font=None, show_measurement=True, 
                      units="px", label=None):
        """
        Draw a distance measurement line with perpendicular markers at each end.
        
        Args:
            surface: Pygame surface to draw on
            start_pos: (x, y) tuple of the starting position
            length: Length of the measurement line
            angle_degrees: Angle of the line in degrees (0 = right, 90 = down)
            color: RGB tuple for line color
            line_width: Width of the line in pixels
            perpendicular_length: Length of the perpendicular end markers
            font: Pygame font object for text (if None, default font will be used)
            show_measurement: Whether to display the measurement text
            units: String for measurement units
        """
        # Convert angle to radians
        #angle_rad = math.radians(angle_degrees)
        
        # Calculate end position
        end_x = start_pos[0] + length * math.cos(angle_rad)
        end_y = start_pos[1] + length * math.sin(angle_rad)
        end_pos = (end_x, end_y)
        
        # Draw the main line
        pygame.draw.line(surface, color, start_pos, end_pos, line_width)
        
        # Calculate perpendicular angle
        perp_angle = angle_rad + math.pi/2  # 90 degrees in radians
        
        # Draw perpendicular line at start position
        perp_start_x1 = start_pos[0] + perpendicular_length/2 * math.cos(perp_angle)
        perp_start_y1 = start_pos[1] + perpendicular_length/2 * math.sin(perp_angle)
        perp_start_x2 = start_pos[0] - perpendicular_length/2 * math.cos(perp_angle)
        perp_start_y2 = start_pos[1] - perpendicular_length/2 * math.sin(perp_angle)
        pygame.draw.line(surface, color, (perp_start_x1, perp_start_y1), 
                        (perp_start_x2, perp_start_y2), line_width)
        
        # Draw perpendicular line at end position
        perp_end_x1 = end_pos[0] + perpendicular_length/2 * math.cos(perp_angle)
        perp_end_y1 = end_pos[1] + perpendicular_length/2 * math.sin(perp_angle)
        perp_end_x2 = end_pos[0] - perpendicular_length/2 * math.cos(perp_angle)
        perp_end_y2 = end_pos[1] - perpendicular_length/2 * math.sin(perp_angle)
        pygame.draw.line(surface, color, (perp_end_x1, perp_end_y1), 
                        (perp_end_x2, perp_end_y2), line_width)
        
        # Display measurement text
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont('Arial', 18)
            
            text = label
            text_surface = font.render(text, True, color)
            
            # Position text above the middle of the line
            text_pos_x = (start_pos[0] + end_pos[0]) / 2 - text_surface.get_width() / 2
            text_pos_y = (start_pos[1] + end_pos[1]) / 2 + 5#- text_surface.get_height() - 5
            
            # Adjust text position if line is vertical
            if np.pi/2 - np.pi/16 < abs(angle_rad) < np.pi/2 + np.pi/16:
                text_pos_x = (start_pos[0] + end_pos[0]) / 2 + 10
                text_pos_y = (start_pos[1] + end_pos[1]) / 2 - text_surface.get_height() / 2
            
            surface.blit(text_surface, (text_pos_x, text_pos_y))
        

    def _draw_angle(self, surface, center, angle, color=(255, 255, 255), 
              radius=30, line_width=2, perpendicular_length=10, font=None, show_measurement=True, 
              label=None, num_points=30, start_angle=3*np.pi/2):
        """
        Draw an angle measurement with an arc and perpendicular markers at endpoints.
        Args:
            surface: Pygame surface to draw on
            center: (x, y) numpy array of the center point
            angle: Angle in radians to draw (positive = counterclockwise, negative = clockwise)
            color: RGB tuple for line color
            radius: Radius of the arc
            line_width: Width of the line in pixels
            perpendicular_length: Length of the perpendicular end markers
            font: Pygame font object for text
            show_measurement: Whether to display the measurement text
            label: Custom label text
            num_points: Number of points for drawing the arc
            start_angle: The starting angle in radians (default is 3*pi/2 = 270° = straight up)
        """
        # Handle zero angle case
        if angle == 0:
            return
        
        # Determine direction based on angle sign
        sign = 1 if angle > 0 else -1
        
        # Calculate the arc angle
        arc_angle = abs(angle)
        
        # Calculate end angle
        end_angle = start_angle + sign * arc_angle
        
        # Draw the arc
        arc_points = [None] * num_points
        for i in range(num_points):
            t = i / (num_points - 1)
            theta = start_angle + sign * (arc_angle * t)
            pos = center + radius * np.array([np.cos(theta), np.sin(theta)])
            arc_points[i] = pos
        
        pygame.draw.lines(surface, color, False, arc_points, line_width)
        
        # Draw perpendicular markers at the endpoints
        # Start point - use radial direction (from center to point)
        start_vector = np.array([np.cos(start_angle), np.sin(start_angle)])
        start_point = center + radius * start_vector
        
        # For perpendicular to the arc, use the radial vector itself
        start_inner = start_point - (perpendicular_length/2) * start_vector
        start_outer = start_point + (perpendicular_length/2) * start_vector
        
        pygame.draw.line(surface, color, start_inner, start_outer, line_width)
        
        # End point - use radial direction
        end_vector = np.array([np.cos(end_angle), np.sin(end_angle)])
        end_point = center + radius * end_vector
        
        # For perpendicular to the arc, use the radial vector itself
        end_inner = end_point - (perpendicular_length/2) * end_vector
        end_outer = end_point + (perpendicular_length/2) * end_vector
        
        pygame.draw.line(surface, color, end_inner, end_outer, line_width)
        
        # Display measurement text if needed
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont('Arial', 18)
            
            # Create the label text
            if label is None:
                angle_degrees = np.degrees(abs(angle))
                label = f"{angle_degrees:.1f}°"
            
            text_surface = font.render(label, True, color)
            
            # Position text in the middle of the arc
            mid_angle = start_angle + sign * (arc_angle/2)
            mid_vector = np.array([np.cos(mid_angle), np.sin(mid_angle)])
            text_pos = center + radius * 1.2 * mid_vector  # Position slightly outside the arc
            
            # Offset to center the text
            text_rect = text_surface.get_rect()
            text_rect.center = text_pos
            
            surface.blit(text_surface, text_rect)


###########################################################################################################


class VesselRender(Window):

    VESSEL_LENGTH = 5.06 # meters
    VESSEL_WIDTH = 2.86 # meters
    VESSEL_CORNER = 0.5 # meters
    VESSEL_TRIANGLE = 0.4 # meters
    VESSEL_MOMENT_MARKER = 0.6 # meters

    def __init__(self, screen, window_pos, scale, title, window_width, window_height):
        super().__init__(screen, window_pos, title, window_width, window_height)

        self.scale = scale
        self.R = np.array([[0, self.scale],
                           [-self.scale, 0]])
        self.center = (window_width/2, window_height/2)
        self.T = np.array([[self.center[1]],[self.center[0]]])

        self.vessel_surface = pygame.Surface((window_width,window_height), pygame.SRCALPHA)
        self.shape = np.array([[self.VESSEL_LENGTH/2, -(self.VESSEL_WIDTH/2-self.VESSEL_CORNER)],
                               [self.VESSEL_LENGTH/2, self.VESSEL_WIDTH/2-self.VESSEL_CORNER],
                               [self.VESSEL_LENGTH/2-self.VESSEL_CORNER, self.VESSEL_WIDTH/2],
                               [-(self.VESSEL_LENGTH/2-self.VESSEL_CORNER), self.VESSEL_WIDTH/2],
                               [-self.VESSEL_LENGTH/2, self.VESSEL_WIDTH/2-self.VESSEL_CORNER],
                               [-self.VESSEL_LENGTH/2, -(self.VESSEL_WIDTH/2-self.VESSEL_CORNER)],
                               [-(self.VESSEL_LENGTH/2-self.VESSEL_CORNER), -self.VESSEL_WIDTH/2],
                               [self.VESSEL_LENGTH/2-self.VESSEL_CORNER, -self.VESSEL_WIDTH/2]])
        self.circle_point = np.zeros((1,2))  
        self.triangle_shape = np.array([[self.VESSEL_LENGTH/2, 0],
                                        [self.VESSEL_LENGTH/2-self.VESSEL_TRIANGLE, self.VESSEL_TRIANGLE/2],
                                        [self.VESSEL_LENGTH/2-self.VESSEL_TRIANGLE, -self.VESSEL_TRIANGLE/2]])
        self.moment_marker_line = np.array([[0, 0], [self.VESSEL_MOMENT_MARKER, 0]])

    def draw_vessel(self, x_err=None, y_err=None, psi_err=None, ned=False):
        if ned:
            shape = self._transform_body2ned(x_err, y_err, psi_err, self.shape)
            circle = self._transform_body2ned(x_err, y_err, psi_err, self.circle_point)
            triangle = self._transform_body2ned(x_err, y_err, psi_err, self.triangle_shape)
        else:
            shape = self.shape
            circle = self.circle_point
            triangle = self.triangle_shape

        self.vessel_surface.fill(Color.AGENT_BLACK.value)
        pygame.draw.polygon(self.vessel_surface, Color.AGENT_BLUE.value, self._world_2_pixels(shape))
        pygame.draw.polygon(self.vessel_surface, Color.BLACK.value, self._world_2_pixels(triangle))
        pygame.draw.circle(self.vessel_surface, Color.BLACK.value, self._world_2_pixels(circle).ravel(), radius=3)
        if not ned:
            line = self._world_2_pixels(self.moment_marker_line)
            pygame.draw.line(self.vessel_surface, Color.BLACK.value, line[0], line[1])
        self.surface.blit(self.vessel_surface, (0,0))

    def draw_target(self, x_err, y_err, psi_err, ned=False, color=Color.DESIRED_YELLOW.value):
        if ned:
            shape = self._transform_vessel(0, 0, psi_err, self.shape)
            #shape = self.shape
            circle = self.circle_point
            #triangle = self.triangle_shape
            triangle = self._transform_vessel(0, 0, psi_err, self.triangle_shape)

        else:
            shape = self._transform_vessel(x_err, y_err, psi_err, self.shape)
            circle = self._transform_vessel(x_err, y_err, psi_err, self.circle_point)
            triangle = self._transform_vessel(x_err, y_err, psi_err, self.triangle_shape)

        points = self._world_2_pixels(shape)
        triangle_points = self._world_2_pixels(triangle)

        for i in range(len(shape)):
            self._draw_dashed_line(points[i], points[(i+1) % len(points)], color)
            if i < len(triangle):
                self._draw_dashed_line(triangle_points[i], triangle_points[(i+1) % len(triangle_points)], color, dash_lenght=5)
            
        
        #pygame.draw.polygon(self.surface, Color.DESIRED_YELLOW.value, self._world_2_pixels(triangle))
        pygame.draw.circle(self.surface, color, self._world_2_pixels(circle).ravel(), radius=3)
        

    def _draw_dashed_line(self, start_pos, end_pos, color, dash_lenght=10, gap_length=5):
        px1, py1 = start_pos
        px2, py2 = end_pos
        length = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
        dashes = int(length / (dash_lenght + gap_length))
        for i in range(dashes):
            t1 = i / dashes
            t2 = (i + 0.5) / dashes
            segment_start = (px1 + (px2 - px1) * t1, py1 + (py2 - py1) * t1)
            segment_end = (px1 + (px2 - px1) * t2, py1 + (py2 - py1) * t2)
            pygame.draw.line(self.surface, color, segment_start, segment_end, 2)
        
    # Transformations #############

    def _transform_vessel(self,x,y,psi, shape):
        angle = np.deg2rad(psi)
        return (self._R2(angle).T @ shape.T + self._T2(x,y)).T

    def _transform_body2ned(self,x,y,psi,shape):
        angle = np.deg2rad(psi)
        return (self._R2(angle) @ (shape.T + self._T2(-x,-y))).T

    def _R2(self,psi):
        return np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])

    def _T2(self,x,y):
        return np.array([[x,y]]).T    

    def _world_2_pixels(self, cord):
        return (self.R @ cord.T + self.T).T

    def _degrees2pygame(self, angle):
        return np.pi * angle / 180 - np.pi/2

    def _scalar2pygame(self, scalar):
        return self.scale * scalar 


###########################################################################################################


class BodyRender(VesselRender):
    SCALE = 50
    ACTUATOR_X = 1.8
    ACTUATOR_Y = 0.8

    def __init__(self, screen, window_pos, title="BODY-frame", window_width=450, window_height=450):
        super().__init__(screen, window_pos, self.SCALE, title, window_width, window_height)
        
        self.label_font =  pygame.font.SysFont('DejaVu Sans', 12)
        self.body_legend_items = (
            (Color.DESIRED_YELLOW.value, "Desired {'total thrust force', 'total trust moment', 'pose'}"),
            (Color.DESIRED_LIGHT_YELLOW.value, "Desired thrust force"),
            (Color.VELOCITY_GREEN.value, "{'surge', 'sway', 'angular'} velocity")
        )

        self.legend_surface, self.box_pos = self.create_legend(self.body_legend_items, self.label_font, padding_bottom=0)

        self.actuator_pos = np.array([[self.ACTUATOR_X, -self.ACTUATOR_Y],
                                      [self.ACTUATOR_X, self.ACTUATOR_Y],
                                      [-self.ACTUATOR_X, self.ACTUATOR_Y],
                                      [-self.ACTUATOR_X, -self.ACTUATOR_Y]])
    
    def _draw_actuator_ref(self, actuator_ref):
        for i, (x,y) in enumerate(self.actuator_pos):
            px,py = self._world_2_pixels(np.array([[x,y]])).ravel()
            angle = self._degrees2pygame(actuator_ref[i][1])
            perp_angle = angle + np.pi/2
            self.draw_arrow(self._degrees2pygame(actuator_ref[i][1]), self._scalar2pygame(actuator_ref[i][0]),
                            Color.DESIRED_LIGHT_YELLOW.value, np.array([px,py])) 
            pygame.draw.line(self.surface, Color.DESIRED_LIGHT_YELLOW.value,
                             (px - 10 * np.cos(perp_angle), py - 10 * np.sin(perp_angle)),
                             (px + 10 * np.cos(perp_angle), py + 10 * np.sin(perp_angle)), width=2)
            pygame.draw.circle(self.surface, Color.BLACK.value, (px,py), radius=3)
            pygame.draw.circle(self.surface, Color.BLACK.value, (px-7*np.sin(perp_angle),py+7*np.cos(perp_angle)), radius=1)

    def _draw_force_and_moment(self, n_d, alpha_d, angular_thrust_ref):
        
        center = self._world_2_pixels(np.zeros((1,2))).ravel()
        self.draw_arrow(self._degrees2pygame(alpha_d), self._scalar2pygame(n_d), Color.DESIRED_YELLOW.value, center)
        #moment_norm = self.calulate_total_moment(thrusters)
        radius = self._scalar2pygame(self.VESSEL_MOMENT_MARKER)
        angular_thrust_ref1 = angular_thrust_ref * self.SCALE/radius
        self._draw_curved_arrow(angular_thrust_ref1, center, Color.DESIRED_YELLOW.value, radius)


    def _draw_velocities(self, u_hat, v_hat, r_hat):
        
        center = self._world_2_pixels(np.zeros((1,2))).ravel()
        self.draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.VELOCITY_GREEN.value, center)
        self.draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.VELOCITY_GREEN.value, center)
        self._draw_curved_arrow(np.deg2rad(r_hat), center, Color.VELOCITY_GREEN.value, self._scalar2pygame(self.VESSEL_MOMENT_MARKER))
        

    def _draw_curved_arrow(self, angle, center, color, radius=10,
                           line_width=2, arrowhead_length=10, arrowhead_widht=10, num_points=30):
        
        if angle == 0:
            return
        
        sign = 1 if angle > 0 else -1

        total_arc_length = radius * abs(angle)

        if total_arc_length > arrowhead_length:
            line_arc_length = total_arc_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_arc_length = 0
            current_arrowhead_length = total_arc_length
            arrowhead_widht *= (total_arc_length / arrowhead_length)

        line_arc_angle = line_arc_length / radius
        arrowhead_arc_angle = current_arrowhead_length / radius

        start_angle = 3*np.pi/2
        tip_angle = start_angle + sign * (line_arc_angle + arrowhead_arc_angle)

        if line_arc_length > 0:
            arc_points = [None] * num_points
            for i in range(num_points):
                t = i / (num_points -1)
                theta = start_angle + sign * (line_arc_angle * t)
                pos = center + radius * np.array([np.cos(theta), np.sin(theta)])
                arc_points[i] = pos
            pygame.draw.lines(self.surface, color, False, arc_points, line_width)
        
        rotate = np.array([np.cos(tip_angle), np.sin(tip_angle)])
        tip = center + radius * rotate
        base_center = tip - current_arrowhead_length * sign * np.array([-rotate[1], rotate[0]])
        left_corner = base_center + arrowhead_widht/2 * sign * rotate
        right_corner = base_center - arrowhead_widht/2 * sign * rotate
        
        pygame.draw.polygon(self.surface, color, [tip, left_corner, right_corner])

    def calulate_total_moment(self, thrusters):
        total_moment_prime = 0.0
        for i, ((x,y), (n, alpha)) in enumerate(zip(self.actuator_pos, thrusters)):
            rad = np.deg2rad(90*(i+1)-alpha)
            if x*y < 0:
                M_prime = abs(x) * n * np.cos(rad) + abs(y) * n * np.sin(rad)
            else:
                M_prime = abs(x) * n * np.sin(rad) + abs(y) * n * np.cos(rad)
            total_moment_prime += M_prime
        return total_moment_prime

    def render(self, actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat):
        # init
        self.render_surface()
        
        # BODY render
        self.surface.fill(Color.OCEAN_BLUE.value)
        self.draw_target(x_tilde, y_tilde, psi_tilde)
        self.draw_vessel(x_tilde, y_tilde, psi_tilde)
        self._draw_actuator_ref(actuator_ref)
        self._draw_force_and_moment(tot_thrust, tot_angle, tot_angular_thrust)
        self._draw_velocities(u_hat, v_hat, r_hat)
        self.draw_legend(self.legend_surface, self.box_pos)
        
        # final render
        self.render_surface_title()
        self.render_window()


###########################################################################################################


class NedRender(VesselRender):
    SCALE = 25
    
    def __init__(self, screen, window_pos, title="NED-frame", window_width=450, window_height=450):
        super().__init__(screen, window_pos, self.SCALE, title, window_width, window_height)
    
    def render(self, x_err, y_err, psi_err, target_heading):
        # init
        self.render_surface()
        
        # NED render
        self.surface.fill(Color.OCEAN_BLUE.value)
        self.draw_target(x_err, y_err, target_heading, ned=True)
        self.draw_vessel(x_err, y_err, psi_err-target_heading, ned=True)

        # final render
        self.render_surface_title()
        self.render_window()


###########################################################################################################


class ShapExplainRender(VesselRender, Utilities):
    
    SCALE = 25
    ACTUATOR_X = 1.8
    ACTUATOR_Y = 0.8
    VESSEL_MOMENT_MARKER = 50/SCALE

    def __init__(self, screen, window_pos, title="Action explained BODY-frame", window_width=450, window_height=450):
        VesselRender.__init__(self, screen, window_pos, self.SCALE, title, window_width, window_height)
        Utilities.__init__(self)

        self.label_font =  pygame.font.SysFont('DejaVu Sans', 12)
        
        # self.body_legend_items = (
        #     (Color.RED.value, "Main feature causing the action"),
        #     (Color.DESIRED_LIGHT_YELLOW.value, "Estimated action output based on main feature"),
        #     (Color.PURPLE.value, "Estimated action output based on all features"),
        #     (Color.VELOCITY_GREEN.value, "Actual action output")
        # )

        self.body_legend_items = (
            (Color.RED.value, "Main feature causing the action"),
            (Color.DESIRED_LIGHT_YELLOW.value, "Estimated action output based on main feature")
        )

        self.legend_surface, self.box_pos = self.create_legend(self.body_legend_items, self.label_font, padding_bottom=0)

        self.idx = 0
        self.prev_idx = -1
        self.explain_offset = 20
        self.explain_mode = -1
        self.RPM_true = True
        self.explain_vector = False
        self.prev_angle_1 = np.array([135, -135, -45, 45])
        self.prev_angle_2 = np.array([135, -135, -45, 45])

        self.actuator_pos = np.array([[self.ACTUATOR_X, -self.ACTUATOR_Y],
                                            [self.ACTUATOR_X, self.ACTUATOR_Y],
                                            [-self.ACTUATOR_X, self.ACTUATOR_Y],
                                            [-self.ACTUATOR_X, -self.ACTUATOR_Y]])
        
    def _find_explaination(self, lists, n=3, list_in_list=True):
        combined_values = []
        if list_in_list:
            # Calculate the combined absolute values for each index
            for i in range(len(lists[0])):
                #total = sum(abs(lst[i]) for lst in lists)
                total = sum(np.sqrt(lists[2*j][i]**2 + lists[2*j+1][i]**2) for j in range(len(lists)//2))
                combined_values.append((i, total))  # Store (index, value) pairs
        else:
            for i in range(len(lists)):
                combined_values.append((i,-lists[i]))

        # Sort by combined value in descending order
        combined_values.sort(key=lambda x: x[1], reverse=True)
        
        # Get only the top n indices
        top_indices = [item[0] for item in combined_values[:n]]
        self.idx = top_indices[0]
    
    def _calculate_action_direction(self, shap_values, thrusters, overide=None):

        x_or_y = [None]*2
        value = [None]*2
        arr = np.array(shap_values)

        for i in range(2):
            s=0
            if i == 0:
                s_prime = arr[:, self.idx]
            else:
                s_prime = np.sum(arr, axis=1)
            for j, thruster in enumerate(thrusters):
                ang = np.deg2rad(thruster[1])
                s += s_prime[j] * np.array([np.cos(ang),np.sin(ang)])
            
            vectors = [
                (s_prime[0], thrusters[0][1]),
                (s_prime[1], thrusters[1][1]),
                (s_prime[2], thrusters[2][1]),
                (s_prime[3], thrusters[3][1]),
            ]
            moment_norm = self._calulate_total_moment(vectors)
            s = np.append(s,moment_norm)
            x_or_y[i] = np.argmax(np.abs(s))
 
            value[i] = s[x_or_y[i]]
            if i ==1 and overide != None:
                value[i] = s[overide]
                a = s

        return x_or_y[0],value[0],x_or_y[1],value[1],a[0],a[1],a[2]
    
    def _calculate_ad(self,thrusters):
        s=0
        for thruster in thrusters:
            ang = np.deg2rad(thruster[1])
            s += thruster[0] * np.array([np.cos(ang),np.sin(ang)])

        moment_norm = self._calulate_total_moment(thrusters)
        s = np.append(s,moment_norm)
        return s
    
    def _calulate_total_moment(self, thrusters):
        total_moment_prime = 0.0
        for i, ((x,y), (n, alpha)) in enumerate(zip(self.actuator_pos, thrusters)):
            rad = np.deg2rad(90*(i+1)-alpha)
            if x*y < 0:
                M_prime = abs(x) * n * np.cos(rad) + abs(y) * n * np.sin(rad)
            else:
                M_prime = abs(x) * n * np.sin(rad) + abs(y) * n * np.cos(rad)
            total_moment_prime += M_prime
        return total_moment_prime
        
    def _draw_explaination(self, shap_values_EV, shap_values_RPM, thrusters, error_x, error_y, error_psi, u_hat, v_hat, r_hat):
        
           
        if self.idx == 0:
            self._draw_distance(self.surface, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), self._scalar2pygame(error_x), self._degrees2pygame(0), Color.RED.value, label=f"{error_x:.2f} m") # Explaination
            total_rpm_x,total_rpm_y,_ = self._calculate_ad(thrusters)
            total_rpm_x *= self.SCALE
            total_rpm_y *= self.SCALE
            magnitude = np.sqrt(total_rpm_x**2 + total_rpm_y**2)
            angle_radians = np.arctan2(total_rpm_y,total_rpm_x)
            angle_degrees = np.degrees(angle_radians)

            #self._draw_arrow(self._degrees2pygame(angle_degrees),magnitude, Color.VELOCITY_GREEN.value, self.center) # Total RPM in x direction

            i, idx_value, j, total_value,x,y,_ = self._calculate_action_direction(shap_values_RPM, thrusters, overide=0)
            #print(f"i:{i}, idx_value{idx_value}\nj:{j}, total_value:{total_value}")
            
            if i != 2:
                idx_value *= self.SCALE
            else: idx_value *= self.SCALE
            total_value *= self.SCALE

            x *= self.SCALE
            y *= self.SCALE
            
            magnitude = np.sqrt(x**2 + y**2)
            angle_radians = np.arctan2(y,x)
            angle_degrees = np.degrees(angle_radians)

            self._draw_arrow(self._degrees2pygame(0), total_value, Color.PURPLE.value, (self.center[0]-2, self.center[1])) # Total SHAP RPM in x direction
            #self._draw_arrow(self._degrees2pygame(angle_degrees),magnitude, Color.PURPLE.value, self.center) # Total RPM in x direction
            self._draw_arrow(self._degrees2pygame(0),total_rpm_x, Color.VELOCITY_GREEN.value, self.center) # Total RPM in x direction


            if i == 0:
                self._draw_arrow(self._degrees2pygame(0), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+10, self.center[1])) # SHAP RPM in x direction
            elif i == 1:
                self._draw_arrow(self._degrees2pygame(90), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+10, self.center[1])) # SHAP RPM in y direction
            elif i == 2:
                self._draw_curved_arrow(np.deg2rad(idx_value), self.center, Color.DESIRED_LIGHT_YELLOW.value,radius=30) # SHAP RPM in moment direction
            else:
                ...

        elif self.idx == 1:
            self._draw_distance(self.surface, (self.center[0], self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), self._scalar2pygame(error_y), self._degrees2pygame(90), Color.RED.value, label=f"{error_y:.2f} m") # Explaination
            _,total_rpm,_ = self._calculate_ad(thrusters)
            total_rpm *= self.SCALE

            i, idx_value, j, total_value ,_,_,_ = self._calculate_action_direction(shap_values_RPM, thrusters, overide=1)
            #print(f"i:{i}, idx_value{idx_value}\nj:{j}, total_value:{total_value}")
            
            if i != 2:
                idx_value *= self.SCALE
            else: idx_value *= self.SCALE
            total_value *= self.SCALE

            self._draw_arrow(self._degrees2pygame(90), total_value, Color.PURPLE.value, (self.center[0], self.center[1]-2)) # Total SHAP RPM in x direction
            self._draw_arrow(self._degrees2pygame(90),total_rpm, Color.VELOCITY_GREEN.value, self.center) # Total RPM in x direction

            if i == 0:
                self._draw_arrow(self._degrees2pygame(0), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1]+10)) # SHAP RPM in x direction
            elif i == 1:
                self._draw_arrow(self._degrees2pygame(90), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1]+10)) # SHAP RPM in y direction
            elif i == 2:
                self._draw_curved_arrow(np.deg2rad(idx_value), self.center, Color.DESIRED_LIGHT_YELLOW.value,radius=30) # SHAP RPM in moment direction
            else:
                ...
        elif self.idx == 2:
            self._draw_angle(self.surface, self.center, np.deg2rad(error_psi), Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, label=f"{error_psi:.0f} °")
            
            _,_,moment = self._calculate_ad(thrusters)
            moment *= self.SCALE

            i, idx_value, j, total_value ,_,_,_ = self._calculate_action_direction(shap_values_RPM, thrusters, overide=2)
            #print(f"i:{i}, idx_value{idx_value}\nj:{j}, total_value:{total_value}")
            
            if i != 2:
                idx_value *= self.SCALE
            else: idx_value *= self.SCALE
            total_value *= self.SCALE

            self._draw_curved_arrow(np.deg2rad(total_value), self.center, Color.PURPLE.value, radius=42)
            self._draw_curved_arrow(np.deg2rad(moment),self.center, Color.VELOCITY_GREEN.value, radius=40)

            if i == 0:
                self._draw_arrow(self._degrees2pygame(0), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1])) # SHAP RPM in x direction
            elif i == 1:
                self._draw_arrow(self._degrees2pygame(90), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1])) # SHAP RPM in y direction
            elif i == 2:
                self._draw_curved_arrow(np.deg2rad(idx_value), self.center, Color.DESIRED_LIGHT_YELLOW.value,radius=30) # SHAP RPM in moment direction
            else:
                ...
        
        elif self.idx == 3:
            if u_hat > 0:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]-self._scalar2pygame(self.VESSEL_LENGTH/2)-self.explain_offset), show_measurement=True, label=f"{u_hat:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), show_measurement=True, label=f"{u_hat:.2f} m/s")
            
            total_rpm_x,total_rpm_y,_ = self._calculate_ad(thrusters)
            total_rpm_x *= self.SCALE
            total_rpm_y *= self.SCALE
            magnitude = np.sqrt(total_rpm_x**2 + total_rpm_y**2)
            angle_radians = np.arctan2(total_rpm_y,total_rpm_x)
            angle_degrees = np.degrees(angle_radians)

            #self._draw_arrow(self._degrees2pygame(angle_degrees),magnitude, Color.VELOCITY_GREEN.value, self.center) # Total RPM in x direction

            i, idx_value, j, total_value,x,y,_ = self._calculate_action_direction(shap_values_RPM, thrusters, overide=0)
            #print(f"i:{i}, idx_value{idx_value}\nj:{j}, total_value:{total_value}")
            
            if i != 2:
                idx_value *= self.SCALE
            else: idx_value *= self.SCALE
            total_value *= self.SCALE

            x *= self.SCALE
            y *= self.SCALE
            
            magnitude = np.sqrt(x**2 + y**2)
            angle_radians = np.arctan2(y,x)
            angle_degrees = np.degrees(angle_radians)

            self._draw_arrow(self._degrees2pygame(0), total_value, Color.PURPLE.value, (self.center[0]-2, self.center[1])) # Total SHAP RPM in x direction
            #self._draw_arrow(self._degrees2pygame(angle_degrees),magnitude, Color.PURPLE.value, self.center) # Total RPM in x direction
            self._draw_arrow(self._degrees2pygame(0),total_rpm_x, Color.VELOCITY_GREEN.value, self.center) # Total RPM in x direction


            if i == 0:
                self._draw_arrow(self._degrees2pygame(0), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+10, self.center[1])) # SHAP RPM in x direction
            elif i == 1:
                self._draw_arrow(self._degrees2pygame(90), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+10, self.center[1])) # SHAP RPM in y direction
            elif i == 2:
                self._draw_curved_arrow(np.deg2rad(idx_value), self.center, Color.DESIRED_LIGHT_YELLOW.value,radius=30) # SHAP RPM in moment direction
            else:
                ...

        elif self.idx == 4:
            if v_hat > 0:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), show_measurement=True, label=f"{v_hat:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]-self._scalar2pygame(self.VESSEL_WIDTH/2)-self.explain_offset,self.center[1]), show_measurement=True, label=f"{v_hat:.2f} m/s")
            _,total_rpm,_ = self._calculate_ad(thrusters)
            total_rpm *= self.SCALE

            i, idx_value, j, total_value ,_,_,_ = self._calculate_action_direction(shap_values_RPM, thrusters, overide=1)
            #print(f"i:{i}, idx_value{idx_value}\nj:{j}, total_value:{total_value}")
            
            if i != 2:
                idx_value *= self.SCALE
            else: idx_value *= self.SCALE
            total_value *= self.SCALE

            self._draw_arrow(self._degrees2pygame(90), total_value, Color.PURPLE.value, (self.center[0], self.center[1]-2)) # Total SHAP RPM in x direction
            self._draw_arrow(self._degrees2pygame(90),total_rpm, Color.VELOCITY_GREEN.value, self.center) # Total RPM in x direction

            if i == 0:
                self._draw_arrow(self._degrees2pygame(0), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1]+10)) # SHAP RPM in x direction
            elif i == 1:
                self._draw_arrow(self._degrees2pygame(90), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1]+10)) # SHAP RPM in y direction
            elif i == 2:
                self._draw_curved_arrow(np.deg2rad(idx_value), self.center, Color.DESIRED_LIGHT_YELLOW.value,radius=30) # SHAP RPM in moment direction
            else:
                ...
            
        elif self.idx == 5:
            self._draw_curved_arrow(np.deg2rad(r_hat), self.center, Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, show_measurement=True, label=f"{r_hat:.0f} °/s")
            _,_,moment = self._calculate_ad(thrusters)

            radius = 40
            moment *= self.SCALE/radius

            i, idx_value, j, total_value ,_,_,_ = self._calculate_action_direction(shap_values_RPM, thrusters, overide=2)
            #print(f"i:{i}, idx_value{idx_value}\nj:{j}, total_value:{total_value}")
            
            if i != 2:
                idx_value *= self.SCALE/radius
            else: idx_value *= self.SCALE/radius
            total_value *= self.SCALE/radius

            self._draw_curved_arrow(total_value, self.center, Color.PURPLE.value, radius=radius+2)
            self._draw_curved_arrow(moment,self.center, Color.VELOCITY_GREEN.value, radius=radius)

            if i == 0:
                self._draw_arrow(self._degrees2pygame(0), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1])) # SHAP RPM in x direction
            elif i == 1:
                self._draw_arrow(self._degrees2pygame(90), idx_value, Color.DESIRED_LIGHT_YELLOW.value, (self.center[0], self.center[1])) # SHAP RPM in y direction
            elif i == 2:
                self._draw_curved_arrow(idx_value, self.center, Color.DESIRED_LIGHT_YELLOW.value,radius=30) # SHAP RPM in moment direction
            else:
                ...

        elif self.idx == 6:
            ...
        elif self.idx == 7:
            ...
        elif self.idx == 8:
            ...
        elif self.idx == 9:
            ...
        elif self.idx == 10:
            ...
        elif self.idx == 11:
            ...
        elif self.idx == 12:
            ...
        elif self.idx == 13:
            ...
        else:
            ...

    def _add_vectors_polar(self, list1, list2=None):
        """
        Add 2D vectors in polar coordinates, with vectors in list2 expressed
        relative to the corresponding vectors in list1.
        
        Parameters:
        list1 (list): List of vectors in polar coordinates [(r1, θ1), (r2, θ2), ...].
        list2 (list, optional): Second list of vectors of the same length as list1,
                            expressed relative to the corresponding vectors in list1.
        
        Returns:
        tuple: Resultant vector in polar coordinates (r, θ).
        """
        
        # If only one list is provided
        if list2 is None:
            vectors = list1
        else:
            # If two lists are provided, add corresponding vectors with list2 in relative frame
            if len(list1) != len(list2):
                raise ValueError("Both lists must have the same length.")
            
            vectors = []
            for vec1, vec2 in zip(list1, list2):
                # Extract polar coordinates
                r1, theta1 = vec1
                r2, theta2 = vec2
                                
                theta_sum = theta1+theta2
                r_sum = r1+r2
                
                vectors.append((r_sum, theta_sum))
        
        # Add all vectors in the list
        x_total = 0
        y_total = 0
        
        for vec in vectors:
            r, theta = vec
            theta_rad = np.radians(theta)
            
            x_total += r * np.cos(theta_rad)
            y_total += r * np.sin(theta_rad)

        if len(list(vectors)) == 4:
            moment_norm = self._calulate_total_moment(vectors)
        else:
            moment_norm = None

        # Convert back to polar
        r_result = np.sqrt(x_total**2 + y_total**2)
        theta_result = np.degrees(np.arctan2(y_total, x_total))

        return (r_result, theta_result, moment_norm)
    

    def _add_vectors_polar_old(self, list1, list2=None):
        """
        Add 2D vectors in polar coordinates, with vectors in list2 expressed
        relative to the corresponding vectors in list1.
        
        Parameters:
        list1 (list): List of vectors in polar coordinates [(r1, θ1), (r2, θ2), ...].
        list2 (list, optional): Second list of vectors of the same length as list1,
                            expressed relative to the corresponding vectors in list1.
        
        Returns:
        tuple: Resultant vector in polar coordinates (r, θ).
        """
        
        # If only one list is provided
        if list2 is None:
            vectors = list1
        else:
            # If two lists are provided, add corresponding vectors with list2 in relative frame
            if len(list1) != len(list2):
                raise ValueError("Both lists must have the same length.")
            
            vectors = []
            for vec1, vec2 in zip(list1, list2):
                # Extract polar coordinates
                r1, theta1 = vec1
                r2, theta2 = vec2
                
                # Convert to radians
                theta1_rad = np.radians(theta1)
                
                # For vec2, the angle is relative to vec1, so we add the angles
                # This transforms vec2 from vec1's reference frame to the global frame
                theta2_global = theta1 + theta2
                theta2_global_rad = np.radians(theta2_global)
                
                # Convert both to Cartesian
                x1 = r1 * np.cos(theta1_rad)
                y1 = r1 * np.sin(theta1_rad)
                
                x2 = r2 * np.cos(theta2_global_rad)
                y2 = r2 * np.sin(theta2_global_rad)
                
                # Add in Cartesian
                x_sum = x1 + x2
                y_sum = y1 + y2
                
                # Convert back to polar
                r_sum = np.sqrt(x_sum**2 + y_sum**2)
                theta_sum = np.degrees(np.arctan2(y_sum, x_sum))
                
                vectors.append((r_sum, theta_sum))
        
        # Add all vectors in the list
        x_total = 0
        y_total = 0
        
        for vec in vectors:
            r, theta = vec
            theta_rad = np.radians(theta)
            
            x_total += r * np.cos(theta_rad)
            y_total += r * np.sin(theta_rad)

        if len(list(vectors)) == 4:
            moment_norm = self._calulate_total_moment(vectors)
        else:
            moment_norm = None

        # Convert back to polar
        r_result = np.sqrt(x_total**2 + y_total**2)
        theta_result = np.degrees(np.arctan2(y_total, x_total))

        return (r_result, theta_result, moment_norm)
    
    def _draw_explaination2(self, error_x, error_y, error_psi, u_hat, v_hat, r_hat, shap_values_RPM, shap_values_angles, base_vectors, n_d, alpha_d, thrusters):
        

        arr_RPM = np.array(shap_values_RPM)
        arr_angles = np.array(shap_values_angles)
        shap_feature_RPM = arr_RPM[:, self.idx] * 1200
        shap_feature_angles = arr_angles[:, self.idx] * 180
        shap_tot_RPM = np.sum(arr_RPM, axis=1) * 1200
        shap_tot_angles = np.sum(arr_angles, axis=1) * 180

        shap_vectors = [(shap_feature_RPM[i], shap_feature_angles[i]) for i in range(4)]
        shap_tot_vectors = [(shap_tot_RPM[i], shap_tot_angles[i]) for i in range(4)]

        base_vector = self._add_vectors_polar(base_vectors)
        main_feature_vector_base = self._add_vectors_polar(base_vectors, shap_vectors)
        main_feature_vector = self._add_vectors_polar([main_feature_vector_base[:2],(-base_vector[0],base_vector[1])])
        tot_vector_base = self._add_vectors_polar(base_vectors, shap_tot_vectors)
        tot_vector = self._add_vectors_polar([tot_vector_base[:2],(-base_vector[0],base_vector[1])])
        _,_,moment = self._calculate_ad(thrusters)


        radius = 30
        if self.explain_vector:
            # Angular RPMS for all SHAP features (estimate of total angular RPM)
            self._draw_curved_arrow(self._scalar2pygame(tot_vector_base[2])/(1200*radius), self.center, Color.PURPLE.value, radius=radius+10)
            # Total angular RPM
            self._draw_curved_arrow(self._scalar2pygame(moment)/radius, self.center, Color.VELOCITY_GREEN.value, radius=radius+10)
        self._draw_curved_arrow(self._scalar2pygame(main_feature_vector_base[2])/(1200*radius), self.center, Color.DESIRED_LIGHT_YELLOW.value, radius=radius, arrowhead_length=5, arrowhead_widht=5)
              
        # Base vector
        #self._draw_arrow(self._degrees2pygame(base_vector[1]), self._scalar2pygame(base_vector[0]/1200), Color.BLACK.value, self.center)
        
        # Convert both to Cartesian
        # x1 = self._scalar2pygame(base_vector[0]/1200) * np.cos(np.deg2rad(base_vector[1]))
        # y1 = self._scalar2pygame(base_vector[0]/1200) * np.sin(np.deg2rad(base_vector[1]))

        # Thruster vector from base for all SHAP features
        #self._draw_arrow(self._degrees2pygame(tot_vector[1]), self._scalar2pygame(tot_vector[0]/1200), Color.RED.value, (self.center[0]+y1,self.center[1]-x1))
        
        if self.explain_vector:
            # Thruster vector for all SHAP features (estimate of totalt thruster vector)
            self._draw_arrow(self._degrees2pygame(tot_vector_base[1]), self._scalar2pygame(tot_vector_base[0]/1200), Color.PURPLE.value, self.center)
            # Total thruster vector
            self._draw_arrow(self._degrees2pygame(alpha_d), self._scalar2pygame(n_d), Color.VELOCITY_GREEN.value, self.center)
        
        # Thruster vector from base for single SHAP feature
        #self._draw_arrow(self._degrees2pygame(main_feature_vector[1]), self._scalar2pygame(main_feature_vector[0]/1200), Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+y1,self.center[1]-x1))
        self._draw_arrow(self._degrees2pygame(main_feature_vector_base[1]), self._scalar2pygame(main_feature_vector_base[0]/1200), Color.DESIRED_LIGHT_YELLOW.value, self.center, arrowhead_width=5, arrowhead_length=5)

        if self.idx == 0:
            self._draw_distance(self.surface, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), self._scalar2pygame(error_x), self._degrees2pygame(0), Color.RED.value, label=f"{error_x:.2f} m") # Explaination
        elif self.idx == 1:
            self._draw_distance(self.surface, (self.center[0], self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), self._scalar2pygame(error_y), self._degrees2pygame(90), Color.RED.value, label=f"{error_y:.2f} m") # Explaination
        elif self.idx == 2:
            self._draw_angle(self.surface, self.center, np.deg2rad(error_psi), Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, label=f"{error_psi:.0f} °")
        elif self.idx == 3:
            if u_hat > 0:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]-self._scalar2pygame(self.VESSEL_LENGTH/2)-self.explain_offset), show_measurement=True, label=f"{u_hat:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), show_measurement=True, label=f"{u_hat:.2f} m/s")
        elif self.idx == 4:
            if v_hat > 0:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), show_measurement=True, label=f"{v_hat:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]-self._scalar2pygame(self.VESSEL_WIDTH/2)-self.explain_offset,self.center[1]), show_measurement=True, label=f"{v_hat:.2f} m/s")
        elif self.idx == 5:
            self._draw_curved_arrow(np.deg2rad(r_hat), self.center, Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, show_measurement=True, label=f"{r_hat:.0f} °/s")

        error = np.array([n_d*1200, alpha_d, moment*1200]) - np.array([tot_vector_base[0], tot_vector_base[1], tot_vector_base[2]])

        acc_RPM = (2400*4-abs(error[0]))/(2400*4)
        acc_angle = (360-abs(error[1]))/360
        acc_moment = (2400*4*1.8 - abs(error[2]))/(2400*4*1.8)

        if acc_RPM > 0.9:
            acc_RPM_color = Color.GREEN.value
        elif acc_RPM > 0.8:
            acc_RPM_color = Color.ORANGE.value
        else:
            acc_RPM_color = Color.RED.value
        
        if acc_angle > 0.9:
            acc_angle_color = Color.GREEN.value
        elif acc_angle > 0.8:
            acc_angle_color = Color.ORANGE.value
        else:
            acc_angle_color = Color.RED.value
        
        if acc_moment > 0.9:
            acc_moment_color = Color.GREEN.value
        elif acc_moment > 0.8:
            acc_moment_color = Color.ORANGE.value
        else:
            acc_moment_color = Color.RED.value

        text_surface1 = self.window_font.render(f"Accuracy RPM: {acc_RPM:.2f}", self.antialias, acc_RPM_color)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (self._title_offset, self._title_offset*10)
        
        text_surface2 = self.window_font.render(f"Accuracy angle: {acc_angle:.2f}", self.antialias, acc_angle_color)
        text_rect2 = text_surface2.get_rect()
        text_rect2.topleft = (self._title_offset, self._title_offset*11+text_rect1.height)

        text_surface3 = self.window_font.render(f"Accuracy moment: {acc_moment:.2f}", self.antialias, acc_moment_color)
        text_rect3 = text_surface3.get_rect()
        text_rect3.topleft = (self._title_offset, self._title_offset*12+text_rect1.height+text_rect2.height)
        
        self.surface.blit(text_surface1, text_rect1)
        self.surface.blit(text_surface2, text_rect2)
        self.surface.blit(text_surface3, text_rect3)

    def _calculate_angular_thrust(self, rpm):
        ang_thrust=0
        for i, (x,y) in enumerate(self.actuator_pos):
            ang_thrust += rpm[2*i] * y
            ang_thrust += rpm[2*i+1] * x
        return ang_thrust        
    
    def _calculate_actuator_inputs(self, action, prev_angle):
        thrusters = np.zeros(4)
        angles = np.zeros(4)
        vec = []
        # Process each thruster
        for i in range(4):
            x = action[i*2]
            y = action[i*2 + 1]

            if x == 0 and y == 0:
                # add 0 thrust and angle
                thrusters[i] = 0
                angles[i] = prev_angle[i]
                vec.append((0.0, prev_angle[i]))
                continue

            angle = np.arctan2(y, x)
            if angle == np.pi and i ==1:
                angle = -np.pi
            thrust = np.clip(np.sqrt(x**2 + y**2), 0, 1)
            
            angles[i] = angle / np.pi * 180
            thrusters[i] = thrust * 900
            vec.append((thrust, angles[i]))
        return thrusters, angles, angles, vec
        
    def _combine_actuator_ref(self, actuator_ref, actuator_pos):
        thrust_x = 0
        thrust_y = 0
        tot_angular_thrust = 0
        for i, ((thrust, angle), (x,y)) in enumerate(zip(actuator_ref, actuator_pos)):
            rad = np.deg2rad(90*(i+1)-angle)
            angle = angle * np.pi/180

            thrust_x += thrust * math.cos(angle)
            thrust_y += thrust * math.sin(angle)

            if x*y < 0:
                ang_thrust = abs(x) * thrust * np.cos(rad) + abs(y) * thrust * np.sin(rad)
            else:
                ang_thrust = abs(x) * thrust * np.sin(rad) + abs(y) * thrust * np.cos(rad)
            tot_angular_thrust += ang_thrust

        tot_thrust = math.hypot(thrust_x, thrust_y)
        tot_angle = math.atan2(thrust_y, thrust_x) * 180/np.pi

        return tot_thrust, tot_angle, tot_angular_thrust

    def _draw_explaination4(self, shap_values_action, shap_values_value, actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat, base_vectors, action_low, action_high):
        

        arr_RPM = np.array(shap_values_action)
        #print(arr_RPM)
        # arr_angles = np.array(shap_values_angles)
        shap_feature_RPM = arr_RPM[:, self.idx]
        summed_array = np.sum(arr_RPM, axis=1)
        # --- CLIP the reconstructed predictions ---

        reconstructed_pred_main_feature = np.clip(shap_feature_RPM + base_vectors, action_low, action_high) # Clip here
        #reconstructed_pred_main_feature = shap_feature_RPM + base_vectors # Clip here
        reconstructed_pred_total = np.clip(summed_array + base_vectors, action_low, action_high)          # Clip here


        thrusters, angles, self.prev_angle_1, vec = self._calculate_actuator_inputs(reconstructed_pred_main_feature, self.prev_angle_1)
        thrusters2, angles2, self.prev_angle_2, vec2 = self._calculate_actuator_inputs(reconstructed_pred_total, self.prev_angle_2)

        vector = self._combine_actuator_ref(vec, self.actuator_pos)
        vector2 = self._combine_actuator_ref(vec2, self.actuator_pos)
        # print('vector2')
        # print(vector2)
        # print('actuator_ref')
        # print(tot_thrust, tot_angle, tot_angular_thrust)
        # print('')
        # print('vec2')
        # print(vec2)
        # print('actuator_ref')
        # print(actuator_ref)
        # print('')


        # print(summed_array + base_vectors)
        # ##print(shap_feature_RPM)
        # #print('')
        # #print(base_vectors)
        # vec = np.zeros((len(shap_feature_RPM)//2,2))
        # vec2 = np.zeros((len(shap_feature_RPM)//2,2))
        # for i in range(len(shap_feature_RPM)//2):
        #     thr =  np.sqrt((shap_feature_RPM[i*2] + base_vectors[i*2])**2  + (shap_feature_RPM[i*2+1] + base_vectors[i*2+1])**2)
        #     thr2 =  np.sqrt((summed_array[i*2] + base_vectors[i*2])**2  + (summed_array[i*2+1] + base_vectors[i*2+1])**2)
        #     ang = np.arctan2((shap_feature_RPM[i*2+1] + base_vectors[i*2+1]),(shap_feature_RPM[i*2] + base_vectors[i*2]))*180/np.pi
        #     ang2 = np.arctan2((summed_array[i*2+1] + base_vectors[i*2+1]),(summed_array[i*2] + base_vectors[i*2]))*180/np.pi
        #     vec[i] = (thr, ang)
        #     vec2[i] = (thr2, ang2)

        # vector = self._add_vectors_polar(vec)
        # vector2 = self._add_vectors_polar(vec2)
        # #print(vector)
        #vector_base = self._add_vectors_polar(base_vector, vector)

        x = sum(shap_feature_RPM[i*2] for i in range(len(shap_feature_RPM)//2))
        y = sum(shap_feature_RPM[i*2+1] for i in range(len(shap_feature_RPM)//2))
        thrust = np.sqrt(x**2+y**2)
        angle = np.arctan2(y,x)
        ang_thrust = self._calculate_angular_thrust(shap_feature_RPM)
        #print(ang_thrust)


        # shap_feature_angles = arr_angles[:, self.idx] * 180
        # shap_tot_RPM = np.sum(arr_RPM, axis=1) * 1200
        # shap_tot_angles = np.sum(arr_angles, axis=1) * 180

        # shap_vectors = [(shap_feature_RPM[i], shap_feature_angles[i]) for i in range(4)]
        # shap_tot_vectors = [(shap_tot_RPM[i], shap_tot_angles[i]) for i in range(4)]

        # base_vector = self._add_vectors_polar(base_vectors)
        # main_feature_vector_base = self._add_vectors_polar(base_vectors, shap_vectors)
        # main_feature_vector = self._add_vectors_polar([main_feature_vector_base[:2],(-base_vector[0],base_vector[1])])
        # tot_vector_base = self._add_vectors_polar(base_vectors, shap_tot_vectors)
        # tot_vector = self._add_vectors_polar([tot_vector_base[:2],(-base_vector[0],base_vector[1])])
        # _,_,moment = self._calculate_ad(thrusters)


        radius = 30
        # if self.explain_vector:
        #     # Angular RPMS for all SHAP features (estimate of total angular RPM)
        #     self._draw_curved_arrow(self._scalar2pygame(tot_vector_base[2])/(1200*radius), self.center, Color.PURPLE.value, radius=radius+10)
        #     # Total angular RPM
        #self._draw_curved_arrow(self._scalar2pygame(tot_angular_thrust)/radius, self.center, Color.VELOCITY_GREEN.value, radius=radius+10)
        #self._draw_curved_arrow(np.clip(self._scalar2pygame(ang_thrust)/(900*radius),-2*np.pi,2*np.pi), self.center, Color.DESIRED_LIGHT_YELLOW.value, radius=radius, arrowhead_length=5, arrowhead_widht=5)
        #self._draw_curved_arrow(self._scalar2pygame(vector2[2])/radius, self.center, Color.PURPLE.value, radius=radius+10)
        self._draw_curved_arrow(self._scalar2pygame(vector[2])/radius, self.center, Color.DESIRED_LIGHT_YELLOW.value, radius=radius)

        #self._draw_curved_arrow(np.clip(self._scalar2pygame(vector2[2])/(900*radius),-2*np.pi,2*np.pi), self.center, Color.PURPLE.value, radius=radius+10)
        #self._draw_curved_arrow(np.clip(self._scalar2pygame(vector[2])/(900*radius),-2*np.pi,2*np.pi), self.center, Color.DESIRED_LIGHT_YELLOW.value, radius=radius, arrowhead_length=5, arrowhead_widht=5)
        #print(self._scalar2pygame(ang_thrust)/(900*radius))      
        # Base vector
        #self._draw_arrow(self._degrees2pygame(base_vector[1]), self._scalar2pygame(base_vector[0]/1200), Color.BLACK.value, self.center)
        
        # Convert both to Cartesian
        # x1 = self._scalar2pygame(base_vector[0]/1200) * np.cos(np.deg2rad(base_vector[1]))
        # y1 = self._scalar2pygame(base_vector[0]/1200) * np.sin(np.deg2rad(base_vector[1]))

        # Thruster vector from base for all SHAP features
        #self._draw_arrow(self._degrees2pygame(tot_vector[1]), self._scalar2pygame(tot_vector[0]/1200), Color.RED.value, (self.center[0]+y1,self.center[1]-x1))
        
        # if self.explain_vector:
        #     # Thruster vector for all SHAP features (estimate of totalt thruster vector)
        #     self._draw_arrow(self._degrees2pygame(tot_vector_base[1]), self._scalar2pygame(tot_vector_base[0]/1200), Color.PURPLE.value, self.center)
        #     # Total thruster vector
        #self._draw_arrow(self._degrees2pygame(tot_angle), self._scalar2pygame(tot_thrust), Color.VELOCITY_GREEN.value, self.center)
        
        # Thruster vector from base for single SHAP feature
        #self._draw_arrow(self._degrees2pygame(main_feature_vector[1]), self._scalar2pygame(main_feature_vector[0]/1200), Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+y1,self.center[1]-x1))
        # self._draw_arrow(self._degrees2pygame(main_feature_vector_base[1]), self._scalar2pygame(main_feature_vector_base[0]/1200), Color.DESIRED_LIGHT_YELLOW.value, self.center, arrowhead_width=5, arrowhead_length=5)
        #self._draw_arrow(self._degrees2pygame(angle*180/np.pi), self._scalar2pygame(thrust/900), Color.DESIRED_LIGHT_YELLOW.value, (self.center[0],self.center[1]))
        #self._draw_arrow(self._degrees2pygame(vector2[1]), self._scalar2pygame(vector2[0]), Color.PURPLE.value, self.center)
        self._draw_arrow(self._degrees2pygame(vector[1]), self._scalar2pygame(vector[0]), Color.DESIRED_LIGHT_YELLOW.value, self.center)
        
        #self._draw_arrow(self._degrees2pygame(vector2[1]), self._scalar2pygame(vector2[0]/900), Color.PURPLE.value, (self.center[0],self.center[1]))
        #self._draw_arrow(self._degrees2pygame(vector[1]), self._scalar2pygame(vector[0]/900), Color.DESIRED_LIGHT_YELLOW.value, (self.center[0],self.center[1]))
        #self._draw_arrow(self._degrees2pygame(vector_base[1]), self._scalar2pygame(vector_base[0]/900), Color.PURPLE.value, (self.center[0],self.center[1]))



        if self.idx == 0:
            self._draw_distance(self.surface, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), self._scalar2pygame(x_tilde), self._degrees2pygame(0), Color.RED.value, label=f"{x_tilde:.2f} m") # Explaination
        elif self.idx == 1:
            self._draw_distance(self.surface, (self.center[0], self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), self._scalar2pygame(y_tilde), self._degrees2pygame(90), Color.RED.value, label=f"{y_tilde:.2f} m") # Explaination
        elif self.idx == 2:
            self._draw_angle(self.surface, self.center, np.deg2rad(psi_tilde), Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, label=f"{psi_tilde:.0f} °")
        elif self.idx == 3:
            if u_hat > 0:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]-self._scalar2pygame(self.VESSEL_LENGTH/2)-self.explain_offset), show_measurement=True, label=f"{u_hat/2:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), show_measurement=True, label=f"{u_hat/2:.2f} m/s")
        elif self.idx == 4:
            if v_hat > 0:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), show_measurement=True, label=f"{v_hat/2:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]-self._scalar2pygame(self.VESSEL_WIDTH/2)-self.explain_offset,self.center[1]), show_measurement=True, label=f"{v_hat/2:.2f} m/s")
        elif self.idx == 5:
            self._draw_curved_arrow(np.deg2rad(r_hat), self.center, Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, show_measurement=True, label=f"{r_hat*112.6/(360*2):.0f} °/s")

        error = np.array([tot_thrust*900, tot_angle, tot_angular_thrust*900]) - np.array([vector[0]*900, vector[1], vector[2]*900])

        acc_RPM = (900*2*2-abs(error[0]))/(900*2*2)
        acc_angle = (360-abs(error[1]))/360
        acc_moment = (900*2*2*(1.8+0.8) - abs(error[2]))/(900*2*2*(1.8+0.8))

        acc_total = (acc_RPM+acc_angle+acc_moment)/3

        if acc_RPM > 0.75:
            acc_RPM_color = Color.GREEN.value
        elif acc_RPM > 0.5:
            acc_RPM_color = Color.ORANGE.value
        else:
            acc_RPM_color = Color.RED.value
        
        if acc_angle > 0.75:
            acc_angle_color = Color.GREEN.value
        elif acc_angle > 0.5:
            acc_angle_color = Color.ORANGE.value
        else:
            acc_angle_color = Color.RED.value
        
        if acc_moment > 0.75:
            acc_moment_color = Color.GREEN.value
        elif acc_moment > 0.5:
            acc_moment_color = Color.ORANGE.value
        else:
            acc_moment_color = Color.RED.value

        if acc_total > 0.75:
            acc_total_color = Color.GREEN.value
        elif acc_total > 0.5:
            acc_total_color = Color.ORANGE.value
        else:
            acc_total_color = Color.RED.value
     
     
        text_surface1 = self.window_font.render(f"Explained RPM: {acc_RPM:.2f}", self.antialias, acc_RPM_color)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (self._title_offset, self._title_offset*10)
        
        text_surface2 = self.window_font.render(f"Explained angle: {acc_angle:.2f}", self.antialias, acc_angle_color)
        text_rect2 = text_surface2.get_rect()
        text_rect2.topleft = (self._title_offset, self._title_offset*11+text_rect1.height)

        text_surface3 = self.window_font.render(f"Explained moment: {acc_moment:.2f}", self.antialias, acc_moment_color)
        text_rect3 = text_surface3.get_rect()
        text_rect3.topleft = (self._title_offset, self._title_offset*12+text_rect1.height+text_rect2.height)

        text_surface4 = self.window_font.render(f"Explained TOTAL: {acc_total:.2f}", self.antialias, acc_total_color)
        text_rect4 = text_surface4.get_rect()
        text_rect4.topleft = (self._title_offset, self._title_offset*14+text_rect1.height+text_rect2.height+text_rect3.height)
        
        self.surface.blit(text_surface1, text_rect1)
        self.surface.blit(text_surface2, text_rect2)
        self.surface.blit(text_surface3, text_rect3)
        self.surface.blit(text_surface4, text_rect4)


    def _draw_explaination3(self, error_x, error_y, error_psi, u_hat, v_hat, r_hat, shap_values_RPM, shap_values_angles, base_vectors, n_d, alpha_d, thrusters):
        

        arr_RPM = np.array(shap_values_RPM)
        arr_angles = np.array(shap_values_angles)
        shap_feature_RPM = arr_RPM[:, self.idx] * 1200
        shap_feature_angles = arr_angles[:, self.idx] * 180
        shap_tot_RPM = np.sum(arr_RPM, axis=1) * 1200
        shap_tot_angles = np.sum(arr_angles, axis=1) * 180

        shap_vectors = [(shap_feature_RPM[i], shap_feature_angles[i]) for i in range(4)]
        shap_tot_vectors = [(shap_tot_RPM[i], shap_tot_angles[i]) for i in range(4)]

        base_vector = self._add_vectors_polar(base_vectors)
        main_feature_vector_base = self._add_vectors_polar(base_vectors, shap_vectors)
        main_feature_vector = self._add_vectors_polar([main_feature_vector_base[:2],(-base_vector[0],base_vector[1])])
        tot_vector_base = self._add_vectors_polar(base_vectors, shap_tot_vectors)
        tot_vector = self._add_vectors_polar([tot_vector_base[:2],(-base_vector[0],base_vector[1])])
        _,_,moment = self._calculate_ad(thrusters)


        radius = 30
        if self.explain_vector:
            # Angular RPMS for all SHAP features (estimate of total angular RPM)
            self._draw_curved_arrow(self._scalar2pygame(tot_vector_base[2])/(1200*radius), self.center, Color.PURPLE.value, radius=radius+10)
            # Total angular RPM
            self._draw_curved_arrow(self._scalar2pygame(moment)/radius, self.center, Color.VELOCITY_GREEN.value, radius=radius+10)
        self._draw_curved_arrow(self._scalar2pygame(main_feature_vector_base[2])/(1200*radius), self.center, Color.DESIRED_LIGHT_YELLOW.value, radius=radius, arrowhead_length=5, arrowhead_widht=5)
              
        # Base vector
        #self._draw_arrow(self._degrees2pygame(base_vector[1]), self._scalar2pygame(base_vector[0]/1200), Color.BLACK.value, self.center)
        
        # Convert both to Cartesian
        # x1 = self._scalar2pygame(base_vector[0]/1200) * np.cos(np.deg2rad(base_vector[1]))
        # y1 = self._scalar2pygame(base_vector[0]/1200) * np.sin(np.deg2rad(base_vector[1]))

        # Thruster vector from base for all SHAP features
        #self._draw_arrow(self._degrees2pygame(tot_vector[1]), self._scalar2pygame(tot_vector[0]/1200), Color.RED.value, (self.center[0]+y1,self.center[1]-x1))
        
        if self.explain_vector:
            # Thruster vector for all SHAP features (estimate of totalt thruster vector)
            self._draw_arrow(self._degrees2pygame(tot_vector_base[1]), self._scalar2pygame(tot_vector_base[0]/1200), Color.PURPLE.value, self.center)
            # Total thruster vector
            self._draw_arrow(self._degrees2pygame(alpha_d), self._scalar2pygame(n_d), Color.VELOCITY_GREEN.value, self.center)
        
        # Thruster vector from base for single SHAP feature
        #self._draw_arrow(self._degrees2pygame(main_feature_vector[1]), self._scalar2pygame(main_feature_vector[0]/1200), Color.DESIRED_LIGHT_YELLOW.value, (self.center[0]+y1,self.center[1]-x1))
        self._draw_arrow(self._degrees2pygame(main_feature_vector_base[1]), self._scalar2pygame(main_feature_vector_base[0]/1200), Color.DESIRED_LIGHT_YELLOW.value, self.center, arrowhead_width=5, arrowhead_length=5)

        if self.idx == 0:
            self._draw_distance(self.surface, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), self._scalar2pygame(error_x), self._degrees2pygame(0), Color.RED.value, label=f"{error_x:.2f} m") # Explaination
        elif self.idx == 1:
            self._draw_distance(self.surface, (self.center[0], self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), self._scalar2pygame(error_y), self._degrees2pygame(90), Color.RED.value, label=f"{error_y:.2f} m") # Explaination
        elif self.idx == 2:
            self._draw_angle(self.surface, self.center, np.deg2rad(error_psi), Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, label=f"{error_psi:.0f} °")
        elif self.idx == 3:
            if u_hat > 0:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]-self._scalar2pygame(self.VESSEL_LENGTH/2)-self.explain_offset), show_measurement=True, label=f"{u_hat:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(0), self._scalar2pygame(u_hat), Color.RED.value, (self.center[0],self.center[1]+self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset), show_measurement=True, label=f"{u_hat:.2f} m/s")
        elif self.idx == 4:
            if v_hat > 0:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]+self._scalar2pygame(self.VESSEL_WIDTH/2)+self.explain_offset, self.center[1]), show_measurement=True, label=f"{v_hat:.2f} m/s")
            else:
                self._draw_arrow(self._degrees2pygame(90), self._scalar2pygame(v_hat), Color.RED.value, (self.center[0]-self._scalar2pygame(self.VESSEL_WIDTH/2)-self.explain_offset,self.center[1]), show_measurement=True, label=f"{v_hat:.2f} m/s")
        elif self.idx == 5:
            self._draw_curved_arrow(np.deg2rad(r_hat), self.center, Color.RED.value, radius=self._scalar2pygame(self.VESSEL_LENGTH/2)+self.explain_offset, show_measurement=True, label=f"{r_hat:.0f} °/s")

        error = np.array([n_d*1200, alpha_d, moment*1200]) - np.array([tot_vector_base[0], tot_vector_base[1], tot_vector_base[2]])

        acc_RPM = (2400*4-abs(error[0]))/(2400*4)
        acc_angle = (360-abs(error[1]))/360
        acc_moment = (2400*4*1.8 - abs(error[2]))/(2400*4*1.8)

        if acc_RPM > 0.75:
            acc_RPM_color = Color.GREEN.value
        elif acc_RPM > 0.5:
            acc_RPM_color = Color.ORANGE.value
        else:
            acc_RPM_color = Color.RED.value
        
        if acc_angle > 0.75:
            acc_angle_color = Color.GREEN.value
        elif acc_angle > 0.50:
            acc_angle_color = Color.ORANGE.value
        else:
            acc_angle_color = Color.RED.value
        
        if acc_moment > 0.75:
            acc_moment_color = Color.GREEN.value
        elif acc_moment > 0.50:
            acc_moment_color = Color.ORANGE.value
        else:
            acc_moment_color = Color.RED.value

        text_surface1 = self.window_font.render(f"Accuracy RPM: {acc_RPM:.2f}", self.antialias, acc_RPM_color)
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (self._title_offset, self._title_offset*10)
        
        text_surface2 = self.window_font.render(f"Accuracy angle: {acc_angle:.2f}", self.antialias, acc_angle_color)
        text_rect2 = text_surface2.get_rect()
        text_rect2.topleft = (self._title_offset, self._title_offset*11+text_rect1.height)

        text_surface3 = self.window_font.render(f"Accuracy moment: {acc_moment:.2f}", self.antialias, acc_moment_color)
        text_rect3 = text_surface3.get_rect()
        text_rect3.topleft = (self._title_offset, self._title_offset*12+text_rect1.height+text_rect2.height)
        
        self.surface.blit(text_surface1, text_rect1)
        self.surface.blit(text_surface2, text_rect2)
        self.surface.blit(text_surface3, text_rect3)


    def render(self, shap_values_action, shap_values_value, actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat, base_vectors, action_low, action_high):
        #print(1)
        # #self._find_explaination(shap_values_EV, n=1, list_in_list=False)
        # if explain_mode != self.explain_mode:
        #     self.explain_mode = explain_mode
        #     if self.RPM_true:
        #         self.RPM_true = False
        #         self._title = "Actions explained based on estimated values (BODY-frame)"
        #     else:
        #         self.RPM_true = True
        #         self._title = "Actions explained based on actions (BODY-frame)"
        # if self.RPM_true:
        #     shap_values = shap_values_RPM
        # else:
        #     shap_values = shap_values_EV

        # if explain_vector != self.explain_vector:
        #     self.explain_vector = explain_vector

        self._find_explaination(shap_values_action, n=1, list_in_list=self.RPM_true)
        #print(2)
        #self.idx=1
        # if self.idx != self.prev_idx:
        #     self.prev_idx = self.idx
        
        # init
        self.render_surface()
        #print(3)
        
        # SHAP explain render
        #self.surface.fill(Color.OCEAN_BLUE.value)
        self.draw_target(x_tilde, y_tilde, psi_tilde, color=Color.GRAY.value)
        #print(4)
        self.draw_vessel()
        #print(5)
        #self._draw_explaination(shap_values_EV, shap_values_RPM, thrusters, y_err, x_err, psi_err, u_hat, v_hat, r_hat)
        #self._draw_explaination2(y_err, x_err, psi_err, u_hat, v_hat, r_hat, shap_values_RPM, shap_values_angles, base_vectors, n_d, alpha_d, thrusters)
        self._draw_explaination4(shap_values_action, shap_values_value, actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat, base_vectors, action_low, action_high)
        #print(6)
        self.draw_legend(self.legend_surface, self.box_pos)
        #print(7)
        # final render
        self.render_surface_title()
        #print(8)
        self.render_window()    
        #print(9)



class ShapRender(Window):
    def __init__(self, screen, window_pos, legend_items, title="SHAP-value", window_width=450, window_height=450):
        super().__init__(screen, window_pos, title, window_width, window_height)

        # fonts
        self.axis_font =  pygame.font.SysFont('DejaVu Sans', 13)
        self.label_font =  pygame.font.SysFont('DejaVu Sans', 12)

        self.window_padding = 40 # pixels
        self.num_bars = 3 #14 # number of features
        self.num_bar_seg = 4 # number of outputs
        self.max_shap_value = 4
        self.increment = 1

        self.bar_gap = 50 # pixels
        self.label_offset = 10 # pixels
        self.label_surface = [None] * 14
        self.label_rect = [None] * 14
        self.bars_py = [None] * self.num_bars

        self.explain_mode = -1
        self.RPM_true = True

        self.feature_names = [
            'x̃ᵇₜ [m]',
            'ỹᵇₜ [m]',
            'ϕₜ [°]',
            'ûₜ [m/s]',
            'v̂ₜ [m/s]',
            'r̂ₜ [°/s]',
            'n_{x1d,ₜ₋₁} [RPM]',
            'n_{y1d,ₜ₋₁} [RPM]',
            'n_{x2d,ₜ₋₁} [RPM]',
            'n_{y2d,ₜ₋₁} [RPM]',
            'n_{x3d,ₜ₋₁} [RPM]',
            'n_{y3d,ₜ₋₁} [RPM]',
            'n_{x4d,ₜ₋₁} [RPM]',
            'n_{y4d,ₜ₋₁} [RPM]'
        ]

        # colors
        self.colors = [
            Color.TABLEAU_BLUE.value,
            Color.TABLEAU_ORANGE.value,
            Color.TABLEAU_GREEN.value,
            Color.TABLEAU_RED.value,
            Color.GREEN.value,
            Color.RED.value
        ]

        # making labels
        for i in range(14):
            self.label_surface[i] = self.label_font.render(self.feature_names[i], self.antialias, Color.BLACK.value)
            self.label_rect[i] = self.label_surface[i].get_rect()

        # defining axis padding width and bar allocation width
        self.axis_padding_width = max([self.label_rect[i].width for i in range(14)]) + self.label_offset
        self.bar_allocation_width = (self.window_width - self.window_padding - self.axis_padding_width) 
        
        self.label_surface = [None] * self.num_bars
        self.label_rect = [None] * self.num_bars

        # making labels
        for i in range(self.num_bars):
            self.label_surface[i] = self.label_font.render(self.feature_names[i], self.antialias, Color.BLACK.value)
            self.label_rect[i] = self.label_surface[i].get_rect()

        # making bottom text
        bottom_text = "Sum of |SHAP values|"
        self.bottom_text_surface = self.axis_font.render(bottom_text, self.antialias, Color.BLACK.value)
        self.bottom_text_rect = self.bottom_text_surface.get_rect()     
        self.bottom_text_rect.midbottom = ((self.window_width + self.axis_padding_width) / 2, self.window_height - self.window_padding / 2)
        
        # making ticks
        self.num_ticks = int(self.max_shap_value / self.increment) + 1
        self.ticks_px = [None] * self.num_ticks
        self.tick_label = [None] * self.num_ticks
        self.tick_rect = [None] * self.num_ticks
        for tick in range(self.num_ticks):
            tick_value = tick * self.increment
            self.ticks_px[tick] = self.window_padding / 2 + self.axis_padding_width + self._shap_value_2_pixels(tick_value)
            self.tick_label[tick] = self.label_font.render(f"{tick_value:.2f}", self.antialias, Color.BLACK.value)
            self.tick_rect[tick] = self.tick_label[tick].get_rect()
            self.tick_rect[tick].midbottom = (self.ticks_px[tick], self.bottom_text_rect.midtop[1] - 5)
        
        # defining axis padding height
        self.axis_padding_height = self.bottom_text_rect.height + 5 + self.tick_rect[0].height + self.label_offset
        
        # defining rest of bar parameters
        self.bar_allocation_hight = (self.window_height - self.window_padding - self.axis_padding_height) / self.num_bars
        self.bar_height = self.bar_allocation_hight - self.bar_gap
        
        # defining axis positions
        self.origo = (self.window_padding / 2 + self.axis_padding_width,
                      self.window_height - self.window_padding / 2 - self.axis_padding_height)
        self.x_axis_end = (self.window_width - self.window_padding / 2,
                           self.window_height - self.window_padding / 2 - self.axis_padding_height)
        self.y_axis_end = (self.window_padding / 2 + self.axis_padding_width,
                           self.window_padding / 2)
        
        # defining position for labels
        for i in range(self.num_bars):
            self.bars_py[i] = self.window_padding / 2 + i * self.bar_allocation_hight
            self.label_rect[i].topright = (self.window_padding / 2 + self.axis_padding_width - self.label_offset,
                                           self.bars_py[i] + (self.bar_height - self.label_rect[i].height) / 2)        

        
        self.legend_items = [(self.colors[i], legend) for i,legend in enumerate(legend_items)]        
        self.legend_surface, self.box_pos = self.create_legend(self.legend_items, self.label_font, margin=self.window_padding / 2)

    # public functions ############################

    def render(self, shap_values, shap_values3, base_value, action_low, action_high):
        # init
        self.render_surface()
        
        # SHAP render
        self._draw_bars(shap_values, shap_values3,base_value, action_low, action_high)
        self._draw_axis()
        self.draw_legend(self.legend_surface,self.box_pos)
        

        # final render
        self.render_surface_title()
        self.render_window()

    
    # private functions ############################
    
    def _shap_value_2_pixels(self, shap_value):
        return int(shap_value / self.max_shap_value * self.bar_allocation_width)
    
    def _find_top_n_abs_combined_indices(self, lists, n=3, list_in_list=True):
        combined_values = []
        if list_in_list:
            # Calculate the combined absolute values for each index
            for i in range(len(lists[0])):
                total = sum(np.sqrt(lists[2*j][i]**2 + lists[2*j+1][i]**2) for j in range(len(lists)//2))
                combined_values.append((i, total))  # Store (index, value) pairs
        else:
            for i in range(len(lists)):
                combined_values.append((i,-lists[i]))

        # Sort by combined value in descending order
        combined_values.sort(key=lambda x: x[1], reverse=True)
        
        # Get only the top n indices
        top_indices = [item[0] for item in combined_values[:n]]
        
        return top_indices

    def _draw_bars(self, shap_values,shap_value3,base_value, action_low, action_high):
        # if explain_mode != self.explain_mode:
        #     self.explain_mode = explain_mode
        #     if self.RPM_true:
        #         self.RPM_true = False
        #         self._title = "Most influential features based on estimated values (SHAP)"
        #     else:
        #         self.RPM_true = True
        #         self._title = "Most influential features based on actions (SHAP)"
        #self.RPM_true=False


        if self.RPM_true:
            shap_val = shap_values
        else:
            shap_val = shap_value3

        indices = self._find_top_n_abs_combined_indices(shap_val, list_in_list=self.RPM_true)

        for i,idx in enumerate(indices):

            bar_seg_px = self.window_padding / 2 + self.axis_padding_width

            shap = shap_value3[idx]      
 
            bar_seg_length = self._shap_value_2_pixels(shap)
            # if shap > 0:
            #     pygame.draw.rect(self.surface, Color.GREEN.value, (bar_seg_px, self.bars_py[i]+self.bar_gap, bar_seg_length, self.bar_height))
            # else:
            #     pygame.draw.rect(self.surface, Color.RED.value, (bar_seg_px, self.bars_py[i]+self.bar_gap, -bar_seg_length, self.bar_height))

            for bar_seg in range(self.num_bar_seg):
                #shap_value = abs(shap_values[bar_seg][idx])
                #print(1)
                arr_RPM = np.array(shap_values)
                reconstructed = np.clip(arr_RPM[:,idx] + base_value, action_low, action_high)
                #print(reconstructed)
                shap_value = np.clip(np.sqrt(reconstructed[2*bar_seg]**2+reconstructed[2*bar_seg+1]**2),0,1)
                #print(2.5)
                bar_seg_length = self._shap_value_2_pixels(shap_value)
                #print(3)
                pygame.draw.rect(self.surface, self.colors[bar_seg],
                                 (bar_seg_px, self.bars_py[i], bar_seg_length, self.bar_height))
                bar_seg_px += bar_seg_length
            
            self.label_surface[i] = self.label_font.render(self.feature_names[idx], self.antialias, Color.BLACK.value)
            self.label_rect[i] = self.label_surface[i].get_rect()
            self.label_rect[i].topright = (self.window_padding / 2 + self.axis_padding_width - self.label_offset,
                                           self.bars_py[i] + (self.bar_height - self.label_rect[i].height) / 2)

            self.surface.blit(self.label_surface[i], self.label_rect[i])

    def _draw_axis(self):
        pygame.draw.line(self.surface, Color.BLACK.value, self.origo, self.x_axis_end, width=2) # x-axis
        pygame.draw.line(self.surface, Color.BLACK.value, self.origo, self.y_axis_end, width=2) # y-axis

        for tick in range(self.num_ticks):
            pygame.draw.line(self.surface, Color.BLACK.value,
                             (self.ticks_px[tick], self.origo[1]-5), 
                             (self.ticks_px[tick], self.origo[1]+5), width=2)
            self.surface.blit(self.tick_label[tick], self.tick_rect[tick])

        self.surface.blit(self.bottom_text_surface, self.bottom_text_rect)


###########################################################################################################


class RenderExplaination():
    
    SCREEN_WIDTH = 975
    SCREEN_HEIGHT = 975
    TITLE = "Explainations"
    
    def __init__(self):
    
        self._screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(self.TITLE)

        self.shap_legend_items1 = (
            "n_{d₁,ₜ} [RPM]",
            "n_{d₂,ₜ} [RPM]",
            "n_{d₃,ₜ} [RPM]",
            "n_{d₄,ₜ} [RPM]",
            "+ estimated value",
            "- estimated value"
        )
        
        self.shap_legend_items2 = (
            "α_{d₁,ₜ} [°]",
            "α_{d₂,ₜ} [°]",
            "α_{d₃,ₜ} [°]",
            "α_{d₄,ₜ} [°]"
        )

        self._body_window = BodyRender(self._screen, window_pos=(25,25))
        self._ned_window = NedRender(self._screen, window_pos=(25,500))
        self._shap_window_top = ShapRender(self._screen, window_pos=(500,25),
                                           legend_items=self.shap_legend_items1,
                                           title="SHAP-values RPM")
        self._shap_window_bottom = ShapRender(self._screen, window_pos=(500,500),
                                              legend_items=self.shap_legend_items2,
                                              title="SHAP-values azimuth angles", window_width=475)
        
        self._shap_explain_window = ShapExplainRender(self._screen, window_pos=(500,500))



    def render_frame(self, shap_values_action, shap_values_value, actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat, base_vectors, action_low, action_high, target_heading):
        self._screen.fill(Color.SCREEN_COLOR.value)
        self._body_window.render(actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat)
        self._ned_window.render(x_tilde, y_tilde, psi_tilde, target_heading)
        self._shap_explain_window.render(shap_values_action, shap_values_value, actuator_ref, tot_thrust, tot_angle, tot_angular_thrust, x_tilde, y_tilde, psi_tilde, u_hat, v_hat, r_hat, base_vectors, action_low, action_high)

        self._shap_window_top.render(shap_values_action, shap_values_value, base_vectors, action_low, action_high)
        pygame.display.flip()
    # def render_frame(self, shap_values1, shap_values2, shap_values3, thrusters, n_d, alpha_d, u_hat, v_hat, r_hat, x_err, y_err, psi_err, explain_mode, base_vectors, explain_vector):
    #     s = sum(shap_values3)
    #     if s < -0.8:
    #         self._screen.fill(Color.TABLEAU_RED.value)
    #     else:
    #         self._screen.fill(Color.SCREEN_COLOR.value)

    #     self._body_window.render(thrusters, n_d, alpha_d, u_hat, v_hat, r_hat, x_err, y_err, psi_err)
    #     self._ned_window.render(x_err, y_err, psi_err)
    #     self._shap_window_top.render(shap_values1, shap_values3, explain_mode)
    #     #self._shap_window_bottom.render(shap_values2, shap_values3)
    #     self._shap_explain_window.render(shap_values3, shap_values1, thrusters, x_err, y_err, psi_err, u_hat, v_hat, r_hat, explain_mode, shap_values2, base_vectors, n_d, alpha_d, explain_vector)

    #     pygame.display.flip()

