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

    def draw_vessel(self, x_err, y_err, psi_err, ned=False):
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

    def draw_target(self, x_err, y_err, psi_err, ned=False):
        if ned:
            shape = self._transform_vessel(0, 0, 45, self.shape)
            #shape = self.shape
            circle = self.circle_point
            #triangle = self.triangle_shape
            triangle = self._transform_vessel(0, 0, 45, self.triangle_shape)

        else:
            shape = self._transform_vessel(x_err, y_err, psi_err, self.shape)
            circle = self._transform_vessel(x_err, y_err, psi_err, self.circle_point)
            triangle = self._transform_vessel(x_err, y_err, psi_err, self.triangle_shape)

        points = self._world_2_pixels(shape)
        triangle_points = self._world_2_pixels(triangle)

        for i in range(len(shape)):
            self._draw_dashed_line(points[i], points[(i+1) % len(points)], Color.DESIRED_YELLOW.value)
            if i < len(triangle):
                self._draw_dashed_line(triangle_points[i], triangle_points[(i+1) % len(triangle_points)], Color.DESIRED_YELLOW.value, dash_lenght=5)
            
        
        #pygame.draw.polygon(self.surface, Color.DESIRED_YELLOW.value, self._world_2_pixels(triangle))
        pygame.draw.circle(self.surface, Color.DESIRED_YELLOW.value, self._world_2_pixels(circle).ravel(), radius=3)
        

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
    THRUSTER_X = 1.8
    THRUSTER_Y = 0.8

    def __init__(self, screen, window_pos, title="BODY-frame", window_width=450, window_height=450):
        super().__init__(screen, window_pos, self.SCALE, title, window_width, window_height)
        
        self.label_font =  pygame.font.SysFont('DejaVu Sans', 12)
        self.body_legend_items = (
            (Color.DESIRED_YELLOW.value, "Desired {'total thrust force', 'total trust moment', 'pose'}"),
            (Color.DESIRED_LIGHT_YELLOW.value, "Desired thrust force"),
            (Color.VELOCITY_GREEN.value, "{'surge', 'sway', 'angular'} velocity")
        )

        self.legend_surface, self.box_pos = self.create_legend(self.body_legend_items, self.label_font, padding_bottom=0)

        self.thruster_positions = np.array([[self.THRUSTER_X, -self.THRUSTER_Y],
                                            [self.THRUSTER_X, self.THRUSTER_Y],
                                            [-self.THRUSTER_X, self.THRUSTER_Y],
                                            [-self.THRUSTER_X, -self.THRUSTER_Y]])
    
    def _draw_thrusters(self, vectors):
        for i, (x,y) in enumerate(self.thruster_positions):
            px,py = self._world_2_pixels(np.array([[x,y]])).ravel()
            angle = self._degrees2pygame(vectors[i][1])
            perp_angle = angle + np.pi/2
            self.draw_arrow(self._degrees2pygame(vectors[i][1]), self._scalar2pygame(vectors[i][0]),
                            Color.DESIRED_LIGHT_YELLOW.value, np.array([px,py])) 
            pygame.draw.line(self.surface, Color.DESIRED_LIGHT_YELLOW.value,
                             (px - 10 * np.cos(perp_angle), py - 10 * np.sin(perp_angle)),
                             (px + 10 * np.cos(perp_angle), py + 10 * np.sin(perp_angle)), width=2)
            pygame.draw.circle(self.surface, Color.BLACK.value, (px,py), radius=3)

    def _draw_force_and_moment(self, n_d, alpha_d, thrusters):
        
        center = self._world_2_pixels(np.zeros((1,2))).ravel()
        self.draw_arrow(self._degrees2pygame(alpha_d), self._scalar2pygame(n_d), Color.DESIRED_YELLOW.value, center)
        moment_d = self.calulate_total_moment(thrusters)
        self._draw_curved_arrow(np.deg2rad(moment_d), center, Color.DESIRED_YELLOW.value, self._scalar2pygame(self.VESSEL_MOMENT_MARKER))


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
        total_moment = 0.0
        for (x,y), (n, alpha) in zip(self.thruster_positions, thrusters):
            F = n * np.array([np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])
            M =  self._scalar2pygame(x) * F[1] + self._scalar2pygame(y) * F[0]
            total_moment += M
        return total_moment

    def render(self, thrusters, n_d, alpha_d, u_hat, v_hat, r_hat, x_err, y_err, psi_err):
        # init
        self.render_surface()
        
        # BODY render
        self.surface.fill(Color.OCEAN_BLUE.value)
        self.draw_target(y_err, x_err, psi_err)
        self.draw_vessel(y_err, x_err, psi_err)
        self._draw_thrusters(thrusters)
        self._draw_force_and_moment(n_d, alpha_d, thrusters)
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
    
    def render(self, x_err, y_err, psi_err):
        # init
        self.render_surface()
        
        # NED render
        self.surface.fill(Color.OCEAN_BLUE.value)
        self.draw_target(y_err, x_err, psi_err, ned=True)
        self.draw_vessel(y_err, x_err, psi_err-45, ned=True)

        # final render
        self.render_surface_title()
        self.render_window()


###########################################################################################################


class ShapRender(Window):
    def __init__(self, screen, window_pos, legend_items, title="SHAP-value", window_width=450, window_height=450):
        super().__init__(screen, window_pos, title, window_width, window_height)

        # fonts
        self.axis_font =  pygame.font.SysFont('DejaVu Sans', 13)
        self.label_font =  pygame.font.SysFont('DejaVu Sans', 12)

        self.window_padding = 40 # pixels
        self.num_bars = 14 # number of features
        self.num_bar_seg = 4 # number of outputs
        self.max_shap_value = 5
        self.increment = 1

        self.bar_gap = 5 # pixels
        self.label_offset = 10 # pixels
        self.label_surface = [None] * self.num_bars
        self.label_rect = [None] * self.num_bars
        self.bars_py = [None] * self.num_bars

        self.feature_names = [
            'x̃ᵇₜ [m]',
            'ỹᵇₜ [m]',
            'ϕₜ [°]',
            'ûₜ [m/s]',
            'v̂ₜ [m/s]',
            'r̂ₜ [°/s]',
            'n_{d₁,ₜ₋₁} [RPM]',
            'n_{d₂,ₜ₋₁} [RPM]',
            'n_{d₃,ₜ₋₁} [RPM]',
            'n_{d₄,ₜ₋₁} [RPM]',
            'α_{d₁,ₜ₋₁} [°]',
            'α_{d₂,ₜ₋₁} [°]',
            'α_{d₃,ₜ₋₁} [°]',
            'α_{d₄,ₜ₋₁} [°]'
        ]

        # colors
        self.colors = [
            Color.TABLEAU_BLUE.value,
            Color.TABLEAU_ORANGE.value,
            Color.TABLEAU_GREEN.value,
            Color.TABLEAU_RED.value
        ]

        # making labels
        for i in range(self.num_bars):
            self.label_surface[i] = self.label_font.render(self.feature_names[i], self.antialias, Color.BLACK.value)
            self.label_rect[i] = self.label_surface[i].get_rect()

        # defining axis padding width and bar allocation width
        self.axis_padding_width = max([self.label_rect[i].width for i in range(self.num_bars)]) + self.label_offset
        self.bar_allocation_width = (self.window_width - self.window_padding - self.axis_padding_width) 

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

    def render(self, shap_values, shap_values3):
        # init
        self.render_surface()
        
        # SHAP render
        self._draw_bars(shap_values, shap_values3)
        self._draw_axis()
        self.draw_legend(self.legend_surface,self.box_pos)

        # final render
        self.render_surface_title()
        self.render_window()

    
    # private functions ############################
    
    def _shap_value_2_pixels(self, shap_value):
        return int(shap_value / self.max_shap_value * self.bar_allocation_width)
    
    def _draw_bars(self, shap_values,shap_value3):
        for i in range(self.num_bars):
            bar_seg_px = self.window_padding / 2 + self.axis_padding_width

            shap = shap_value3[i]            
            bar_seg_length = self._shap_value_2_pixels(shap)
            if shap < 0:
                pygame.draw.rect(self.surface, Color.GREEN.value, (bar_seg_px+bar_seg_length, self.bars_py[i]+5, -bar_seg_length, self.bar_height))
            else:
                pygame.draw.rect(self.surface, Color.GREEN.value, (bar_seg_px, self.bars_py[i]+5, bar_seg_length, self.bar_height))

            for bar_seg in range(self.num_bar_seg):
                shap_value = abs(shap_values[bar_seg][i])
                bar_seg_length = self._shap_value_2_pixels(shap_value)
                pygame.draw.rect(self.surface, self.colors[bar_seg],
                                 (bar_seg_px, self.bars_py[i], bar_seg_length, self.bar_height))
                bar_seg_px += bar_seg_length

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
    
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 975
    TITLE = "Explainations"
    
    def __init__(self):
    
        self._screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(self.TITLE)

        self.shap_legend_items1 = (
            "n_{d₁,ₜ} [RPM]",
            "n_{d₂,ₜ} [RPM]",
            "n_{d₃,ₜ} [RPM]",
            "n_{d₄,ₜ} [RPM]"
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
                                           title="SHAP-values RPM", window_width=475)
        self._shap_window_bottom = ShapRender(self._screen, window_pos=(500,500),
                                              legend_items=self.shap_legend_items2,
                                              title="SHAP-values azimuth angles", window_width=475)



    def render_frame(self, shap_values1, shap_values2, shap_values3, thrusters, n_d, alpha_d, u_hat, v_hat, r_hat, x_err, y_err, psi_err):
        s = sum(shap_values3)
        if s < -0.8:
            self._screen.fill(Color.TABLEAU_RED.value)
        else:
            self._screen.fill(Color.SCREEN_COLOR.value)

        self._body_window.render(thrusters, n_d, alpha_d, u_hat, v_hat, r_hat, x_err, y_err, psi_err)
        self._ned_window.render(x_err, y_err, psi_err)
        self._shap_window_top.render(shap_values1, shap_values3)
        self._shap_window_bottom.render(shap_values2, shap_values3)

        pygame.display.flip()