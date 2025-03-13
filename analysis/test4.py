import pygame
from enum import Enum
import random
import math

class Colors(Enum):
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    GRAY = (150,150,150)
    BLUE = (0, 122, 255)
    YELLOW = (255, 155, 0)
    SHAP_RED = (255, 0, 93)


class Window:
    def __init__(self, screen, title="Window", width=450, height=450, x=0, y=0):
        self.screen = screen
        self.title = title
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont('DejaVu Sans', 12) 

    def render_window(self):
        self.surface.fill(Colors.WHITE.value)
        
    def render_window_titel(self):
        text_surface = self.font.render(self.title, True, Colors.GRAY.value)
        text_rect = text_surface.get_rect()
        text_rect.topright = (self.x + self.width - 5, self.y + 5)
        self.screen.blit(text_surface, text_rect)
    

class VesselRender(Window):
    
    # Render constants
    SCALE = 50
    FRAME = 10 # meters

    VESSEL_LENGTH = 5.06 # meters
    VESSEL_WIDTH = 2.86 # meters
    VESSEL_CORNER = 0.6 # meters

    # Agent
    AGENT_W = VESSEL_WIDTH * SCALE
    AGENT_H = VESSEL_LENGTH * SCALE
    AGENT_C = VESSEL_CORNER * SCALE
    AGENT_TRIANGLE_SIZE = 24
    AGENT_CIRCLE_RADIUS = 3    

    def __init__(self, screen, title, x, y):
        super().__init__(screen, title=title, x=x, y=y)
        self.center_x = self.width // 2
        self.center_y = self.height // 2

    def draw_arrow(self, angle, arrow_length, color, arrow_width=10, arrowhead_angle=math.pi/6, pct_explained=None):
        angle_rad = math.radians(-angle+90)

        if arrow_length < 0:
            arrow_length = abs(arrow_length)
            angle_rad += math.pi

        end_x = self.center_x + arrow_length * math.cos(angle_rad)
        end_y = self.center_y - arrow_length * math.sin(angle_rad)

        left_x = end_x - arrow_width * math.cos(angle_rad - arrowhead_angle)
        left_y = end_y + arrow_width * math.sin(angle_rad - arrowhead_angle)
        right_x = end_x - arrow_width * math.cos(angle_rad + arrowhead_angle)
        right_y = end_y + arrow_width * math.sin(angle_rad + arrowhead_angle)

        line_end_x = self.center_x + (arrow_length - arrow_width/2) * math.cos(angle_rad)
        line_end_y = self.center_y - (arrow_length - arrow_width/2) * math.sin(angle_rad)

        pygame.draw.line(self.surface, color, (self.center_x, self.center_y), (line_end_x, line_end_y), width=4)
        pygame.draw.polygon(self.surface, color, [(end_x, end_y), (left_x, left_y), (right_x, right_y)])

        if pct_explained:
            antialias = True
            color = Colors.SHAP_RED.value
            text = self.font.render(f"{pct_explained:.2f} %", antialias, color)
            self.surface.blit(text, ((self.center_x + end_x) // 2 - text.get_width() // 2, (self.center_y + end_y) // 2 - text.get_height() // 2))

    def draw_explanation(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, pct_explained, explain, color=Colors.SHAP_RED.value, line_end_len=5, text_offset=10):
        if explain == 4:
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
    
    def draw_vessel(self, center_x, center_y, heading, color=Colors.YELLOW.value, agent=True):
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
        super().__init__(screen, title, x, y)
    
    def render(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain):
        self.render_window()
        
        self.draw_vessel(0, 0, 0)
        self.draw_vessel(x_tilde, y_tilde, psi_tilde, agent=False)
        self.draw_explanation(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, pct_explained, explain)
        self.draw_action(n_d, alpha_d)

        self.screen.blit(self.surface, (self.x, self.y))
        self.render_window_titel()

class NedRender(VesselRender):
    def __init__(self, screen, x, y, title="NedRender",):
        super().__init__(screen, title, x, y)

    
    def render(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain):
        self.render_window()
        
        self.draw_vessel(-x_tilde, -y_tilde, -psi_tilde)
        self.draw_vessel(0, 0, 0, agent=False)
       
        self.screen.blit(self.surface, (self.x, self.y))
        self.render_window_titel()

class ShapRender(Window):
    def __init__(self, screen, x, y, title="ShapRender",):
        super().__init__(screen, title=title, x=x, y=y)

        # Settings
        self.widow_padding = 200
        self.axis_padding = 100
        self.num_bars = 14
        self.max_value = 3.0
        self.increment = 0.2

        self.bar_height = (self.height - self.widow_padding) // self.num_bars
        self.bar_gap = 5
        # placeholder
        self.bar_values = [round(random.uniform(0.00, self.max_value), 2) for _ in range(self.num_bars)]
        self.max_bar_length = self.width - self.widow_padding

        # self.feature_names = [r'$\tilde{x}^b_t$ [m]',r'$\tilde{y}^b_t$ [m]',
        #                       r'$\tilde{\phi}_t$ [$^\circ$]',r'$\hat{u}_t$ [m/s]',
        #                       r'$\hat{v}_t$ [m/s]',r'$\hat{r}_t$ [$^\circ$/s]',
        #                       r'$n_{d_1,t-1}$ [RPM]',r'$n_{d_2,t-1}$ [RPM]',
        #                       r'$n_{d_3,t-1}$ [RPM]',r'$n_{d_4,t-1}$ [RPM]',
        #                       r'$\alpha_{d_1,t-1}$ [$^\circ$]',r'$\alpha_{d_2,t-1}$ [$^\circ$]',
        #                       r'$\alpha_{d_3,t-1}$ [$^\circ$]',r'$\alpha_{d_4,t-1}$ [$^\circ$]']
        self.feature_names = [
            'x̄ᵇₜ [m]',  # x̄ with subscript bₜ
            'ȳᵇₜ [m]',  # ȳ with subscript bₜ
            'ϕₜ [°]',    # Phi with subscript t and degree symbol
            'ûₜ [m/s]',  # û with subscript t
            'ᵥₜ [m/s]',  # v with subscript t
            'r̂ₜ [°/s]', # r̂ with subscript t
            'n_{d₁,ₜ₋₁} [RPM]',  # n with d₁, t-1
            'n_{d₂,ₜ₋₁} [RPM]',
            'n_{d₃,ₜ₋₁} [RPM]',
            'n_{d₄,ₜ₋₁} [RPM]',
            'α_{d₁,ₜ₋₁} [°]',
            'α_{d₂,ₜ₋₁} [°]',
            'α_{d₃,ₜ₋₁} [°]',
            'α_{d₄,ₜ₋₁} [°]'
        ]

    def _draw_bars(self, bar_values):
        self.bar_values = bar_values

        for i in range(self.num_bars):
            bar_x = self.axis_padding
            bar_y = 50 + i * (self.bar_height + self.bar_gap)
            bar_length = int((self.bar_values[i] / self.max_value) * self.max_bar_length)
            pygame.draw.rect(self.surface, Colors.BLUE.value, (bar_x, bar_y, bar_length, self.bar_height))

            
            label_surface = self.font.render(self.feature_names[i], True, Colors.BLACK.value)
            label_rect = label_surface.get_rect()
            label_rect.topleft = (self.axis_padding - label_rect.width - 10, bar_y + (self.bar_height - label_rect.height) // 2)
            self.surface.blit(label_surface, label_rect)

    def _draw_axis(self):
        axis = 50 + self.num_bars * (self.bar_height + self.bar_gap) + 30
        pygame.draw.line(self.surface, Colors.BLACK.value, (self.axis_padding, 50), (self.axis_padding, axis), 2) # Y-axis
        pygame.draw.line(self.surface, Colors.BLACK.value, (self.axis_padding, axis), (self.width - 20, axis), 2) # X-axis

        num_ticks = int(self.max_value / self.increment) + 1
        for tick in range(num_ticks):
            tick_value = tick * self.increment
            tick_x = self.axis_padding + int((tick_value / self.max_value) * self.max_bar_length)
            pygame.draw.line(self.surface, Colors.BLACK.value, (tick_x, axis - 5), (tick_x, axis + 5), 2)
            tick_label = self.font.render(f"{tick_value:.2f}", True, Colors.BLACK.value)
            self.surface.blit(tick_label, (tick_x - 10, axis + 10))
        
        bottom_text = "Sum of |SHAP values|"
        bottom_text_surface = self.font.render(bottom_text, True, Colors.BLACK.value)
        bottom_text_rect = bottom_text_surface.get_rect()
        bottom_text_rect.center = (self.width // 2, axis + 40)
        self.surface.blit(bottom_text_surface, bottom_text_rect)

    def render(self, shap_values):
        self.render_window()
        
        self._draw_bars(shap_values)
        self._draw_axis()

        self.screen.blit(self.surface, (self.x, self.y))
        self.render_window_titel()



class RenderExplaination():
    def __init__(self, screen_width=1000, screen_height=1000, title="Explainations"):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(title)
        
        self.body_window = BodyRender(self.screen, x=25, y=25)
        self.ned_window = NedRender(self.screen, x=25, y=525)
        self.shap_window_top = ShapRender(self.screen, title="ShapRender_top", x=525, y=25)
        self.shap_window_bottom = ShapRender(self.screen, title="ShapRender_bottom", x=525, y=525)

    def render_frame(self, x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain, shap_values):
        # Fill the main screen
        self.screen.fill(Colors.BLACK.value)

        # Render each window
        self.body_window.render(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain)
        self.ned_window.render(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, pct_explained, explain)
        self.shap_window_top.render(shap_values)
        self.shap_window_bottom.render(shap_values)

        # Update the display
        pygame.display.flip()   
