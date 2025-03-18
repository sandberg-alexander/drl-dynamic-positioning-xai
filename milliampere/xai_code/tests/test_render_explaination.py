import unittest
import sys
import os
import pygame
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from milliampere.xai_code.render_explanation import RenderExplaination

class TestShapExplainRender(unittest.TestCase):
    def setUp(self):
        pygame.init()  # This is correct - initialize pygame first
        # Use a small hidden display to avoid opening a visible window
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        self.render_explaination = RenderExplaination()
        self.round = 11
        
    def tearDown(self):
        pygame.quit()
        
    def test_calulate_total_moment(self):
        vectors = [(1, 90), (1, 90), (1, 90), (1, 90)]
        self.assertAlmostEqual(round(self.render_explaination._shap_explain_window._calulate_total_moment(vectors), self.round), 0.0, self.round)
        
        vectors = [(1, 90), (1, 90), (1, -90), (-1, 90)]
        self.assertAlmostEqual(round(self.render_explaination._shap_explain_window._calulate_total_moment(vectors), self.round), 7.2, self.round)

        vectors = [(1, 0), (1, 0), (1, 0), (1, 0)]
        self.assertAlmostEqual(round(self.render_explaination._shap_explain_window._calulate_total_moment(vectors), self.round), 0.0, self.round)

        vectors = [(-1, 0), (-1, -180), (1, 0), (1, 180)]
        self.assertAlmostEqual(round(self.render_explaination._shap_explain_window._calulate_total_moment(vectors), self.round), -3.2, self.round)

        vectors = [(0.23, 42), (-0.79, -17), (0.1, -83), (0.13, 175)]
        self.assertAlmostEqual(round(self.render_explaination._shap_explain_window._calulate_total_moment(vectors), self.round), 1.47880595622, self.round)

     




if __name__ == '__main__':
    unittest.main()