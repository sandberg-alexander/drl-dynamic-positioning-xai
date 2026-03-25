"""SHAP bar chart window showing per-observation feature contributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame
from milliampere_dp.rendering import Color
from milliampere_dp.vessel import MAX_THRUSTER_RPM

from milliampere_xai.rendering._window import Window


class ShapRender(Window):
    """SHAP bar chart window showing per-observation feature contributions."""

    def __init__(
        self,
        screen,
        window_pos,
        legend_items,
        title="SHAP-value",
        window_width=450,
        window_height=450,
        renderer=None,
    ):
        super().__init__(
            screen, window_pos, title, window_width, window_height, renderer=renderer
        )

        # fonts
        self.axis_font = pygame.font.SysFont("DejaVu Sans", 13)
        self.label_font = pygame.font.SysFont("DejaVu Sans", 12)

        self.window_padding = 40  # pixels
        self.num_bars = 4  # 14 # number of features
        self.num_bar_seg = 4  # number of outputs
        self.max_shap_value = 4
        self.increment = 1

        self.bar_gap = 25  # pixels
        self.label_offset = 10  # pixels
        self.label_surface: list[Any] = [None] * 14
        self.label_rect: list[Any] = [None] * 14
        self.bars_py: list[Any] = [None] * self.num_bars

        self.explain_mode = -1
        self.RPM_true = True

        self.feature_names = [
            "\u1e8d\u1d47\u209c [m]",
            "\u1ef3\u1d47\u209c [m]",
            "\u03c8\u0303\u209c[\u00b0]",
            "\u00fb\u209c [m/s]",
            "\u1e7d\u0302\u209c [m/s]",
            "r\u0302\u209c [\u00b0/s]",
            "n_{\u2093\u2081,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u1d67\u2081,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u2093\u2082,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u1d67\u2082,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u2093\u2083,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u1d67\u2083,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u2093\u2084,d,\u209c\u208b\u2081} [RPM]",
            "n_{\u1d67\u2084,d,\u209c\u208b\u2081} [RPM]",
        ]
        self.feature_names_full = self.feature_names[:]  # <- new

        # colors
        self.colors = [
            Color.TABLEAU_BLUE.value,
            Color.TABLEAU_ORANGE.value,
            Color.TABLEAU_GREEN.value,
            Color.TABLEAU_RED.value,
            Color.GREEN.value,
            Color.RED.value,
        ]

        # making labels
        for i in range(14):
            self.label_surface[i] = self.label_font.render(
                self.feature_names[i], self.antialias, Color.BLACK.value
            )
            self.label_rect[i] = self.label_surface[i].get_rect()

        # defining axis padding width and bar allocation width
        self.axis_padding_width = (
            max([self.label_rect[i].width for i in range(14)]) + self.label_offset
        )
        self.bar_allocation_width = (
            self.window_width - self.window_padding - self.axis_padding_width
        )

        self.label_surface: list[Any] = [None] * self.num_bars
        self.label_rect: list[Any] = [None] * self.num_bars

        # making labels
        for i in range(self.num_bars):
            self.label_surface[i] = self.label_font.render(
                self.feature_names[i], self.antialias, Color.BLACK.value
            )
            self.label_rect[i] = self.label_surface[i].get_rect()

        # making bottom text
        bottom_text = "Sum of SHAP based thrust components"
        self.bottom_text_surface = self.axis_font.render(
            bottom_text, self.antialias, Color.BLACK.value
        )
        self.bottom_text_rect = self.bottom_text_surface.get_rect()
        self.bottom_text_rect.midbottom = (
            (self.window_width + self.axis_padding_width) / 2,
            self.window_height - self.window_padding / 2,
        )

        # making ticks
        self.num_ticks = int(self.max_shap_value / self.increment) + 1
        self.ticks_px: list[Any] = [None] * self.num_ticks
        self.tick_label: list[Any] = [None] * self.num_ticks
        self.tick_rect: list[Any] = [None] * self.num_ticks
        for tick in range(self.num_ticks):
            tick_value = tick * self.increment
            self.ticks_px[tick] = (
                self.window_padding / 2
                + self.axis_padding_width
                + self._shap_value_2_pixels(tick_value)
            )
            self.tick_label[tick] = self.label_font.render(
                f"{tick_value * MAX_THRUSTER_RPM}", self.antialias, Color.BLACK.value
            )
            self.tick_rect[tick] = self.tick_label[tick].get_rect()
            self.tick_rect[tick].midbottom = (
                self.ticks_px[tick],
                self.bottom_text_rect.midtop[1] - 5,
            )

        # defining axis padding height
        self.axis_padding_height = (
            self.bottom_text_rect.height
            + 5
            + self.tick_rect[0].height
            + self.label_offset
        )

        # defining rest of bar parameters
        self.bar_allocation_hight = (
            self.window_height - self.window_padding - self.axis_padding_height
        ) / self.num_bars
        self.bar_height = self.bar_allocation_hight - self.bar_gap

        # defining axis positions
        self.origo = (
            self.window_padding / 2 + self.axis_padding_width,
            self.window_height - self.window_padding / 2 - self.axis_padding_height,
        )
        self.x_axis_end = (
            self.window_width - self.window_padding / 2,
            self.window_height - self.window_padding / 2 - self.axis_padding_height,
        )
        self.y_axis_end = (
            self.window_padding / 2 + self.axis_padding_width,
            self.window_padding / 2,
        )

        # defining position for labels
        for i in range(self.num_bars):
            self.bars_py[i] = self.window_padding / 2 + i * self.bar_allocation_hight
            self.label_rect[i].topright = (
                self.window_padding / 2 + self.axis_padding_width - self.label_offset,
                self.bars_py[i] + (self.bar_height - self.label_rect[i].height) / 2,
            )

        self.legend_items = [
            (self.colors[i], legend) for i, legend in enumerate(legend_items)
        ]
        self.legend_surface, self.box_pos = self.create_legend(
            self.legend_items, self.label_font, margin=self.window_padding / 2
        )

    # public functions ############################

    def render(self, shap_values, shap_values3, base_value, action_low, action_high):
        # init
        self.render_surface()

        # SHAP render
        self._draw_bars2(shap_values, shap_values3, base_value, action_low, action_high)
        self._draw_axis()
        self.draw_legend(self.legend_surface, self.box_pos)

        # final render
        self.render_surface_title()
        self.render_window()

    # private functions ############################

    def _shap_value_2_pixels(self, shap_value):
        return int(shap_value / self.max_shap_value * self.bar_allocation_width)

    def _draw_bars2(
        self, shap_values_action, shap_values_vf, base_value, action_low, action_high
    ):
        """
        Draw four horizontal bars:
        \u2013 the three most influential individual features
        \u2013 one aggregated \u201cOthers\u201d bar
        """

        # shap_values_action (8x14)
        # shap_values_vf (14)

        sv_action = np.array(shap_values_action)
        sv_base = np.array(base_value)
        act_low = np.array(action_low)
        act_high = np.array(action_high)

        sv_all_actions = np.clip(sv_action.sum(axis=1) + sv_base, act_low, act_high)

        pos_totalt = np.where(sv_action > 0, sv_action, 0).sum(axis=1) + np.where(
            sv_base > 0, sv_base, 0
        )
        neg_totalt = np.where(sv_action < 0, sv_action, 0).sum(axis=1) + np.where(
            sv_base < 0, sv_base, 0
        )

        denominators = np.where(act_high == 1, pos_totalt, neg_totalt)

        c = np.zeros_like(sv_all_actions, dtype=float)

        np.divide(
            sv_all_actions, denominators, out=c, where=denominators != 0
        )  # Avoid division by zero

        # sv_action: shape (8, 14)
        # c:          shape (8,)
        # We\u2019ll treat (0,1), (2,3), (4,5), (6,7) as your (x,y) pairs.

        # 1) First, reshape into \u201cpairs\u201d:
        num_actions, num_feats = sv_action.shape
        assert num_actions % 2 == 0, "Need an even number of action\u2010rows"
        sv_pairs = sv_action.reshape(-1, 2, num_feats)  # shape (4, 2, 14)
        c_pairs = c.reshape(-1, 2)  # shape (4, 2)

        # 2) Scale each x\u2010row by its c and each y\u2010row by its c:
        #    we broadcast c_pairs (4,2) over the features\u2010axis
        scaled = sv_pairs * c_pairs[:, :, None]  # shape (4, 2, 14)

        # 3) Square + sum over the x/y axis, then sqrt \u2192 Euclidean distance
        # sq       = scaled ** 2                           # shape (4, 2, 14)
        # sum_sq   = sq.sum(axis=1)                        # shape (4, 14)
        # dists    = np.sqrt(sum_sq)                       # shape (4, 14)
        dists = np.hypot(scaled[:, 0, :], scaled[:, 1, :])

        # 4) Clip each distance between 0 and 1
        dists_clipped = np.clip(dists, 0, 1)  # shape (4, 14)
        sumed_features = dists_clipped.sum(axis=0)

        top3_idx = np.argpartition(sumed_features, -3)[-3:]  # get any order of top3
        top3_idx = top3_idx[
            np.argsort(-sumed_features[top3_idx])
        ]  # then sort those three
        top3_idx = list(top3_idx)
        # print(f"top3_idx:{top3_idx}")

        others_idx = [
            i for i in range(len(self.feature_names_full)) if i not in top3_idx
        ]

        draw_order = top3_idx + [-1]  # \u20131 will mean \u201cOTHERS\u201d

        # --------------------------------------------------
        # 3. loop over the four bars to draw
        # --------------------------------------------------
        for bar_no, feat_idx in enumerate(draw_order):
            # ---------- build the 4 coloured segments ----------
            seg_width_px = []  # length-4 list of pixel widths
            for seg in range(self.num_bar_seg):
                if feat_idx == -1:  # -------- OTHERS ------------
                    # sum this segment's contribution over *all* residual features
                    seg_val = 0.0
                    for jj in others_idx:
                        seg_val += dists_clipped[seg, jj]
                else:  # -------- single feature ----
                    seg_val = dists_clipped[seg, feat_idx]
                seg_width_px.append(self._shap_value_2_pixels(seg_val))

            # ---------- actually draw the bar ----------
            x0 = self.window_padding / 2 + self.axis_padding_width
            y0 = self.bars_py[bar_no]
            for seg, w in enumerate(seg_width_px):
                self._renderer.draw_rect(
                    self.surface, self.colors[seg], (x0, y0, w, self.bar_height)
                )
                x0 += w  # next segment starts where the last finished

            # ---------- draw the label ----------
            label = "Others" if feat_idx == -1 else self.feature_names_full[feat_idx]
            self.label_surface[bar_no] = self.label_font.render(
                label, self.antialias, Color.BLACK.value
            )
            self.label_rect[bar_no] = self.label_surface[bar_no].get_rect()
            self.label_rect[bar_no].topright = (
                self.window_padding / 2 + self.axis_padding_width - self.label_offset,
                y0 + (self.bar_height - self.label_rect[bar_no].height) / 2,
            )
            self.surface.blit(self.label_surface[bar_no], self.label_rect[bar_no])

    def _draw_axis(self):
        self._renderer.draw_line(
            self.surface, Color.BLACK.value, self.origo, self.x_axis_end, width=2
        )  # x-axis
        self._renderer.draw_line(
            self.surface, Color.BLACK.value, self.origo, self.y_axis_end, width=2
        )  # y-axis

        for tick in range(self.num_ticks):
            self._renderer.draw_line(
                self.surface,
                Color.BLACK.value,
                (self.ticks_px[tick], self.origo[1] - 5),
                (self.ticks_px[tick], self.origo[1] + 5),
                width=2,
            )
            self.surface.blit(self.tick_label[tick], self.tick_rect[tick])

        self.surface.blit(self.bottom_text_surface, self.bottom_text_rect)
