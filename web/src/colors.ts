/**
 * Color constants matching milliampere_dp/rendering/colors.py.
 *
 * All values as CSS hex strings for Canvas 2D.
 */

export const Colors = {
  WHITE: "#ffffff",
  GRAY: "#969696",
  BLACK: "#000000",
  GREEN: "#00ff00",
  PURPLE: "#ff00ff",
  RED: "#ff0000",
  ORANGE: "#ff8000",

  SCREEN_COLOR: "#f0f0f0",

  TABLEAU_BLUE: "#5778a4",
  TABLEAU_ORANGE: "#e49444",
  TABLEAU_GREEN: "#6a9f58",
  TABLEAU_RED: "#d1615d",

  LEGEND_BOX: "rgba(200, 200, 200, 0.71)",

  AGENT_BLUE: "rgba(0, 122, 255, 0.86)",
  DESIRED_YELLOW: "#ff6600",
  DESIRED_LIGHT_YELLOW: "#ffcd80",
  OCEAN_BLUE: "#d1edff",
  OCEAN_GRID: "#b6cede",
  VELOCITY_GREEN: "#00ff7b",

  LIGHT_RED: "#ffabab",
} as const;
