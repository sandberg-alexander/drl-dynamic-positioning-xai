/**
 * TypeScript interfaces matching the Pydantic schemas in
 * milliampere_dp/rendering/frames.py.
 *
 * Field names use snake_case to match the msgpack keys from the server.
 */

export interface VesselState {
  x_tilde: number;
  y_tilde: number;
  psi_tilde: number;
  u_hat: number;
  v_hat: number;
  r_hat: number;
  target_pose: [number, number, number]; // (north_m, east_m, heading_deg)
  epsilon_ned: [number, number, number]; // (north_m, east_m, heading_deg)
}

export interface ActuatorState {
  actuator_ref: [number, number][]; // [(thrust_normalised, angle_deg)] per thruster
  tot_thrust: number;
  tot_angle: number;
  tot_angular_thrust: number;
}

export interface ShapFrame {
  shap_values_action: number[][]; // (8, 14) action SHAP values
  shap_values_value: number[]; // (14,) value function SHAP values
  base_vectors: number[]; // (8,) expected action values
  action_low: number[]; // (8,) action lower bounds
  action_high: number[]; // (8,) action upper bounds
}

export interface RenderFrame {
  vessel: VesselState;
  actuators: ActuatorState;
  shap: ShapFrame;
  time_step: number;
  time_seconds: number;
}

export interface KeyboardInput {
  key: number; // deploy mode 0-5
}

/** Messages from main thread to render worker. */
export type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas }
  | { type: "frame"; frame: RenderFrame }
  | { type: "resize"; width: number; height: number };
