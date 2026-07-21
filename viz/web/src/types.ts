// Data contracts mirror viz/inference.py (_graph_meta / _tick_record).

export type NodeKind = "farm" | "hub" | "dc" | "retail";

export interface GraphNode {
  name: string;
  kind: NodeKind;
}

export interface Thresholds {
  optimal_temp_low: number;
  optimal_temp_high: number;
  chill_injury: number;
  optimal_humidity_low: number;
  optimal_humidity_high: number;
}

export interface Meta {
  type: "meta";
  fruit: string;
  source: string;
  target: string;
  max_steps: number;
  nodes: GraphNode[];
  edges: [string, string][];
  restock_paths: string[][];
  thresholds: Thresholds;
  n_windows: number;
  horizon?: number;
  mediator?: string;
}

export interface Shipment {
  current_node: string;
  target_node: string;
  age_ticks: number;
  spoilage_risk: number;
  freshness_score: number;
  sensor_temp: number;
  desired_temp: number;
  sensor_humidity: number;
}

export interface Vehicle {
  assigned_node: string;
  chosen_slot: number;
  busy_until: number;
  delay: number;
  sla_violated: boolean;
  conflict: boolean;
  current_node: string;
  carrying: number | null;
  route_transit: number;
  route_emissions: number;
  sla_window_ticks: number;
  emissions: number;
}

export interface Cargo {
  vehicle: number;
  instance: number;
  departure_tick: number;
  arrival_tick: number;
  qty: number;
}

export interface Inventory {
  levels: number[];
  order: number[];
  unmet: number[];
  demand_today: number[];
  forecast: number[];
}

export interface NegotiationEvent {
  slot: number;
  agents: string[];
  mediator: string;
  initial: Record<string, number>;
  final: Record<string, number>;
  costs?: Record<string, number>;
  resolved: boolean;
  rounds: number;
  summary: string;
}

export interface Tick {
  type: "tick";
  tick: number;
  shipment_no?: number;
  actions: Record<string, unknown>;
  infos: Record<string, Record<string, number>>;
  shipment: Shipment;
  ambient: { weather: string; temp: number; humidity: number };
  calendar: { day_of_year: number; weekday: number; event_multiplier: number };
  inventory: Inventory;
  cargo: Cargo[];
  order_queue: [number, number][];
  vehicles: Vehicle[];
  disruptions: { type: string; target: string }[];
  spoilage_prediction: number;
  energy_usage: number;
  rewards: Record<string, number>;
  negotiations?: NegotiationEvent[];
}

export interface Episode {
  name: string;
  meta: Meta;
  ticks: Tick[];
}

export const KIND_ORDER: NodeKind[] = ["farm", "hub", "dc", "retail"];
export const KIND_LABEL: Record<NodeKind, string> = {
  farm: "farm",
  hub: "hub",
  dc: "DC",
  retail: "retail",
};
export const KIND_COLOR: Record<NodeKind, string> = {
  farm: "#16a34a",
  hub: "#2563eb",
  dc: "#d97706",
  retail: "#db2777",
};
