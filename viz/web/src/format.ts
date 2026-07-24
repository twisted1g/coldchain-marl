export const AGENT_ORDER = [
  "temperature",
  "routing",
  "spoilage",
  "inventory_0",
  "inventory_1",
  "inventory_2",
  "inventory_3",
  "delivery_0",
  "delivery_1",
  "delivery_2",
];

export const AGENT_GROUP: Record<string, string> = {
  temperature: "temperature",
  routing: "routing",
  spoilage: "spoilage",
  inventory_0: "inventory",
  inventory_1: "inventory",
  inventory_2: "inventory",
  inventory_3: "inventory",
  delivery_0: "delivery",
  delivery_1: "delivery",
  delivery_2: "delivery",
};

export const GROUP_COLOR: Record<string, string> = {
  temperature: "#0ea5e9",
  routing: "#8b5cf6",
  spoilage: "#f43f5e",
  inventory: "#f59e0b",
  delivery: "#10b981",
};

export function fmt(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return "–";
  return x.toFixed(digits);
}

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
