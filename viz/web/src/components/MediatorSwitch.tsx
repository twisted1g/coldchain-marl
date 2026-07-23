export type Mediator = "greedy" | "llm";

const OPTIONS: { value: Mediator; label: string }[] = [
  { value: "greedy", label: "greedy" },
  { value: "llm", label: "LLM" },
];

/**
 * Two-way switch for the delivery-slot conflict solver (paper Alg 6): a fast
 * deterministic greedy reassignment vs the LLM mediator. In live mode flipping it
 * restarts the stream with the new mediator so the change is visible immediately.
 */
export function MediatorSwitch({
  value,
  onChange,
  disabled,
}: {
  value: Mediator;
  onChange: (m: Mediator) => void;
  disabled?: boolean;
}) {
  return (
    <label className="field">
      conflict solver
      <div className="seg" role="group" aria-label="conflict mediator">
        {OPTIONS.map((o) => (
          <button
            key={o.value}
            type="button"
            className={value === o.value ? "seg-on" : ""}
            aria-pressed={value === o.value}
            disabled={disabled}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </label>
  );
}
