interface Props {
  index: number;
  count: number;
  playing: boolean;
  onIndex: (i: number) => void;
  onTogglePlay: () => void;
  tickLabel?: string;
}

/** Episode transport: play/pause + step, plus a scrubber over all ticks. */
export function StepBar({ index, count, playing, onIndex, onTogglePlay, tickLabel }: Props) {
  const last = Math.max(0, count - 1);
  const clamp = (i: number) => Math.max(0, Math.min(last, i));

  return (
    <div className="stepbar">
      <div className="stepbtns">
        <button onClick={() => onIndex(clamp(index - 1))} disabled={index <= 0} title="previous tick">
          ◀
        </button>
        <button onClick={onTogglePlay} disabled={count === 0} title={playing ? "pause" : "play"}>
          {playing ? "❚❚" : "▶"}
        </button>
        <button onClick={() => onIndex(clamp(index + 1))} disabled={index >= last} title="next tick">
          ▶
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={last}
        value={index}
        onChange={(e) => onIndex(clamp(Number(e.target.value)))}
      />
      <span className="ticklabel">{tickLabel ?? `${index} / ${last}`}</span>
    </div>
  );
}
