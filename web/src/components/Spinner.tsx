interface SpinnerProps {
  size?: number;
}

export default function Spinner({ size = 32 }: SpinnerProps) {
  return (
    <span
      className="loading-orb"
      style={{ width: size, height: size }}
      role="status"
      aria-label="Loading"
    />
  );
}
