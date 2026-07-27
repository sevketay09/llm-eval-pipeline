import type { CSSProperties } from "react";

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  className?: string;
  style?: CSSProperties;
}

export default function Skeleton({ width, height = "1rem", className, style }: SkeletonProps) {
  return (
    <span
      aria-hidden="true"
      className={`skeleton ${className ?? ""}`.trim()}
      style={{ display: "block", width: width ?? "100%", height, ...style }}
    />
  );
}
