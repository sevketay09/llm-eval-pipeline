import type { HTMLAttributes } from "react";
import clsx from "clsx";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  quiet?: boolean;
  roomy?: boolean;
}

export default function Card({ quiet, roomy, className, children, ...rest }: CardProps) {
  return (
    <div
      className={clsx(
        "panel-surface",
        quiet && "panel-quiet",
        roomy && "panel-roomy",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
