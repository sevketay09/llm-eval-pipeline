import type { ButtonHTMLAttributes, ReactNode } from "react";
import clsx from "clsx";
import Spinner from "./Spinner";

type Variant = "primary" | "secondary" | "danger";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  loading?: boolean;
  icon?: ReactNode;
}

const VARIANT_CLASS: Record<Variant, string> = {
  primary: "button-primary",
  secondary: "button-secondary",
  danger: "button-danger",
};

export default function Button({
  variant = "primary",
  loading = false,
  icon,
  disabled,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={clsx(VARIANT_CLASS[variant], className)}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner size={16} /> : icon}
      {children}
    </button>
  );
}
