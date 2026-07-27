import type {
  InputHTMLAttributes,
  TextareaHTMLAttributes,
  SelectHTMLAttributes,
  ReactNode,
} from "react";
import clsx from "clsx";

export function Field({ label, children }: { label?: string; children: ReactNode }) {
  return (
    <label className="control-group">
      {label && <span className="ds-field-mini">{label}</span>}
      {children}
    </label>
  );
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={clsx("control-surface", className)} {...rest} />;
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={clsx("control-surface", className)} {...rest} />;
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={clsx("control-surface", className)} {...rest}>
      {children}
    </select>
  );
}
