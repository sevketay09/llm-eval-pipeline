import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { modelsApi } from "@/api/client";
import { Field, Select } from "./Field";

interface DecisionModelSelectProps {
  value: string;
  onChange: (modelKey: string) => void;
  label?: string;
}

export default function DecisionModelSelect({
  value,
  onChange,
  label = "Decision model (Jev)",
}: DecisionModelSelectProps) {
  const [models, setModels] = useState<string[] | null>(null);

  useEffect(() => {
    modelsApi
      .capabilities()
      .then((c) => setModels(c.decision_models))
      .catch(() => setModels([]));
  }, []);

  useEffect(() => {
    const first = models?.[0];
    if (first && !value) onChange(first);
  }, [models]); // eslint-disable-line react-hooks/exhaustive-deps

  if (models === null) {
    return (
      <Field label={label}>
        <Select disabled value="">
          <option value="">Loading…</option>
        </Select>
      </Field>
    );
  }

  if (models.length === 0) {
    return (
      <Field label={label}>
        <Select disabled value="">
          <option value="">No decision model configured</option>
        </Select>
        <p className="micro-copy" style={{ marginTop: "0.35rem" }}>
          Add a Jev key on the <Link to="/models">Models</Link> page to enable this.
        </p>
      </Field>
    );
  }

  return (
    <Field label={label}>
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </Select>
    </Field>
  );
}
