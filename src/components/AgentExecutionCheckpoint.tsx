type Props = {
  checkpoint: unknown;
  workspace: string;
  locale: string;
  disabled: boolean;
  onContinue: (command: string) => void;
};

function object(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : undefined;
}

export function AgentExecutionCheckpoint({ checkpoint, workspace, locale, disabled, onContinue }: Props) {
  const data = object(checkpoint);
  const plan = object(data?.plan);
  if (data?.version !== 1 || data.workspace !== workspace || !Array.isArray(plan?.tasks)) return null;
  const tasks = plan.tasks.map(object);
  if (!tasks.length || tasks.some(task => !task || typeof task.id !== "string" || typeof task.title !== "string")) return null;
  const done = tasks.filter(task => task?.status === "done").length;
  const approval = data.awaitingApproval === true;
  const es = locale === "es";
  const telemetry = object(data.telemetry);
  const counters = [
    [es ? "Recuperaciones" : "Recoveries", telemetry?.recoveries, telemetry?.recoveryLimit],
  ] as const;
  return (
    <section aria-label={es ? "Plan de ejecución" : "Execution plan"} style={{ padding: 12, border: "1px solid var(--border)", borderRadius: 8, marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <strong>{es ? "Plan de ejecución" : "Execution plan"} · {done}/{tasks.length}</strong>
        {done < tasks.length && <button type="button" className="composer-env-pill" disabled={disabled}
          onClick={() => onContinue(approval ? "/approve" : "/resume")}>
          {approval ? (es ? "Aprobar y ejecutar" : "Approve and run") : (es ? "Reanudar" : "Resume")}
        </button>}
      </div>
      <progress value={done} max={tasks.length} aria-label={es ? "Progreso" : "Progress"} style={{ width: "100%" }} />
      {telemetry && <div style={{ display: "flex", flexWrap: "wrap", gap: 12, fontSize: 12 }}>
        {counters.map(([label, used, limit]) => typeof used === "number" && typeof limit === "number"
          && Number.isFinite(used) && Number.isFinite(limit)
          ? <span key={label}>{label}: {used}/{limit}</span> : null)}
      </div>}
      <details open={approval}>
        <summary>{es ? "Tareas y criterios" : "Tasks and criteria"}</summary>
        <ol style={{ maxHeight: 220, overflowY: "auto", paddingLeft: 24 }}>
          {tasks.map(task => {
            if (!task) return null;
            const status = String(task.status);
            const dependencies = Array.isArray(task.depends_on) ? task.depends_on.filter(x => typeof x === "string").join(", ") : "";
            const acceptance = Array.isArray(task.acceptance) ? task.acceptance.filter(x => typeof x === "string").join("; ") : typeof task.acceptance === "string" ? task.acceptance : "";
            return <li key={String(task.id)} style={{ marginBottom: 6 }}>
              <strong>{String(task.title)}</strong> · {status}
              {dependencies && <div>{es ? "Depende de: " : "Depends on: "}{dependencies}</div>}
              {acceptance && <div>{es ? "Criterios: " : "Criteria: "}{acceptance}</div>}
            </li>;
          })}
        </ol>
      </details>
    </section>
  );
}
