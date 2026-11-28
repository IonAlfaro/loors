import { ModalBackdrop } from "./ModalBackdrop";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ClipboardCopy,
  Download,
  FolderOpen,
  Save,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { invoke } from "../lib/bridge";
import { useGraphStore } from "../store/useGraphStore";
import { useI18n } from "../i18n";
import { HARNESS_TEMPLATES } from "../templates/harnesses";
import { parseWorkflowJson } from "../types/graph";

interface SavedWorkflowEntry {
  name: string;
  path: string;
}

interface WorkflowIoModalProps {
  open: boolean;
  onClose: () => void;
  onSave: () => Promise<{ name: string; path: string } | false>;
  onNotice: (msg: string) => void;
}

export function WorkflowIoModal({ open, onClose, onNotice, onSave }: WorkflowIoModalProps) {
  const { locale, t } = useI18n();
  const L = (en: string, es: string) => (locale === "es" ? es : en);
  const workflowName = useGraphStore((s) => s.workflowName);
  const exportWorkflow = useGraphStore((s) => s.exportWorkflow);
  const importWorkflow = useGraphStore((s) => s.importWorkflow);
  const loadTemplate = useGraphStore((s) => s.loadTemplate);

  const [tab, setTab] = useState<"library" | "templates" | "json">("library");
  const [search, setSearch] = useState("");
  const [savedList, setSavedList] = useState<SavedWorkflowEntry[]>([]);
  const [pasteJson, setPasteJson] = useState("");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refreshLibrary = () => {
    invoke<SavedWorkflowEntry[]>("list_saved_workflows")
      .then(setSavedList)
      .catch(() => setSavedList([]));
  };

  useEffect(() => {
    if (!open) return;
    setStatusMsg(null);
    setErrorMsg(null);
    setSearch("");
    refreshLibrary();
  }, [open]);

  const filteredSaved = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return savedList;
    return savedList.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.path.toLowerCase().includes(q)
    );
  }, [savedList, search]);

  const filteredTemplates = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return HARNESS_TEMPLATES;
    return HARNESS_TEMPLATES.filter((tpl) => {
      const { title, input, output } = tpl.copy[locale];
      return [title, input, output].some((value) => value.toLowerCase().includes(q));
    });
  }, [locale, search]);

  if (!open) return null;

  const handleSaveLibrary = async () => {
    setErrorMsg(null);
    setStatusMsg(null);
    try {
      const saved = await onSave();
      if (!saved) {
        setErrorMsg(L(
          "The harness changed during saving. Save again to keep the latest changes.",
          "El arnés cambió mientras se guardaba. Vuelve a guardar los últimos cambios."
        ));
        return;
      }
      setStatusMsg(L(`Saved to library: ${saved.path}`, `Guardado en biblioteca: ${saved.path}`));
      onNotice(L(`Saved: ${saved.name}`, `Guardado: ${saved.name}`));
      refreshLibrary();
    } catch (error) {
      setErrorMsg(error instanceof Error ? error.message : String(error));
    }
  };

  const handleLoadFromLibrary = async (entry: SavedWorkflowEntry) => {
    setErrorMsg(null);
    try {
      const raw = await invoke<string>("load_workflow_from_library", {
        path: entry.path,
      });
      const wf = parseWorkflowJson(raw);
      importWorkflow(wf, entry.name);
      onNotice(L(`Loaded harness: ${wf.name}`, `Arnés cargado: ${wf.name}`));
      onClose();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteFromLibrary = async (entry: SavedWorkflowEntry) => {
    try {
      await invoke("delete_workflow_from_library", { path: entry.path });
      refreshLibrary();
    } catch (err) {
      setErrorMsg(
        typeof err === "string"
          ? err
          : L("Could not delete harness.", "No se pudo eliminar.")
      );
    }
  };

  const handleDownloadFile = () => {
    const data = exportWorkflow();
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${data.name || "workflow"}.loors.json`;
    a.click();
    URL.revokeObjectURL(url);
    setStatusMsg(
      L(
        `Exported ${data.name || "workflow"}.loors.json`,
        `Exportado ${data.name || "workflow"}.loors.json`
      )
    );
  };

  const handleCopyJson = async () => {
    const content = JSON.stringify(exportWorkflow(), null, 2);
    try {
      await navigator.clipboard.writeText(content);
      setStatusMsg(L("JSON copied to clipboard.", "JSON copiado al portapapeles."));
    } catch {
      setErrorMsg(L("Could not access clipboard.", "No se pudo acceder al portapapeles."));
    }
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    try {
      const wf = parseWorkflowJson(await file.text());
      importWorkflow(wf);
      onNotice(L(`Imported: ${wf.name}`, `Importado: ${wf.name}`));
      onClose();
    } catch (err) {
      setErrorMsg(
        err instanceof Error
          ? err.message
          : L("Invalid JSON file.", "Archivo JSON inválido.")
      );
    }
    e.target.value = "";
  };

  const handleImportPaste = () => {
    setErrorMsg(null);
    try {
      const wf = parseWorkflowJson(pasteJson);
      importWorkflow(wf);
      setPasteJson("");
      onNotice(L(`Imported from JSON: ${wf.name}`, `Importado desde JSON: ${wf.name}`));
      onClose();
    } catch (err) {
      setErrorMsg(
        err instanceof Error ? err.message : L("Invalid JSON.", "JSON inválido.")
      );
    }
  };

  return (
    <ModalBackdrop label={t("library.title")} onClose={onClose}>
      <div
        className="modal-card modal-wide workflow-library-dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title">
            <FolderOpen size={18} />
            <span>{t("library.title")}</span>
          </div>
          <button
            type="button"
            className="icon-btn"
            aria-label={t("models.close")}
            onClick={onClose}
          >
            <X size={16} />
          </button>
        </div>

        <div className="io-modal-toolbar">
          <div className="io-actions" style={{ flexWrap: "wrap", gap: 8 }}>
            <button
              type="button"
              className="btn-primary"
              title={workflowName}
              onClick={handleSaveLibrary}
            >
              <Save size={14} />
              <span>{t("library.save")}</span>
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleDownloadFile}
            >
              <Download size={14} />
              <span>{t("library.export")}</span>
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleCopyJson}
            >
              <ClipboardCopy size={14} />
              <span>{t("library.copyJson")}</span>
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => fileRef.current?.click()}
            >
              <Upload size={14} />
              <span>{t("library.import")}</span>
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".json,.loors.json"
              hidden
              onChange={handleImportFile}
            />
          </div>

          <div className="io-tabs-bar">
            <div className="io-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={tab === "library"}
                className={`io-tab-btn ${tab === "library" ? "active" : ""}`}
                onClick={() => setTab("library")}
              >
                <span>{L("My Harnesses", "Mis arneses")}</span>
                <span className="io-tab-count">{savedList.length}</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "templates"}
                className={`io-tab-btn ${tab === "templates" ? "active" : ""}`}
                onClick={() => setTab("templates")}
              >
                <span>{L("Templates", "Plantillas")}</span>
                <span className="io-tab-count">{HARNESS_TEMPLATES.length}</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "json"}
                className={`io-tab-btn ${tab === "json" ? "active" : ""}`}
                onClick={() => setTab("json")}
              >
                <span>{L("Paste JSON", "Pegar JSON")}</span>
              </button>
            </div>

            {tab !== "json" && (
              <div className="io-search-box">
                <Search size={13} />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder={
                    tab === "library"
                      ? L("Filter saved harnesses...", "Filtrar arneses...")
                      : L("Filter templates...", "Filtrar plantillas...")
                  }
                />
              </div>
            )}
          </div>
        </div>

        <div className="modal-body" style={{ paddingTop: 10 }}>
          {statusMsg && <div className="hw-note">{statusMsg}</div>}
          {errorMsg && <div className="hw-error">{errorMsg}</div>}

          {tab === "library" && (
            <div className="installed-list">
              {filteredSaved.length === 0 ? (
                <div className="hw-note">
                  {savedList.length === 0
                    ? L(
                        "No saved harnesses yet. Click Save to library, or drag & drop any .loors.json directly onto the canvas.",
                        "Aún no has guardado arneses en tu biblioteca local. También puedes arrastrar y soltar un archivo .loors.json directamente sobre el canvas."
                      )
                    : L(
                        "No saved harnesses match your search.",
                        "Ningún arnés guardado coincide con tu búsqueda."
                      )}
                </div>
              ) : (
                filteredSaved.map((entry) => (
                  <div key={entry.path} className="installed-item">
                    <div className="installed-meta">
                      <div className="installed-name">{entry.name}</div>
                      <div className="installed-path">{entry.path}</div>
                    </div>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => handleLoadFromLibrary(entry)}
                    >
                      {L("Load", "Cargar")}
                    </button>
                    <button
                      type="button"
                      className="icon-btn danger"
                      title={L("Delete harness", "Eliminar arnés")}
                      onClick={() => handleDeleteFromLibrary(entry)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))
              )}
            </div>
          )}

          {tab === "templates" && (
            <div className="installed-list">
              {filteredTemplates.length === 0 ? (
                <div className="hw-note">
                  {L(
                    "No templates match your search.",
                    "Ninguna plantilla coincide con tu búsqueda."
                  )}
                </div>
              ) : (
                filteredTemplates.map((tpl) => {
                  const { title, input, output } = tpl.copy[locale];
                  const openTemplate = (useDemo = false) => {
                    loadTemplate(tpl.id, useDemo);
                    onNotice(L(`Template opened in a new tab · ${title}`, `Plantilla abierta en una pestaña nueva · ${title}`));
                    onClose();
                  };
                  return (
                    <div key={tpl.id} className="installed-item io-template">
                      <div className="installed-meta">
                        <div className="installed-name">{title}</div>
                        <dl className="io-template-details">
                          <dt>{L("Setup", "Configuración")}</dt>
                          <dd>{input}</dd>
                          <dt>{L("Output", "Resultado")}</dt>
                          <dd>{output}</dd>
                        </dl>
                      </div>
                      <div className="io-template-actions">
                        {tpl.demo && (
                          <button type="button" className="btn-secondary" onClick={() => openTemplate(true)}>
                            {L("Try sample", "Probar ejemplo")}
                          </button>
                        )}
                        <button type="button" className="btn-secondary" onClick={() => openTemplate()}>
                          {tpl.demo ? L("Use my PDFs", "Usar mis PDF") : L("Use template", "Usar plantilla")}
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {tab === "json" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <label className="field">
                <span>
                  {L("Import by pasting JSON directly", "Importar pegando JSON directamente")}
                </span>
                <textarea
                  rows={10}
                  placeholder={L(
                    'Paste .loors.json content here {"schemaVersion": 1, "name": "...", "nodes": [...], "edges": [...]}',
                    'Pega aquí el contenido de un .loors.json {"schemaVersion": 1, "name": "...", "nodes": [...], "edges": [...]}'
                  )}
                  value={pasteJson}
                  onChange={(e) => setPasteJson(e.target.value)}
                />
              </label>
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button
                  type="button"
                  className="btn-primary"
                  disabled={!pasteJson.trim()}
                  onClick={handleImportPaste}
                >
                  <Upload size={14} />
                  <span>{L("Import pasted JSON", "Importar JSON pegado")}</span>
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer io-modal-footer">
          <button type="button" className="btn-secondary" onClick={onClose}>
            {L("Close", "Cerrar")}
          </button>
        </div>
      </div>
    </ModalBackdrop>
  );
}
