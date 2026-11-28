import { AgentExecutionCheckpoint } from "./AgentExecutionCheckpoint";
import { AgentLimitSettings } from "./AgentLimitSettings";
import { AgentPermissionPicker } from "./AgentPermissionPicker";
import { AgentModelList } from "./AgentModelList";
import { useAgentModelGroups } from "./useAgentModelGroups";
import { AgentModelSearch } from "./AgentModelSearch";
import { AgentModelPickerTrigger } from "./AgentModelPickerTrigger";
import { AgentComposerActions } from "./AgentComposerActions";
import { AgentComposerAttachments } from "./AgentComposerAttachments";
import { AgentComposerInput } from "./AgentComposerInput";
import { AgentHistory } from "./AgentHistory";
import { AgentDiff } from "./AgentDiff";
import { PanelResizer } from "./PanelResizer";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Cloud,
  Copy,
  Download,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  GitBranch,
  Globe,
  ListChecks,
  MessageSquare,
  MoreHorizontal,
  Paperclip,
  Pencil,
  Pin,
  PinOff,
  Plus,
  RefreshCw,
  Search,
  Settings,
  Shield,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import { invoke } from "../lib/bridge";
import { listen } from "../lib/bridge";
import { useGraphStore } from "../store/useGraphStore";
import { useI18n } from "../i18n";
import { InteractiveBrowserPane } from "./InteractiveBrowserPane";
import {
  MarkdownContent,
  MessageQueueTray,
  ReasoningEffortSelect,
  WorkspaceFileTree,
  type QueuedMessageItem,
} from "./ChatActivity";
import {
  isCloudModelPath,
  parseGgufMetadata,
  type AgentArtifactPreview,
  type AgentChatEvent,
  type AgentChatMessage,
  type AgentSession,
  type AgentSessionSummary,
  type AgentTodoItem,
  type PendingAgentQuestion,
  type BrowserFrameEvent,
  type CloudProviderConfig,
  type HardwareProfile,
  type LocalModelFile,
  type ModelTuningProfile,
  type RuntimeBackendKind,
  type RuntimeDownloadEvent,
  type RuntimeStatusSnapshot,
  type SkillSummary,
} from "../types/graph";

function loadAgentAuxMap<T>(key: string): Map<string, T> {
  try {
    const raw = localStorage.getItem(key);
    return new Map<string, T>(raw ? JSON.parse(raw) : []);
  } catch { return new Map(); }
}

function saveAgentAuxMap<T>(key: string, map: Map<string, T>) {
  try { localStorage.setItem(key, JSON.stringify([...map])); } catch { /* Storage unavailable or full. */ }
}

function makeSessionId(): string {
  return `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

function normWs(p: string): string {
  const path = p.trim().replace(/\\/g, "/").replace(/^\/\/\?\/UNC\//i, "//").replace(/^\/\/\?\//, "");
  const windows = /^[a-z]:\//i.test(path) || path.startsWith("//");
  const root = path.startsWith("//") ? "//" : path.startsWith("/") ? "/" : "";
  const parts = path.split("/").filter((part) => part && part !== ".");
  const normalized = root + parts.join("/");
  return windows ? normalized.toLowerCase() : normalized;
}

function summarizeSession(session: AgentSession): AgentSessionSummary {
  const { id, title, workspace, updatedAt, permissionMode, pinned, messages } = session;
  return { id, title, workspace, updatedAt, permissionMode, pinned, messageCount: messages.length };
}

function displaySessionTitle(title: string, locale: string): string {
  const t = title.trim();
  if (t === "Nuevo chat" || t === "New chat") {
    return locale === "es" ? "Nuevo chat" : "New chat";
  }
  if (t === "Nueva sesión" || t === "New session" || !t) {
    return locale === "es" ? "Nueva sesión" : "New session";
  }
  return title;
}

function formatRelativeAge(epochSecs: number, locale: string): string {
  if (!epochSecs) return "";
  const diff = Math.max(0, Math.floor(Date.now() / 1000) - epochSecs);
  if (diff < 60) return locale === "es" ? "ahora" : "now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

function loadStoredProjects(storageKey = "loors_agent_projects"): string[] {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const unique = new Map<string, string>();
    for (const path of parsed) {
      if (typeof path !== "string" || !normWs(path)) continue;
      const key = normWs(path);
      if (!unique.has(key)) unique.set(key, path);
    }
    return [...unique.values()];
  } catch {
    return [];
  }
}

function saveStoredProjects(list: string[], storageKey = "loors_agent_projects") {
  try {
    localStorage.setItem(storageKey, JSON.stringify(list));
  } catch {
    // ignore
  }
}



export function AgentWorkspaceView() {
  const { locale, t } = useI18n();
  const nodes = useGraphStore((s) => s.nodes);
  const selectedBackend = useGraphStore((s) => s.selectedBackend);
  const setSelectedBackend = useGraphStore((s) => s.setSelectedBackend);
  const openTuningModal = useGraphStore((s) => s.openTuningModal);
  const activeModelPath = useGraphStore((s) => s.activeModelPath);
  const setActiveModelPath = useGraphStore((s) => s.setActiveModelPath);
  const modelLoadProgress = useGraphStore((s) => s.modelLoadProgress);
  const setAgentStatus = useGraphStore((s) => s.setAgentStatus);

  const defaultHarnessWorkspace = String(
    nodes.find((n) => n.data.kind === "task_input")?.data.config.workspace ?? "."
  );
  const rawHarnessModel = String(
    nodes.find((n) => n.data.kind === "local_model")?.data.config.modelPath ?? ""
  ).trim();
  const defaultHarnessModel =
    rawHarnessModel === "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
      ? ""
      : rawHarnessModel;

  const sidebarOpen = useGraphStore((s) => s.leftPanelOpen);
  const rightPanelOpen = useGraphStore((s) => s.appMode === "agent" && s.rightPanelOpen);
  const setRightPanelOpen = useGraphStore((s) => s.setRightPanelOpen);
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("loors_agent_left_w"));
    return Number.isFinite(saved) && saved >= 190 && saved <= 420 ? saved : 252;
  });
  const [rightPanelWidth, setRightPanelWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("loors_agent_right_w"));
    return Number.isFinite(saved) && saved >= 260 && saved <= 680 ? saved : 290;
  });
  const [diffPanelWidth, setDiffPanelWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("loors_agent_diff_w"));
    return Number.isFinite(saved) && saved >= 320 && saved <= 680 ? saved : 420;
  });
  const [browserPanelWidth, setBrowserPanelWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("loors_agent_browser_w"));
    return Number.isFinite(saved) && saved >= 420 && saved <= 980 ? saved : 580;
  });

  const [projects, setProjects] = useState<string[]>(() => loadStoredProjects());
  const [collapsedProjects, setCollapsedProjects] = useState<Record<string, boolean>>({});
  const [sessions, setSessions] = useState<AgentSessionSummary[]>([]);
  const [localModels, setLocalModels] = useState<LocalModelFile[]>([]);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelPickerQuery, setModelPickerQuery] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<string[]>([]);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [compactPopoverOpen, setCompactPopoverOpen] = useState(false);
  const [compactDraft, setCompactDraft] = useState<{ key: string; tokens: number } | null>(null);
  const [wsMenuOpen, setWsMenuOpen] = useState(false);
  const [permMenuOpen, setPermMenuOpen] = useState(false);
  const [todoItems, setTodoItems] = useState<AgentTodoItem[]>([]);
  const [initialTodos] = useState(() => loadAgentAuxMap<AgentTodoItem[]>("loors_agent_todos_v1"));
  const todosBySession = useRef(initialTodos);
  const [initialArtifacts] = useState(() => loadAgentAuxMap<AgentArtifactPreview>("loors_agent_artifacts_v1"));
  const artifactsBySession = useRef(initialArtifacts);
  const [todoCollapsed, setTodoCollapsed] = useState(false);
  const [questions, setQuestions] = useState<Record<string, PendingAgentQuestion>>({});
  const setPendingQuestion = (question: PendingAgentQuestion | null, id = activeSessionIdRef.current) => {
    setQuestions((prev) => {
      if (question) return { ...prev, [id]: question };
      if (!prev[id]) return prev;
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };
  const [selectedQuestionOpts, setSelectedQuestionOpts] = useState<string[]>([]);
  const [customQuestionAnswer, setCustomQuestionAnswer] = useState("");
  const [activeArtifact, setActiveArtifact] = useState<AgentArtifactPreview | null>(null);
  const [artifactExpanded, setArtifactExpanded] = useState(false);
  const [confirmRemoveProj, setConfirmRemoveProj] = useState<string | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [stagingFiles, setStagingFiles] = useState(false);
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [session, setSession] = useState<AgentSession>(() => ({
    id: makeSessionId(),
    title: locale === "es" ? "Nueva sesión" : "New session",
    workspace:
      defaultHarnessWorkspace && defaultHarnessWorkspace !== "."
        ? defaultHarnessWorkspace
        : loadStoredProjects()[0] ?? ".",
    modelPath: activeModelPath || defaultHarnessModel,
    permissionMode: "ask",
    useRag: true,
    compactAtTokens: 0,
    reasoningEffort: "medium",
    enabledSkills: [],
    pinnedFiles: [],
    allowedSessionTools: [],
    messages: [],
    updatedAt: Math.floor(Date.now() / 1000),
  }));

  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  const [input, setInput] = useState("");
  const [workMode, setWorkMode] = useState<"chat" | "plan" | "agent">("agent");
  const [workModeMenuOpen, setWorkModeMenuOpen] = useState(false);
  const workModeMenuRef = useRef<HTMLDivElement>(null);
  const workModeButtonRef = useRef<HTMLButtonElement>(null);
  const workModeLabel = locale === "es" ? "Modo de trabajo" : "Work mode";
  const workModes = [
    { id: "chat", label: "Chat", icon: MessageSquare, description: locale === "es" ? "Conversar sin herramientas" : "Chat without tools" },
    { id: "plan", label: "Plan", icon: ListChecks, description: locale === "es" ? "Consultar información y preparar un plan" : "Inspect and prepare a plan" },
    { id: "agent", label: locale === "es" ? "Agente" : "Agent", icon: Boxes, description: locale === "es" ? "Ejecutar tareas según los permisos" : "Carry out tasks within the current permissions" },
  ] as const;
  const activeWorkMode = workModes.find((mode) => mode.id === workMode) ?? workModes[2];
  const WorkModeIcon = activeWorkMode.icon;
  const composerPrompt = () => {
    const body = input.trim();
    if (/^\/(chat|plan|agent|resume|approve)(\s|$)/.test(body)) return body;
    if (!body && pendingAttachments.length === 0) return "";
    return workMode === "agent" ? body : `/${workMode} ${body}`;
  };
  const runningTurns = useRef(new Map<string, AgentSession>());
  const discardBufferedTokens = useRef<(id: string) => void>(() => {});
  const [runningSessionIds, setRunningSessionIds] = useState<string[]>([]);
  const running = runningSessionIds.length > 0;
  const isCurrentRunning = runningSessionIds.includes(session.id);
  const [queuedMessages, setMessageQueue] = useState<(QueuedMessageItem & { sessionId: string })[]>([]);
  const messageQueue = queuedMessages.filter((message) => message.sessionId === session.id);
  const [pinnedModels, setPinnedModels] = useState<string[]>(() => {
    try {
      const raw = localStorage.getItem("loors_pinned_models");
      return raw ? (JSON.parse(raw) as string[]) : [];
    } catch {
      return [];
    }
  });
  const [hiddenModels, setHiddenModels] = useState<string[]>(() => {
    try {
      const raw = localStorage.getItem("loors_hidden_models");
      return raw ? (JSON.parse(raw) as string[]) : [];
    } catch {
      return [];
    }
  });
  const [showHiddenModels, setShowHiddenModels] = useState(false);
  const [collapsedModelGroups, setCollapsedModelGroups] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [gpuBusyPrompt, setGpuBusyPrompt] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const [pinQuery, setPinQuery] = useState("");
  const [pinResults, setPinResults] = useState<string[]>([]);
  const [rightPanelTab, setRightPanelTab] = useState<"files" | "diff" | "browser">("files");
  const browserPanel = rightPanelTab === "browser";
  const diffPanel = rightPanelTab === "diff";
  const activePanelWidth = browserPanel ? browserPanelWidth : diffPanel ? diffPanelWidth : rightPanelWidth;
  const [workspaceFiles, setWorkspaceFiles] = useState<string[]>([]);
  const [fileFilter, setFileFilter] = useState("");
  const [selectedPreviewFile, setSelectedPreviewFile] = useState<string | null>(null);
  const [selectedPreviewContent, setSelectedPreviewContent] = useState<string>("");
  const [gitDiffText, setGitDiffText] = useState("");
  const diffRequestId = useRef(0);
  const previewRequestId = useRef(0);
  const activeTurnIds = useRef(new Map<string, string>());
  const pinRequestId = useRef(0);
  const refreshWorkspaceDiff = async (id: string, workspace: string) => {
    const request = ++diffRequestId.current;
    const current = () => request === diffRequestId.current && activeSessionIdRef.current === id && sessionRef.current.workspace === workspace;
    try {
      const diff = await invoke<string>("get_workspace_diff", { workspace: workspace || "." });
      if (current()) setGitDiffText(diff);
    } catch (err) {
      if (current()) setGitDiffText(`Error: ${String(err)}`);
    }
  };
  const [turnOutcomes, setTurnOutcomes] = useState<Record<string, Extract<AgentChatEvent, { type: "turn_finished" }>>>({});
  const [harnessProgress, setHarnessProgress] = useState<Record<string, Extract<AgentChatEvent, { type: "harness_progress" }>>>({});
  const clearHarnessProgress = (id: string) => setHarnessProgress(prev => {
    if (!prev[id]) return prev;
    const next = { ...prev };
    delete next[id];
    return next;
  });
  const [contextUsage, setContextUsage] = useState<
    Record<string, Extract<AgentChatEvent, { type: "context_usage" }>>
  >({});
  const [contextProfile, setContextProfile] = useState<{
    modelPath: string;
    profile: ModelTuningProfile;
  } | null>(null);
  const [liveBrowserFrame, setLiveBrowserFrame] = useState<BrowserFrameEvent | null>(null);
  const [browserExpanded, setBrowserExpanded] = useState(false);
  const [chatMenuId, setChatMenuId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renamingDraft, setRenamingDraft] = useState("");
  const [savedShots, setSavedShots] = useState<Record<string, string>>({});
  type PermissionRequest = {
    callId: string;
    toolName: string;
    summary: string;
    preview: string;
  };
  const [permissions, setPermissions] = useState<Record<string, PermissionRequest>>({});
  const pendingQuestion = questions[session.id];
  const permissionReq = permissions[session.id];
  const setPermissionReq = (request: PermissionRequest | null, id = activeSessionIdRef.current) => {
    setPermissions((prev) => {
      if (request) return { ...prev, [id]: request };
      if (!prev[id]) return prev;
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  const feedRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const modelPickerRef = useRef<HTMLDivElement>(null);
  const attachMenuRef = useRef<HTMLDivElement>(null);
  const compactPopoverRef = useRef<HTMLDivElement>(null);
  const wsMenuRef = useRef<HTMLDivElement>(null);
  const permMenuRef = useRef<HTMLDivElement>(null);
  const activeSessionIdRef = useRef(session.id);
  activeSessionIdRef.current = session.id;
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const pausedQueues = useRef(new Set<string>());
  const completedTurns = useRef(new Map<string, AgentSession>());
  const turnErrors = useRef(new Map<string, string>());
  const drafts = useRef(new Map<string, { text: string; attachments: string[] }>());
  if (input || pendingAttachments.length) drafts.current.set(session.id, { text: input, attachments: pendingAttachments });
  else drafts.current.delete(session.id);
  const sessionSelection = useRef(0);
  const sessionRefresh = useRef(0);
  const [hiddenProjects] = useState(() => new Set(loadStoredProjects("loors_hidden_agent_projects")));

  const showSession = (next: AgentSession) => {
    const previousId = activeSessionIdRef.current;
    if (next.id !== previousId || next.workspace !== sessionRef.current.workspace) {
      ++diffRequestId.current;
      ++previewRequestId.current;
      ++pinRequestId.current;
      setPinResults([]);
      setGitDiffText("");
      setSelectedPreviewFile(null);
      setSelectedPreviewContent("");
    }
    if (next.id !== previousId) {
      stickToBottomRef.current = true;
      setShowJumpToLatest(false);
      const nextDraft = drafts.current.get(next.id);
      setInput(nextDraft?.text ?? "");
      setPendingAttachments(nextDraft?.attachments ?? []);
      setTodoItems(todosBySession.current.get(next.id) ?? []);
      setActiveArtifact(artifactsBySession.current.get(next.id) ?? null);
      setArtifactExpanded(false);
      setGpuBusyPrompt(null);
    }
    activeSessionIdRef.current = next.id;
    sessionRef.current = next;
    setSession(next);
  };

  const registerProject = useCallback((wsPath: string, reopen = false) => {
    const trimmed = wsPath.trim();
    if (!trimmed || trimmed === ".") return;
    const key = normWs(trimmed);
    if (hiddenProjects.has(key)) {
      if (!reopen) return;
      hiddenProjects.delete(key);
      saveStoredProjects([...hiddenProjects], "loors_hidden_agent_projects");
    }
    setProjects((prev) => {
      if (prev.some((p) => normWs(p) === key)) return prev;
      const next = [...prev, trimmed];
      saveStoredProjects(next);
      return next;
    });
  }, [hiddenProjects]);

  const refreshSessions = useCallback(() => {
    const request = ++sessionRefresh.current;
    invoke<AgentSessionSummary[]>("list_agent_sessions")
      .then((list) => {
        if (request !== sessionRefresh.current) return;
        setSessions(list);
        for (const s of list) {
          if (s.workspace && s.workspace.trim() !== "" && s.workspace.trim() !== ".") {
            registerProject(s.workspace);
          }
        }
      })
      .catch(() => {});
  }, [registerProject]);

  const [cloudProviders, setCloudProviders] = useState<CloudProviderConfig[]>([]);
  const [autoInstallingRt, setAutoInstallingRt] = useState(false);
  const [rtInstallProgress, setRtInstallProgress] = useState<RuntimeDownloadEvent | null>(null);

  const refreshCloudProviders = useCallback(() => {
    invoke<CloudProviderConfig[]>("get_cloud_providers")
      .then((list) => setCloudProviders(list))
      .catch(() => setCloudProviders([]));
  }, []);

  const refreshLocalModels = useCallback(() => {
    invoke<LocalModelFile[]>("scan_local_models")
      .then((list) => {
        const mainModels = list.filter(
          (m) => !m.name.toLowerCase().includes("mmproj")
        );
        setLocalModels(mainModels);
        setSession((prev) => {
          if (activeModelPath && isCloudModelPath(activeModelPath)) {
            return { ...prev, modelPath: activeModelPath };
          }
          if (prev.modelPath && isCloudModelPath(prev.modelPath)) {
            return prev;
          }
          if (
            activeModelPath &&
            mainModels.some((m) => m.path === activeModelPath)
          ) {
            return { ...prev, modelPath: activeModelPath };
          }
          if (prev.modelPath && mainModels.some((m) => m.path === prev.modelPath)) {
            return prev;
          }
          const preferred =
            mainModels.find((m) => m.path === defaultHarnessModel)?.path ??
            mainModels[0]?.path ??
            prev.modelPath;
          return { ...prev, modelPath: preferred };
        });
      })
      .catch(() => {});
  }, [activeModelPath, defaultHarnessModel]);

  useEffect(() => {
    refreshSessions();
    refreshLocalModels();
    refreshCloudProviders();
    const onModelsChanged = () => refreshLocalModels();
    const onProvidersChanged = () => refreshCloudProviders();
    window.addEventListener("focus", refreshSessions);
    window.addEventListener("loors:models-changed", onModelsChanged);
    window.addEventListener("loors:providers-changed", onProvidersChanged);
    return () => {
      window.removeEventListener("focus", refreshSessions);
      window.removeEventListener("loors:models-changed", onModelsChanged);
      window.removeEventListener("loors:providers-changed", onProvidersChanged);
    };
  }, [refreshSessions, refreshLocalModels, refreshCloudProviders]);

  const handleAutoInstallRuntime = useCallback(async () => {
    if (autoInstallingRt) return;
    setAutoInstallingRt(true);
    setRtInstallProgress(null);
    let unlisten: (() => void) | undefined;
    try {
      unlisten = await listen<RuntimeDownloadEvent>("loors://runtime-download", (ev) => {
        setRtInstallProgress(ev.payload);
      });
      const hw = await invoke<HardwareProfile>("detect_hardware");
      const targetBackend = hw.recommended_backend || "cpu";
      await invoke("install_runtime", { backend: targetBackend });
      const snap = await invoke<RuntimeStatusSnapshot>("get_runtime_status");
      setInstalledBackends((snap.installed ?? []).map((r) => r.backend));
      setSelectedBackend(targetBackend);
    } catch (err) {
      setError(
        typeof err === "string"
          ? err
          : "No se pudo instalar llama.cpp automáticamente."
      );
    } finally {
      if (unlisten) unlisten();
      setAutoInstallingRt(false);
    }
  }, [autoInstallingRt, setSelectedBackend]);

  const [installedBackends, setInstalledBackends] = useState<RuntimeBackendKind[]>([]);
  const [setupChecked, setSetupChecked] = useState(false);
  const [onboardingDismissed, setOnboardingDismissed] = useState<boolean>(() => {
    try {
      return localStorage.getItem("loors_onboarding_dismissed") === "1";
    } catch {
      return false;
    }
  });
  const dismissOnboarding = () => {
    setOnboardingDismissed(true);
    try {
      localStorage.setItem("loors_onboarding_dismissed", "1");
    } catch {
      // ignore
    }
  };
  useEffect(() => {
    invoke<RuntimeStatusSnapshot>("get_runtime_status")
      .then((snap) => {
        const inst = snap.installed ?? [];
        setInstalledBackends(inst.map((r) => r.backend));
        setSetupChecked(true);
      })
      .catch(() => {
        setSetupChecked(true);
      });
  }, [selectedBackend, modelPickerOpen]);

  useEffect(() => {
    if (defaultHarnessWorkspace && defaultHarnessWorkspace !== ".") {
      registerProject(defaultHarnessWorkspace);
    }
  }, [defaultHarnessWorkspace, registerProject]);

  useEffect(() => {
    if (!activeModelPath) return;
    setSession((prev) =>
      prev.modelPath === activeModelPath
        ? prev
        : { ...prev, modelPath: activeModelPath }
    );
  }, [activeModelPath]);

  useEffect(() => {
    const modelPath = session.modelPath;
    if (!modelPath || modelPath.startsWith("cloud::")) return;
    let current = true;
    invoke<ModelTuningProfile>("get_model_profile", { modelPath, agentMode: true })
      .then((profile) => {
        if (current) setContextProfile({ modelPath, profile });
      })
      .catch(() => {});

    const onProfileUpdated = (ev: Event) => {
      const detail = (
        ev as CustomEvent<{ modelPath?: string; profile?: ModelTuningProfile }>
      ).detail;
      if (detail?.profile && detail.modelPath === modelPath) {
        current = false;
        setContextProfile({ modelPath, profile: detail.profile });
      }
    };
    window.addEventListener("loors:model-profile-updated", onProfileUpdated);
    return () => {
      current = false;
      window.removeEventListener("loors:model-profile-updated", onProfileUpdated);
    };
  }, [session.modelPath]);

  useEffect(() => {
    if (!modelPickerOpen) return;
    refreshLocalModels();
    const onMouseDown = (e: MouseEvent) => {
      if (
        modelPickerRef.current &&
        !modelPickerRef.current.contains(e.target as Node)
      ) {
        setModelPickerOpen(false);
      }
    };
    window.addEventListener("mousedown", onMouseDown);
    return () => window.removeEventListener("mousedown", onMouseDown);
  }, [modelPickerOpen, refreshLocalModels]);

  useEffect(() => {
    if (!attachMenuOpen) return;
    void handleSearchPin(pinQuery);
    const onMouseDown = (e: MouseEvent) => {
      if (
        attachMenuRef.current &&
        !attachMenuRef.current.contains(e.target as Node)
      ) {
        setAttachMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onMouseDown);
    return () => window.removeEventListener("mousedown", onMouseDown);
  }, [attachMenuOpen]);

  useEffect(() => {
    if (!compactPopoverOpen) return;
    const onMouseDown = (e: MouseEvent) => {
      if (
        compactPopoverRef.current &&
        !compactPopoverRef.current.contains(e.target as Node)
      ) {
        setCompactPopoverOpen(false);
      }
    };
    window.addEventListener("mousedown", onMouseDown);
    return () => window.removeEventListener("mousedown", onMouseDown);
  }, [compactPopoverOpen]);

  const activeLocalFile = useMemo(
    () => localModels.find((m) => m.path === session.modelPath) ?? null,
    [localModels, session.modelPath]
  );

  const activeModelMeta = useMemo(
    () =>
      session.modelPath
        ? parseGgufMetadata(
            session.modelPath,
            activeLocalFile ? activeLocalFile.size_mb * 1024 * 1024 : undefined
          )
        : null,
    [session.modelPath, activeLocalFile]
  );

  const groupedPickerModels = useAgentModelGroups(localModels, modelPickerQuery);

  useEffect(() => {
    const workspace = session.workspace;
    let cancelled = false;
    setSkills([]);
    setWorkspaceFiles([]);
    invoke<SkillSummary[]>("list_skills", { workspace: workspace || "." })
      .then(items => { if (!cancelled && sessionRef.current.workspace === workspace) setSkills(items.filter(item => item.name !== "architect-loors")); })
      .catch(() => {});
    setSelectedPreviewFile(null);
    setSelectedPreviewContent("");
    return () => { cancelled = true; };
  }, [session.workspace]);

  useEffect(() => {
    const workspace = session.workspace;
    let cancelled = false;
    if (!workspace || workspace === ".") { setWorkspaceFiles([]); return; }
    invoke<string[]>("find_workspace_files", { workspace, query: fileFilter })
      .then(files => { if (!cancelled && sessionRef.current.workspace === workspace) setWorkspaceFiles(files); })
      .catch(() => { if (!cancelled && sessionRef.current.workspace === workspace) setWorkspaceFiles([]); });
    return () => { cancelled = true; };
  }, [session.workspace, fileFilter, rightPanelTab, rightPanelOpen]);

  // Listen to live Agent Chat events & Browser CDP frames
  useEffect(() => {
    let cancelled = false;
    let unlistenAgent: (() => void) | undefined;
    let unlistenBrowser: (() => void) | undefined;
    let renderTimer: ReturnType<typeof setTimeout> | undefined;
    const pendingTokens = new Map<string, Map<string, { content: string; thinking: string }>>();
    discardBufferedTokens.current = (id) => { pendingTokens.delete(id); };
    const flushTokens = (publish = true) => {
      if (renderTimer !== undefined) clearTimeout(renderTimer);
      renderTimer = undefined;
      for (const [sessionId, chunks] of pendingTokens) {
        const current = runningTurns.current.get(sessionId);
        if (!current) continue;
        const messages = [...current.messages];
        const indices = new Map(messages.map((m, i) => [m.id, i]));
        for (const [id, chunk] of chunks) {
          const idx = indices.get(id);
          if (idx === undefined) {
            messages.push({ id, role: "assistant", content: chunk.content, thinking: chunk.thinking, toolCalls: [], timestamp: Math.floor(Date.now() / 1000) });
          } else {
            const m = messages[idx];
            messages[idx] = { ...m, content: m.content + chunk.content, thinking: (m.thinking ?? "") + chunk.thinking };
          }
        }
        const next = { ...current, messages };
        runningTurns.current.set(sessionId, next);
        if (publish && !cancelled && sessionId === activeSessionIdRef.current) showSession(next);
      }
      pendingTokens.clear();
    };

    listen<AgentChatEvent>("loors://agent-chat-event", (ev) => {
      if (cancelled) return;
      const p = ev.payload;
      if (!runningTurns.current.has(p.session_id) || p.turn_id !== activeTurnIds.current.get(p.session_id)) return;
      if (p.type !== "token" && pendingTokens.size) flushTokens();
      if (p.type === "harness_progress") {
        setHarnessProgress(prev => ({ ...prev, [p.session_id]: p }));
        return;
      }
      if (p.type === "context_usage") {
        setContextUsage((prev) => ({ ...prev, [p.session_id]: p }));
        return;
      }
      const visible = p.session_id === activeSessionIdRef.current;
      if (p.type === "permission_requested") {
        setPermissionReq({
          callId: p.call_id,
          toolName: p.tool_name,
          summary: p.summary,
          preview: p.preview,
        }, p.session_id);
        return;
      }
      if (p.type === "question_requested") {
        setPendingQuestion({
          callId: p.call_id,
          question: p.question,
          options: p.options,
          allowMultiple: p.allow_multiple,
        }, p.session_id);
        if (visible) {
          setSelectedQuestionOpts([]);
          setCustomQuestionAnswer("");
        }
        return;
      }
      if (p.type === "turn_finished") {
        setTurnOutcomes(prev => ({ ...prev, [p.session_id]: p }));
        clearHarnessProgress(p.session_id);
        setPermissionReq(null, p.session_id);
        setPendingQuestion(null, p.session_id);
      } else if (p.type === "tool_finished") {
        setPermissions(prev => {
          if (prev[p.session_id]?.callId !== p.call.id) return prev;
          const next = { ...prev };
          delete next[p.session_id];
          return next;
        });
        setQuestions(prev => {
          if (prev[p.session_id]?.callId !== p.call.id) return prev;
          const next = { ...prev };
          delete next[p.session_id];
          return next;
        });
      }

      const updateTurn = (update: (session: AgentSession) => AgentSession) => {
        const current = runningTurns.current.get(p.session_id);
        if (!current) return;
        const next = update(current);
        runningTurns.current.set(p.session_id, next);
        if (visible) showSession(next);
      };

      if (!visible && p.type !== "token" && p.type !== "tool_started" && p.type !== "tool_finished" && p.type !== "todo_updated" && p.type !== "artifact_updated") return;

      if (p.type === "todo_updated") {
        todosBySession.current.set(p.session_id, p.items); saveAgentAuxMap("loors_agent_todos_v1", todosBySession.current);
        if (visible) setTodoItems(p.items);
      } else if (p.type === "artifact_updated") {
        const artifact: AgentArtifactPreview = {
          title: p.title,
          path: p.path,
          kind: p.kind,
          content: p.content,
        };
        artifactsBySession.current.set(p.session_id, artifact); saveAgentAuxMap("loors_agent_artifacts_v1", artifactsBySession.current);
        if (visible) {
          setActiveArtifact(artifact);
          setArtifactExpanded(false);
        }
      } else if (p.type === "token") {
        if (!runningTurns.current.has(p.session_id)) return;
        let chunks = pendingTokens.get(p.session_id);
        if (!chunks) { chunks = new Map(); pendingTokens.set(p.session_id, chunks); }
        let chunk = chunks.get(p.message_id);
        if (!chunk) { chunk = { content: "", thinking: "" }; chunks.set(p.message_id, chunk); }
        if (p.is_thinking) chunk.thinking += p.token;
        else chunk.content += p.token;
        if (renderTimer === undefined) renderTimer = setTimeout(() => flushTokens(), 40);
      } else if (p.type === "tool_started") {
        updateTurn((prev) => {
          const msgs = [...prev.messages];
          let idx = msgs.findIndex((m) => m.id === p.message_id);
          if (idx === -1) {
            msgs.push({ id: p.message_id, role: "assistant", content: "", thinking: "", toolCalls: [], timestamp: Math.floor(Date.now() / 1000) });
            idx = msgs.length - 1;
          }
          const target = { ...msgs[idx] };
          target.toolCalls = target.toolCalls.some((c) => c.id === p.call.id) ? target.toolCalls : [...target.toolCalls, p.call];
          msgs[idx] = target;
          return { ...prev, messages: msgs };
        });
      } else if (p.type === "tool_finished") {
        updateTurn((prev) => {
          const msgs = [...prev.messages];
          let idx = msgs.findIndex((m) => m.id === p.message_id);
          if (idx === -1) {
            msgs.push({ id: p.message_id, role: "assistant", content: "", thinking: "", toolCalls: [], timestamp: Math.floor(Date.now() / 1000) });
            idx = msgs.length - 1;
          }
          const target = { ...msgs[idx] };
          const cIdx = target.toolCalls.findIndex((c) => c.id === p.call.id);
          if (cIdx >= 0) {
            const nextCalls = [...target.toolCalls];
            nextCalls[cIdx] = p.call;
            target.toolCalls = nextCalls;
          } else {
            target.toolCalls = [...target.toolCalls, p.call];
          }
          msgs[idx] = target;
          return { ...prev, messages: msgs };
        });
        if (!visible) return;
        if (p.call.screenshotPath) {
          const shotPath = p.call.screenshotPath;
          invoke<string>("read_screenshot_base64", { path: shotPath })
            .then((b64) =>
              setSavedShots((prev) => ({ ...prev, [shotPath]: b64 }))
            )
            .catch(() => {});
        }
        if (p.call.diffPreview || p.call.addedLines > 0 || p.call.removedLines > 0) {
          const workspace = runningTurns.current.get(p.session_id)?.workspace ?? sessionRef.current.workspace;
          const eventSessionId = p.session_id;
          if (!cancelled && activeSessionIdRef.current === eventSessionId) void refreshWorkspaceDiff(eventSessionId, workspace);
        }
      }
    })
      .then((fn) => {
        if (cancelled) fn();
        else unlistenAgent = fn;
      })
      .catch(() => {});

    listen<BrowserFrameEvent>("loors://browser-frame", (ev) => {
      if (!cancelled && ev.payload.session_id === activeSessionIdRef.current) {
        setLiveBrowserFrame(ev.payload);
      }
    })
      .then((fn) => {
        if (cancelled) fn();
        else unlistenBrowser = fn;
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      flushTokens(false);
      discardBufferedTokens.current = () => {};
      if (unlistenAgent) unlistenAgent();
      if (unlistenBrowser) unlistenBrowser();
    };
  }, []);

  useEffect(() => {
    const el = feedRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [session.messages, permissionReq, liveBrowserFrame]);

  const persistSession = useCallback(
    async (next: AgentSession) => {
      if (runningTurns.current.has(next.id)) {
        setError(locale === "es" ? "Espera a que termine la respuesta para guardar los cambios." : "Wait for the response to finish before saving changes.");
        return;
      }
      if (activeSessionIdRef.current === next.id) {
        sessionRef.current = next;
        setSession(next);
      }
      if (completedTurns.current.has(next.id)) completedTurns.current.set(next.id, next);
      try {
        await invoke("save_agent_session", { session: next });
        refreshSessions();
      } catch (err) {
        if (activeSessionIdRef.current === next.id) setError(String(err));
      }
    },
    [locale, refreshSessions]
  );

  const togglePinModel = useCallback((modelKey: string) => {
    setPinnedModels((prev) => {
      const next = prev.includes(modelKey)
        ? prev.filter((k) => k !== modelKey)
        : [modelKey, ...prev];
      try {
        localStorage.setItem("loors_pinned_models", JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, []);

  const toggleHideModel = useCallback((modelKey: string) => {
    setHiddenModels((prev) => {
      const hiding = !prev.includes(modelKey);
      const next = hiding
        ? [...prev, modelKey]
        : prev.filter((k) => k !== modelKey);
      try {
        localStorage.setItem("loors_hidden_models", JSON.stringify(next));
      } catch {
        // ignore
      }
      if (hiding) {
        setPinnedModels((pins) => {
          if (!pins.includes(modelKey)) return pins;
          const nextPins = pins.filter((k) => k !== modelKey);
          try {
            localStorage.setItem("loors_pinned_models", JSON.stringify(nextPins));
          } catch {
            // ignore
          }
          return nextPins;
        });
      }
      return next;
    });
  }, []);

  const handleNewSession = (targetWorkspace?: string) => {
    const ws = targetWorkspace ?? (session.workspace || defaultHarnessWorkspace);
    if (ws && ws !== ".") registerProject(ws);
    setError(null);
    setLiveBrowserFrame(null);
    ++sessionSelection.current;
    setSelectedQuestionOpts([]);
    setCustomQuestionAnswer("");
    showSession({
      id: makeSessionId(),
      title: ws
        ? locale === "es"
          ? "Nueva sesión"
          : "New session"
        : locale === "es"
          ? "Nuevo chat"
          : "New chat",
      workspace: ws,
      modelPath: session.modelPath || localModels[0]?.path || "",
      permissionMode: session.permissionMode,
      useRag: session.useRag,
      compactAtTokens: session.compactAtTokens ?? 0,
      agentLimits: session.agentLimits,
      reasoningEffort: session.reasoningEffort ?? "medium",
      enabledSkills: session.enabledSkills,
      pinnedFiles: [],
      allowedSessionTools: [],
      messages: [],
      updatedAt: Math.floor(Date.now() / 1000),
    });
  };

  const handleSelectSession = async (id: string) => {
    if (id === activeSessionIdRef.current) {
      ++sessionSelection.current;
      return;
    }
    const request = ++sessionSelection.current;
    try {
      const loaded = runningTurns.current.get(id)
        ?? await invoke<AgentSession>("load_agent_session", { id });
      if (request !== sessionSelection.current) return;
      showSession(runningTurns.current.get(id) ?? loaded);
      setError(turnErrors.current.get(id) ?? null);
      setSelectedQuestionOpts([]);
      setCustomQuestionAnswer("");
      setLiveBrowserFrame(null);
    } catch (err) {
      if (request === sessionSelection.current) setError(String(err));
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await invoke("delete_agent_session", { id });
      setSessions((prev) => prev.filter((summary) => summary.id !== id));
      refreshSessions();
      setMessageQueue((prev) => prev.filter((message) => message.sessionId !== id));
      completedTurns.current.delete(id);
      pausedQueues.current.delete(id);
      turnErrors.current.delete(id);
      drafts.current.delete(id);
      todosBySession.current.delete(id); saveAgentAuxMap("loors_agent_todos_v1", todosBySession.current);
      artifactsBySession.current.delete(id); saveAgentAuxMap("loors_agent_artifacts_v1", artifactsBySession.current);
      if (activeSessionIdRef.current === id) {
        handleNewSession(sessionRef.current.workspace);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  useEffect(() => {
    if (!chatMenuId) return;
    const onDocClick = () => setChatMenuId(null);
    window.addEventListener("click", onDocClick);
    return () => window.removeEventListener("click", onDocClick);
  }, [chatMenuId]);

  const handleTogglePinSession = async (id: string) => {
    setChatMenuId(null);
    try {
      const target: AgentSession =
        id === session.id
          ? session
          : await invoke<AgentSession>("load_agent_session", { id });
      const updated: AgentSession = { ...target, pinned: !target.pinned };
      await invoke("save_agent_session", { session: updated });
      if (id === activeSessionIdRef.current) {
        showSession(updated);
      }
      refreshSessions();
    } catch (err) {
      setError(String(err));
    }
  };

  const handleStartRenameSession = (s: AgentSessionSummary) => {
    setChatMenuId(null);
    setRenamingId(s.id);
    setRenamingDraft(displaySessionTitle(s.title, locale));
  };

  const handleCommitRenameSession = async (id: string) => {
    const trimmed = renamingDraft.trim();
    setRenamingId(null);
    if (!trimmed) return;
    try {
      const target: AgentSession =
        id === session.id
          ? session
          : await invoke<AgentSession>("load_agent_session", { id });
      const updated: AgentSession = { ...target, title: trimmed };
      await invoke("save_agent_session", { session: updated });
      if (id === activeSessionIdRef.current) showSession(updated);
      refreshSessions();
    } catch (err) {
      setError(String(err));
    }
  };

  const handleAddProject = async () => {
    try {
      const picked = await invoke<string | null>("pick_folder", {
        initialDir: session.workspace && session.workspace !== "." ? session.workspace : null,
      });
      if (picked) {
        registerProject(picked, true);
        handleNewSession(picked);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const handleRemoveProject = (e: React.MouseEvent, projPath: string) => {
    e.stopPropagation();
    const key = normWs(projPath);
    hiddenProjects.add(key);
    saveStoredProjects([...hiddenProjects], "loors_hidden_agent_projects");
    setConfirmRemoveProj(null);
    setProjects((prev) => {
      const next = prev.filter((p) => normWs(p) !== key);
      saveStoredProjects(next);
      return next;
    });
  };

  const handlePickWorkspace = async () => {
    try {
      const picked = await invoke<string | null>("pick_folder", {
        initialDir: session.workspace && session.workspace !== "." ? session.workspace : null,
      });
      if (picked) {
        registerProject(picked, true);
        await persistSession({ ...session, workspace: picked, pinnedFiles: [] });
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const handlePickAttachFiles = async () => {
    try {
      const picked = await invoke<string[]>("pick_files", {
        initialDir: session.workspace && session.workspace !== "." ? session.workspace : null,
      });
      if (picked && picked.length > 0) {
        setPendingAttachments((prev) => {
          const next = [...prev];
          for (const p of picked) {
            if (!next.includes(p)) next.push(p);
          }
          return next;
        });
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const stageExternalFiles = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;
    setStagingFiles(true);
    try {
      const added: string[] = [];
      for (const f of files) {
        const nativePath = (f as File & { path?: string }).path;
        if (nativePath && nativePath.trim()) {
          added.push(nativePath);
        } else {
          const buf = await f.arrayBuffer();
          const stagedPath = await invoke<string>("stage_uploaded_file", {
            name: f.name || "adjunto.png",
            bytes: Array.from(new Uint8Array(buf)),
          });
          added.push(stagedPath);
        }
      }
      if (added.length > 0) {
        setPendingAttachments((prev) => {
          const next = [...prev];
          for (const p of added) {
            if (!next.includes(p)) next.push(p);
          }
          return next;
        });
      }
    } catch (err) {
      setError(`Error adjuntando archivo: ${String(err)}`);
    } finally {
      setStagingFiles(false);
    }
  };

  const handleSearchPin = async (q: string) => {
    setPinQuery(q);
    const request = ++pinRequestId.current;
    const current = () => request === pinRequestId.current;
    try {
      const matches = await invoke<string[]>("find_workspace_files", {
        workspace: session.workspace || ".",
        query: q,
      });
      if (current()) setPinResults(matches.slice(0, 12));
    } catch {
      if (current()) setPinResults([]);
    }
  };

  const togglePinFile = (rel: string) => {
    const exists = session.pinnedFiles.includes(rel);
    const nextPinned = exists
      ? session.pinnedFiles.filter((p) => p !== rel)
      : [...session.pinnedFiles, rel];
    void persistSession({ ...session, pinnedFiles: nextPinned });
    setPinQuery("");
    setPinResults([]);
  };

  const sendTurn = async (
    promptText: string,
    forceStopHarness = false,
    explicitAttachments?: string[],
    queuedSession?: AgentSession,
  ) => {
    const trimmed = promptText.trim();
    const attachmentsToSend = explicitAttachments ?? pendingAttachments;
    if (!trimmed && attachmentsToSend.length === 0) return;

    const curSession = queuedSession ?? sessionRef.current;
    const id = curSession.id;
    if (!curSession.modelPath) return;
    if (runningTurns.current.has(id)) {
      setMessageQueue((prev) => [
        ...prev,
        {
          id: makeSessionId(),
          text: trimmed,
          attachments: attachmentsToSend.length > 0 ? [...attachmentsToSend] : undefined,
          sessionId: id,
        },
      ]);
      if (!explicitAttachments) {
        setInput("");
        setPendingAttachments([]);
      }
      return;
    }

    pausedQueues.current.delete(id);
    completedTurns.current.delete(id);
    turnErrors.current.delete(id);
    if (id === activeSessionIdRef.current) {
      setError(null);
      setGpuBusyPrompt(null);
    }

    const activeWs =
      curSession.workspace && curSession.workspace.trim() !== "."
        ? curSession.workspace.trim()
        : "";
    if (activeWs) {
      registerProject(activeWs);
    }

    const titleFallback =
      attachmentsToSend[0]?.split(/[\\/]/).pop() ?? "Archivo adjunto";
    const userMsg: AgentChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: trimmed,
      thinking: "",
      attachments: attachmentsToSend,
      toolCalls: [],
      timestamp: Math.floor(Date.now() / 1000),
    };
    const nextSession: AgentSession = {
      ...curSession,
      workspace: activeWs,
      title:
        curSession.messages.length === 0 &&
        (curSession.title === "Nueva sesión" ||
          curSession.title === "Nuevo chat" ||
          curSession.title === "New session" ||
          curSession.title === "New chat")
          ? (trimmed || titleFallback).slice(0, 46)
          : curSession.title,
      messages: [...curSession.messages, userMsg],
      updatedAt: Math.floor(Date.now() / 1000),
    };

    const turnId = crypto.randomUUID();
    clearHarnessProgress(id);
    setTurnOutcomes(prev => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    activeTurnIds.current.set(id, turnId);
    runningTurns.current.set(id, nextSession);
    setRunningSessionIds([...runningTurns.current.keys()]);
    setAgentStatus(true, nextSession.title);
    if (id === activeSessionIdRef.current) {
      showSession(nextSession);
      stickToBottomRef.current = true;
      if (!explicitAttachments) {
        setInput("");
        setPendingAttachments([]);
      }
    }

    try {
      if (forceStopHarness) await invoke("stop_workflow");
      const updated = await invoke<AgentSession>("run_agent_turn", {
        backend: selectedBackend,
        session: { ...curSession, workspace: activeWs, title: nextSession.title },
        userPrompt: trimmed,
        overrideCtx: null,
        attachments: attachmentsToSend,
        turnId,
      });
      discardBufferedTokens.current(id);
      completedTurns.current.set(id, updated);
      setSessions((prev) => [summarizeSession(updated), ...prev.filter((summary) => summary.id !== id)]);
      if (activeSessionIdRef.current === id) showSession(updated);
    } catch (err) {
      const msg = String(err);
      pausedQueues.current.add(id);
      const persisted = await invoke<AgentSession>("load_agent_session", { id }).catch(() => curSession);
      discardBufferedTokens.current(id);
      completedTurns.current.set(id, persisted);
      const needsRetry = persisted.messages.length === curSession.messages.length;
      if (needsRetry && !drafts.current.has(id)) {
        drafts.current.set(id, { text: trimmed, attachments: attachmentsToSend });
      }
      if (activeSessionIdRef.current === id) {
        showSession(persisted);
        const draft = drafts.current.get(id);
        if (needsRetry && draft) {
          setInput(draft.text);
          setPendingAttachments(draft.attachments);
        }
        if (msg.includes("__GPU_BUSY_HARNESS__")) {
          setGpuBusyPrompt(trimmed);
        } else {
          setError(msg);
        }
      }
      turnErrors.current.set(id, msg);
    } finally {
      discardBufferedTokens.current(id);
      activeTurnIds.current.delete(id);
      runningTurns.current.delete(id);
      clearHarnessProgress(id);
      setRunningSessionIds([...runningTurns.current.keys()]);
      const otherTurn = runningTurns.current.values().next().value;
      setAgentStatus(Boolean(otherTurn), otherTurn?.title ?? null);
      setPermissionReq(null, id);
      setPendingQuestion(null, id);
      refreshSessions();
    }
  };

  useEffect(() => {
    const waiting = new Set(queuedMessages.map((message) => message.sessionId));
    for (const [id, completed] of completedTurns.current) {
      if (!waiting.has(id)) {
        completedTurns.current.delete(id);
      } else if (!pausedQueues.current.has(id) && !runningTurns.current.has(id)) {
        const next = queuedMessages.find((message) => message.sessionId === id);
        if (!next) continue;
        const target = activeSessionIdRef.current === id ? sessionRef.current : completed;
        if (target.id !== id || !target.modelPath) continue;
        setMessageQueue((prev) => prev.filter((message) => message.id !== next.id));
        void sendTurn(next.text, false, next.attachments ?? [], target);
      }
    }
  }, [runningSessionIds, queuedMessages]);

  const handleStop = async () => {
    const id = activeSessionIdRef.current;
    pausedQueues.current.add(id);
    try {
      await invoke("stop_agent_turn", { sessionId: id });
    } catch (err) {
      if (id === activeSessionIdRef.current) setError(String(err));
    }
  };

  const handlePermission = async (approved: boolean, alwaysAllowTool = false) => {
    const id = session.id;
    try {
      await invoke("resolve_agent_permission", {
        sessionId: id,
        decision: { approved, alwaysAllowTool },
      });
      if (approved && alwaysAllowTool && permissionReq && id === activeSessionIdRef.current) {
        setSession((prev) => ({
          ...prev,
          allowedSessionTools: prev.allowedSessionTools.includes(permissionReq.toolName)
            ? prev.allowedSessionTools
            : [...prev.allowedSessionTools, permissionReq.toolName],
        }));
      }
      setPermissions((prev) => {
        if (!prev[id] || prev[id].callId !== permissionReq?.callId) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (err) {
      if (id === activeSessionIdRef.current) setError(String(err));
    }
  };

  const handleUndoToolCall = async (messageId: string, callId: string) => {
    try {
      const updated = await invoke<AgentSession>("undo_agent_tool_call", {
        sessionId: session.id,
        messageId,
        callId,
      });
      if (activeSessionIdRef.current === updated.id) showSession(updated);
    } catch (err) {
      if (activeSessionIdRef.current === session.id) setError(String(err));
    }
  };

  const handleOpenRightTab = async (tab: "files" | "diff" | "browser") => {
    if (rightPanelOpen && rightPanelTab === tab) {
      setRightPanelOpen(false);
      return;
    }
    setRightPanelOpen(true);
    setRightPanelTab(tab);
    if (tab === "diff") {
      await refreshWorkspaceDiff(session.id, session.workspace);
    }
  };

  const handlePreviewWorkspaceFile = async (relPath: string) => {
    const request = ++previewRequestId.current;
    const id = session.id;
    const workspace = session.workspace;
    const current = () => request === previewRequestId.current && activeSessionIdRef.current === id && sessionRef.current.workspace === workspace;
    setSelectedPreviewFile(relPath);
    setSelectedPreviewContent("");
    try {
      const content = await invoke<string>("read_workspace_file_preview", {
        workspace: session.workspace || ".",
        relPath,
      });
      if (current()) setSelectedPreviewContent(content);
    } catch (err) {
      if (current()) setSelectedPreviewContent(`${locale === "es" ? "Error leyendo archivo" : "Error reading file"}: ${String(err)}`);
    }
  };

  const copyText = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId((current) => current === id ? null : current), 1500);
    } catch { setCopiedId(null); }
  };

  const lastUsage = contextUsage[session.id];
  const usage = lastUsage?.model_path === session.modelPath ? lastUsage : null;
  const profile = contextProfile?.modelPath === session.modelPath ? contextProfile.profile : null;
  const cloudContext = session.modelPath.startsWith("cloud::");
  const effectiveMaxTokens = (isCurrentRunning ? usage?.max_tokens : undefined)
    ?? (cloudContext ? 128000 : profile?.effectiveContextSize || usage?.max_tokens || 0);
  const modelDefaultCompact = profile?.compactAtTokens ?? 0;
  const inheritedCompact = !session.compactAtTokens;
  const configuredCompactTokens = Math.min(
    effectiveMaxTokens,
    session.compactAtTokens || modelDefaultCompact || Math.round(effectiveMaxTokens * 0.75),
  );
  const effectiveCompactTokens = (isCurrentRunning ? usage?.compact_at_tokens : undefined)
    ?? configuredCompactTokens;
  const compactKey = `${session.id}:${session.modelPath}`;
  const shownCompactTokens = !isCurrentRunning && compactDraft?.key === compactKey
    ? compactDraft.tokens : effectiveCompactTokens;
  const compactPct = effectiveMaxTokens ? (shownCompactTokens / effectiveMaxTokens) * 100 : 0;
  const commitCompactThreshold = (tokens: number) => {
    setCompactDraft(null);
    const current = sessionRef.current;
    if (current.id !== session.id || current.modelPath !== session.modelPath ||
      runningTurns.current.has(current.id) || !Number.isFinite(tokens) || tokens === effectiveCompactTokens) return;
    void persistSession({ ...current, compactAtTokens: tokens });
  };
  const usedPct = effectiveMaxTokens && usage ? (usage.used_tokens / effectiveMaxTokens) * 100 : 0;
  const formatTokens = (tokens: number) => tokens.toLocaleString(locale);
  const contextLabel = locale === "es" ? "Contexto" : "Context";
  const compactLabel = locale === "es" ? "Umbral de compactación" : "Compaction threshold";
  useEffect(() => {
    if (!wsMenuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (wsMenuRef.current && !wsMenuRef.current.contains(e.target as Node)) {
        setWsMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [wsMenuOpen]);

  useEffect(() => {
    if (!workModeMenuOpen) return;
    const dismissOutside = (event: MouseEvent) => {
      if (!workModeMenuRef.current?.contains(event.target as Node)) setWorkModeMenuOpen(false);
    };
    const dismissOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setWorkModeMenuOpen(false);
        workModeButtonRef.current?.focus();
      }
    };
    window.addEventListener("mousedown", dismissOutside);
    window.addEventListener("keydown", dismissOnEscape);
    return () => {
      window.removeEventListener("mousedown", dismissOutside);
      window.removeEventListener("keydown", dismissOnEscape);
    };
  }, [workModeMenuOpen]);

  useEffect(() => {
    if (!permMenuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (permMenuRef.current && !permMenuRef.current.contains(e.target as Node)) {
        setPermMenuOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPermMenuOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [permMenuOpen]);

  const projectLabels = useMemo(() => {
    const parts = projects.map((path) => path.split(/[\\/]/).filter(Boolean));
    return new Map(projects.map((path, index) => {
      const segments = parts[index];
      for (let count = 1; count <= segments.length; count += 1) {
        const label = segments.slice(-count).join("/");
        if (!parts.some((other, i) => i !== index &&
          other.slice(-count).join("/").toLowerCase() === label.toLowerCase())) {
          return [path, label];
        }
      }
      return [path, path];
    }));
  }, [projects]);

  const activeWsShort =
    !session.workspace || session.workspace === "."
      ? t("agent.noProject")
      : projectLabels.get(session.workspace) ?? session.workspace.split(/[\\/]/).filter(Boolean).pop() ?? session.workspace;

  const sidebarSessions = useMemo(() => {
    const byId = new Map(sessions.map((summary) => [summary.id, summary]));
    for (const id of runningSessionIds) {
      if (byId.has(id)) continue;
      const turn = runningTurns.current.get(id);
      if (turn) byId.set(id, summarizeSession(turn));
    }
    return [...byId.values()];
  }, [sessions, runningSessionIds]);

  const orphanSessions = useMemo(() => {
    const knownKeys = new Set(projects.map((p) => normWs(p)));
    return sidebarSessions.filter(
      (s) => !s.workspace || s.workspace === "." || !knownKeys.has(normWs(s.workspace))
    );
  }, [projects, sidebarSessions]);

  const retryUserMessageRef = useRef(sendTurn);
  retryUserMessageRef.current = sendTurn;
  const retryUserMessage = useCallback((msg: AgentChatMessage) => {
    void retryUserMessageRef.current(msg.content, false, msg.attachments);
  }, []);
  const assistantActions = useRef({ copyText, handleUndoToolCall });
  assistantActions.current = { copyText, handleUndoToolCall };
  const assistantCopy = useCallback((id: string, content: string) => { void assistantActions.current.copyText(id, content); }, []);
  const assistantUndo = useCallback((id: string, callId: string) => { void assistantActions.current.handleUndoToolCall(id, callId); }, []);
  const assistantToggleThinking = useCallback((id: string, next: boolean) => setExpandedThinking(p => ({ ...p, [id]: next })), []);
  const assistantToggleTool = useCallback((id: string, next: boolean) => setExpandedTools(p => ({ ...p, [id]: next })), []);


  return (
    <div className="agent-workspace-root">
      {sidebarOpen && (
        <>
          <aside
            className="agent-sidebar"
            style={{ width: sidebarWidth, minWidth: sidebarWidth }}
          >
          <div className="sidebar-top-action">
            <button
              type="button"
              className="sidebar-new-conv-btn"
              onClick={() => handleNewSession("")}
            >
              <Plus size={14} />
              <span>{t("agent.newConversation")}</span>
            </button>
          </div>

          <div className="projects-tree-scroll">
            {orphanSessions.length > 0 && (
              <div className="standalone-chats-block">
                {orphanSessions.map((s) => {
                  const isCurrent = s.id === session.id;
                  const isLive = runningSessionIds.includes(s.id);
                  const isRenaming = renamingId === s.id;
                  const isMenuOpen = chatMenuId === s.id;
                  return (
                    <div
                      key={s.id}
                      className={`project-chat-pill ${isCurrent ? "active" : ""} ${isMenuOpen ? "menu-open" : ""}`}
                      onClick={() => {
                        if (!isRenaming) void handleSelectSession(s.id);
                      }}
                    >
                      {s.pinned ? (
                        <Pin size={11} style={{ color: "#a78bfa", flexShrink: 0 }} />
                      ) : (
                        <MessageSquare size={12} style={{ color: "#64748b", flexShrink: 0 }} />
                      )}
                      {isRenaming ? (
                        <input
                          type="text"
                          autoFocus
                          className="chat-rename-input"
                          value={renamingDraft}
                          onClick={(e) => e.stopPropagation()}
                          onChange={(e) => setRenamingDraft(e.target.value)}
                          onBlur={() => void handleCommitRenameSession(s.id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.currentTarget.blur();
                            } else if (e.key === "Escape") {
                              setRenamingId(null);
                            }
                          }}
                        />
                      ) : (
                        <button type="button" aria-current={isCurrent ? "true" : undefined} className="project-chat-title truncate-text">
                          {displaySessionTitle(s.title, locale)}
                        </button>
                      )}
                      <span className="project-chat-right">
                        {isLive ? (
                          <span className="chat-active-dot" />
                        ) : (
                          <span className="project-chat-age tabular-nums">
                            {formatRelativeAge(s.updatedAt, locale)}
                          </span>
                        )}
                        <button
                          type="button"
                          className="chat-more-btn icon-btn"
                          title={locale === "es" ? "Opciones" : "Options"}
                          onClick={(e) => {
                            e.stopPropagation();
                            setChatMenuId(isMenuOpen ? null : s.id);
                          }}
                        >
                          <MoreHorizontal size={13} />
                        </button>
                      </span>
                      {isMenuOpen && (
                        <div
                          className="chat-row-menu"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            className="chat-row-menu-item"
                            onClick={() => void handleTogglePinSession(s.id)}
                          >
                            {s.pinned ? <PinOff size={12} /> : <Pin size={12} />}
                            <span>
                              {s.pinned
                                ? locale === "es" ? "Desfijar" : "Unpin"
                                : locale === "es" ? "Fijar" : "Pin"}
                            </span>
                          </button>
                          <button
                            type="button"
                            className="chat-row-menu-item"
                            onClick={() => handleStartRenameSession(s)}
                          >
                            <Pencil size={12} />
                            <span>{locale === "es" ? "Renombrar" : "Rename"}</span>
                          </button>
                          <button
                            type="button"
                            className="chat-row-menu-item danger"
                            onClick={(e) => {
                              setChatMenuId(null);
                              void handleDeleteSession(e, s.id);
                            }}
                          >
                            <Trash2 size={12} />
                            <span>{locale === "es" ? "Eliminar" : "Delete"}</span>
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            <div className="projects-section-bar">
              <span className="projects-title">
                {locale === "es" ? "Proyectos" : "Projects"}
              </span>
              <button
                type="button"
                className="icon-btn"
                title="Añadir carpeta de proyecto"
                onClick={() => void handleAddProject()}
              >
                <FolderPlus size={14} />
              </button>
            </div>
            {projects.length === 0 ? (
              <div className="projects-empty-cta">
                <button
                  type="button"
                  className={`ws-selector-card ${!session.workspace || session.workspace === "." ? "unset" : ""}`}
                  onClick={() => void handleAddProject()}
                >
                  <FolderPlus size={14} />
                  <span>{t("agent.addFirstProject")}</span>
                </button>
              </div>
            ) : (
              projects.map((projPath) => {
                const key = normWs(projPath);
                const projName =
                  projPath.split(/[\\/]/).filter(Boolean).pop() || projPath;
                const isActiveProj = normWs(session.workspace) === key;
                const isProjCollapsed = Boolean(collapsedProjects[key]);

                const savedChats = sidebarSessions.filter(
                  (s) => normWs(s.workspace) === key
                );
                const hasCurrentInSaved = savedChats.some(
                  (s) => s.id === session.id
                );
                const chatList: AgentSessionSummary[] =
                  isActiveProj && !hasCurrentInSaved
                    ? [
                        {
                          id: session.id,
                          title:
                            session.title ||
                            (locale === "es" ? "Nueva sesión" : "New session"),
                          workspace: projPath,
                          updatedAt: session.updatedAt,
                          messageCount: session.messages.length,
                          permissionMode: session.permissionMode,
                        },
                        ...savedChats,
                      ]
                    : savedChats;

                return (
                  <div key={key} className="project-group">
                    <div
                      className={`project-group-header ${isActiveProj ? "is-active" : ""}`}
                      title={projPath}
                      onClick={() =>
                        setCollapsedProjects((prev) => ({
                          ...prev,
                          [key]: !prev[key],
                        }))
                      }
                    >
                      <span className="project-folder-icon">
                        {isActiveProj ? <FolderOpen size={14} /> : <Folder size={14} />}
                      </span>
                      <span className="project-group-name truncate-text">
                        {projectLabels.get(projPath) ?? projName}
                      </span>
                      <div className="project-group-actions">
                        <button
                          type="button"
                          className="icon-btn"
                          title={`Nueva conversación en ${projName}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleNewSession(projPath);
                          }}
                        >
                          <Plus size={13} />
                        </button>
                        {confirmRemoveProj === projPath ? (
                          <>
                            <button
                              type="button"
                              className="icon-btn danger-hover"
                              style={{ color: "#f87171" }}
                              title={
                                locale === "es"
                                  ? "Confirmar: quitar proyecto de la lista (no borra archivos en disco)"
                                  : "Confirm: remove project from sidebar (disk files are kept)"
                              }
                              onClick={(e) => handleRemoveProject(e, projPath)}
                            >
                              <Check size={12} />
                            </button>
                            <button
                              type="button"
                              className="icon-btn"
                              title={locale === "es" ? "Cancelar" : "Cancel"}
                              onClick={(e) => {
                                e.stopPropagation();
                                setConfirmRemoveProj(null);
                              }}
                            >
                              <X size={12} />
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            className="icon-btn danger-hover"
                            title={
                              locale === "es"
                                ? "Quitar proyecto de la lista"
                                : "Remove project from list"
                            }
                            onClick={(e) => {
                              e.stopPropagation();
                              setConfirmRemoveProj(projPath);
                            }}
                          >
                            <Trash2 size={12} />
                          </button>
                        )}
                      </div>
                    </div>

                    {!isProjCollapsed && (
                      <div className="project-chats-list">
                        {chatList.length === 0 ? (
                          <button
                            type="button"
                            className="project-empty-chats"
                            onClick={() => handleNewSession(projPath)}
                          >
                            {locale === "es"
                              ? "Sin conversaciones aún"
                              : "No conversations yet"}
                          </button>
                        ) : (
                          chatList.map((s) => {
                            const isCurrent = s.id === session.id;
                            const isLive = runningSessionIds.includes(s.id);
                            const isRenaming = renamingId === s.id;
                            const isMenuOpen = chatMenuId === s.id;
                            const isSaved = savedChats.some((saved) => saved.id === s.id);
                            return (
                              <div
                                key={s.id}
                                className={`project-chat-pill ${isCurrent ? "active" : ""} ${isMenuOpen ? "menu-open" : ""}`}
                                onClick={() => {
                                  if (!isRenaming && (hasCurrentInSaved || !isCurrent)) {
                                    void handleSelectSession(s.id);
                                  }
                                }}
                              >
                                {s.pinned && (
                                  <Pin size={10} style={{ color: "#a78bfa", flexShrink: 0 }} />
                                )}
                                {isRenaming ? (
                                  <input
                                    type="text"
                                    autoFocus
                                    className="chat-rename-input"
                                    value={renamingDraft}
                                    onClick={(e) => e.stopPropagation()}
                                    onChange={(e) => setRenamingDraft(e.target.value)}
                                    onBlur={() => void handleCommitRenameSession(s.id)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.currentTarget.blur();
                                      } else if (e.key === "Escape") {
                                        setRenamingId(null);
                                      }
                                    }}
                                  />
                                ) : (
                                  <button type="button" aria-current={isCurrent ? "true" : undefined} className="project-chat-title truncate-text">
                                    {displaySessionTitle(s.title, locale)}
                                  </button>
                                )}
                                <span className="project-chat-right">
                                  {isLive ? (
                                    <span className="chat-active-dot" />
                                  ) : (
                                    <span className="project-chat-age tabular-nums">
                                      {formatRelativeAge(s.updatedAt, locale)}
                                    </span>
                                  )}
                                  <button
                                    type="button"
                                    className="chat-more-btn icon-btn"
                                    title={locale === "es" ? "Opciones" : "Options"}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setChatMenuId(isMenuOpen ? null : s.id);
                                    }}
                                  >
                                    <MoreHorizontal size={13} />
                                  </button>
                                </span>
                                {isMenuOpen && (
                                  <div
                                    className="chat-row-menu"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    <button
                                      type="button"
                                      className="chat-row-menu-item"
                                      onClick={() => void handleTogglePinSession(s.id)}
                                    >
                                      {s.pinned ? <PinOff size={12} /> : <Pin size={12} />}
                                      <span>
                                        {s.pinned
                                          ? locale === "es" ? "Desfijar" : "Unpin"
                                          : locale === "es" ? "Fijar" : "Pin"}
                                      </span>
                                    </button>
                                    <button
                                      type="button"
                                      className="chat-row-menu-item"
                                      onClick={() => handleStartRenameSession(s)}
                                    >
                                      <Pencil size={12} />
                                      <span>{locale === "es" ? "Renombrar" : "Rename"}</span>
                                    </button>
                                    {isSaved && (
                                      <button
                                        type="button"
                                        className="chat-row-menu-item danger"
                                        onClick={(e) => {
                                          setChatMenuId(null);
                                          void handleDeleteSession(e, s.id);
                                        }}
                                      >
                                        <Trash2 size={12} />
                                        <span>{locale === "es" ? "Eliminar" : "Delete"}</span>
                                      </button>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
          </aside>
          <PanelResizer
            label={locale === "es" ? "Ajustar panel izquierdo" : "Resize left panel"}
            side="left"
            width={sidebarWidth}
            min={190}
            max={420}
            defaultWidth={252}
            storageKey="loors_agent_left_w"
            onResize={setSidebarWidth}
          />
        </>
      )}

      <div
        className={`agent-main-col ${isDraggingOver ? "drop-overlay-active" : ""}`}
        onDragOver={(e) => {
          if (e.dataTransfer?.types?.includes("Files")) {
            e.preventDefault();
            setIsDraggingOver(true);
          }
        }}
        onDragLeave={(e) => {
          if (e.currentTarget === e.target) {
            setIsDraggingOver(false);
          }
        }}
        onDrop={(e) => {
          e.preventDefault();
          setIsDraggingOver(false);
          if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
            void stageExternalFiles(e.dataTransfer.files);
          }
        }}
      >
        <div className="agent-session-header">
          <div className="agent-session-identity">
            <button
              type="button"
              className="header-crumb-btn truncate-text"
              title={session.workspace || (locale === "es" ? "Elegir proyecto" : "Choose project")}
              onClick={() => void handlePickWorkspace()}
            >
              {activeWsShort}
            </button>
            <span className="header-crumb-sep">/</span>
            <input
              className="session-title-input"
              aria-label={locale === "es" ? "Título de la conversación" : "Conversation title"}
              value={displaySessionTitle(session.title, locale)}
              onChange={(e) => setSession({ ...session, title: e.target.value })}
              onBlur={() => void persistSession(session)}
            />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div className="workbench-tab-group">
              <button
                type="button"
                className={`workbench-tab-btn ${rightPanelOpen && rightPanelTab === "files" ? "active" : ""}`}
                onClick={() => void handleOpenRightTab("files")}
                title={t("agent.files")}
                aria-pressed={rightPanelOpen && rightPanelTab === "files"}
              >
                <Folder size={13} />
                <span>{t("agent.files")}</span>
              </button>
              <button
                type="button"
                className={`workbench-tab-btn ${rightPanelOpen && rightPanelTab === "diff" ? "active" : ""}`}
                aria-pressed={rightPanelOpen && rightPanelTab === "diff"}
                title={t("agent.diff")}
                onClick={() => void handleOpenRightTab("diff")}
              >
                <GitBranch size={13} />
                <span>{t("agent.diff")}</span>
              </button>
              <button
                type="button"
                className={`workbench-tab-btn ${rightPanelOpen && rightPanelTab === "browser" ? "active" : ""}`}
                aria-pressed={rightPanelOpen && rightPanelTab === "browser"}
                title={t("agent.browser")}
                onClick={() => void handleOpenRightTab("browser")}
              >
                <Globe size={13} />
                <span>{t("agent.browser")}</span>
                {liveBrowserFrame && <span className="mode-live-dot" />}
              </button>
            </div>
          </div>
        </div>

        <div className="agent-body-split">
          <div className="agent-chat-column">
          <div
            className="agent-feed-scroll"
            ref={feedRef}
            onScroll={() => {
              const el = feedRef.current;
              if (!el) return;
              stickToBottomRef.current =
                el.scrollHeight - el.scrollTop - el.clientHeight < 48;
              setShowJumpToLatest(!stickToBottomRef.current);
            }}
          >
            <div className="agent-feed-inner">
              {session.messages.length === 0 && (
                <div className="agent-zero-minimal">
                  <h2>{t("agent.heroTitle")}</h2>
                  <p>{locale === "es" ? "Escribe una tarea o parte de una idea." : "Write a task or start with an idea."}</p>
                  <div className="zero-suggestions">
                    {(session.workspace && session.workspace !== "."
                      ? [
                          { label: locale === "es" ? "Explorar el proyecto" : "Explore project", prompt: t("suggest.ws1"), Icon: FolderOpen },
                          { label: locale === "es" ? "Revisar cambios" : "Review changes", prompt: t("suggest.ws2"), Icon: GitBranch },
                          { label: locale === "es" ? "Revisar rendimiento" : "Review performance", prompt: t("suggest.ws3"), Icon: Zap },
                        ]
                      : [
                          { label: locale === "es" ? "Investigar un tema" : "Research a topic", prompt: t("suggest.free1"), Icon: Globe },
                          { label: locale === "es" ? "Diseñar una solución" : "Design a solution", prompt: t("suggest.free2"), Icon: Boxes },
                          { label: locale === "es" ? "Analizar un archivo" : "Analyze a file", prompt: t("suggest.free3"), Icon: FileText },
                        ]
                    ).map(({ label, prompt, Icon }) => (
                      <button
                        key={prompt}
                        type="button"
                        className="zero-suggestion-btn"
                        title={prompt}
                        onClick={() => {
                          setInput((draft) => draft.trim() ? `${draft}\n\n${prompt}` : prompt);
                          composerInputRef.current?.focus();
                        }}
                      >
                        <Icon size={15} aria-hidden="true" />
                        <span>{label}</span>
                        <ArrowUpRight size={12} aria-hidden="true" />
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <AgentHistory key={session.id} messages={session.messages} running={isCurrentRunning} locale={locale} modelPath={session.modelPath} expandedThinking={expandedThinking} copiedId={copiedId} modelLoadProgress={modelLoadProgress} expandedTools={expandedTools} savedShots={savedShots} onRetry={retryUserMessage} onCopy={assistantCopy} onToggleThinking={assistantToggleThinking} onToggleTool={assistantToggleTool} onUndo={assistantUndo} />

              {/* Interactive Permission Approval Card */}
              {permissionReq && (
                <div className="permission-card">
                  <div className="permission-head">
                    <Shield size={16} style={{ color: "#f59e0b" }} />
                    <span>
                      {locale === "es" ? "Permiso requerido:" : "Permission required:"} <strong>{permissionReq.toolName}</strong>
                    </span>
                  </div>
                  <pre className="permission-preview">{permissionReq.preview}</pre>
                  <div className="permission-actions">
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={() => void handlePermission(true, false)}
                    >
                      {locale === "es" ? "Aprobar 1 vez" : "Approve once"}
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => void handlePermission(true, true)}
                    >
                      Permitir siempre {permissionReq.toolName} en esta sesión
                    </button>
                    <button
                      type="button"
                      className="btn-danger"
                      onClick={() => void handlePermission(false, false)}
                    >
                      {locale === "es" ? "Rechazar" : "Reject"}
                    </button>
                  </div>
                </div>
              )}

              {gpuBusyPrompt && (
                <div className="permission-card" style={{ borderColor: "rgba(99,102,241,0.5)" }}>
                  <div className="permission-head">
                    <Zap size={15} style={{ color: "#818cf8" }} />
                    <span>El Arnés está usando llama-server (GPU) ahora mismo</span>
                  </div>
                  <div className="permission-actions">
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={() => void sendTurn(gpuBusyPrompt, true)}
                    >
                      Detener arnés y enviar ahora
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => setGpuBusyPrompt(null)}
                    >
                      Esperar
                    </button>
                  </div>
                </div>
              )}

              {error && <div className="hw-error">{error}</div>}
              {isCurrentRunning && harnessProgress[session.id] && (
                <div className="agent-turn-status is-running" role="status">
                  <span className="agent-turn-status-label">{({
                    reasoning: locale === "es" ? "Razonando" : "Reasoning",
                    executing: locale === "es" ? "Ejecutando herramientas" : "Executing tools",
                    verifying: locale === "es" ? "Verificando" : "Verifying",
                    recovering: locale === "es" ? "Replanteando estrategia" : "Recovering strategy",
                  } as Record<string, string>)[harnessProgress[session.id].phase] ?? harnessProgress[session.id].phase}</span>
                  {harnessProgress[session.id].task_title && (
                    <span className="agent-turn-status-detail" title={harnessProgress[session.id].task_title ?? undefined}>
                      {harnessProgress[session.id].task_title}
                    </span>
                  )}
                </div>
              )}
              {!isCurrentRunning && turnOutcomes[session.id] && turnOutcomes[session.id].status !== "completed" && (
                <div className="agent-turn-status" role="status">
                  <strong>{({
                    incomplete: locale === "es" ? "Trabajo incompleto" : "Incomplete work",
                    cancelled: locale === "es" ? "Cancelado" : "Cancelled",
                    step_limit: locale === "es" ? "Límite de iteraciones" : "Iteration limit",
                    error: locale === "es" ? "Error de ejecución" : "Execution error",
                  } as Record<string, string>)[turnOutcomes[session.id].status] ?? turnOutcomes[session.id].status}</strong>
                  {turnOutcomes[session.id].summary && <span>{turnOutcomes[session.id].summary}</span>}
                </div>
              )}
            </div>
          </div>

          {showJumpToLatest && (
            <div className="agent-jump-latest">
              <button type="button" className="btn-secondary btn-xs" onClick={() => {
                stickToBottomRef.current = true;
                setShowJumpToLatest(false);
                const el = feedRef.current;
                if (el) el.scrollTop = el.scrollHeight;
              }}><ArrowDown size={13} /> {locale === "es" ? "Ir al último mensaje" : "Jump to latest"}</button>
            </div>
          )}
          <div className="composer-wrap">
            {pendingQuestion && (
              <div
                style={{
                  marginBottom: 8,
                  padding: "12px 14px",
                  borderRadius: 10,
                  background: "var(--bg-surface-1)",
                  border: "1px solid var(--accent)",
                  display: "flex",
                  flexDirection: "column",
                  gap: 8,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                  <strong style={{ fontSize: 13 }}>{pendingQuestion.question}</strong>
                  <button
                    type="button"
                    className="icon-btn"
                    title="Dismiss"
                    onClick={() => {
                      setPendingQuestion(null);
                      void invoke("resolve_agent_question", { sessionId: session.id, decision: { answers: [] } });
                    }}
                  >
                    <X size={13} />
                  </button>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {pendingQuestion.options.map((opt) => {
                    const selected = selectedQuestionOpts.includes(opt);
                    return (
                      <button
                        key={opt}
                        type="button"
                        className={selected ? "btn-primary btn-xs" : "btn-secondary btn-xs"}
                        onClick={() => {
                          if (!pendingQuestion.allowMultiple) {
                            setPendingQuestion(null);
                            void invoke("resolve_agent_question", { sessionId: session.id, decision: { answers: [opt] } });
                          } else {
                            setSelectedQuestionOpts((prev) =>
                              prev.includes(opt) ? prev.filter((x) => x !== opt) : [...prev, opt]
                            );
                          }
                        }}
                      >
                        {opt}
                      </button>
                    );
                  })}
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    type="text"
                    placeholder="Or type a custom answer..."
                    value={customQuestionAnswer}
                    onChange={(e) => setCustomQuestionAnswer(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (customQuestionAnswer.trim() || selectedQuestionOpts.length > 0)) {
                        const answers = [
                          ...selectedQuestionOpts,
                          ...(customQuestionAnswer.trim() ? [customQuestionAnswer.trim()] : []),
                        ];
                        setPendingQuestion(null);
                        void invoke("resolve_agent_question", { sessionId: session.id, decision: { answers } });
                      }
                    }}
                    style={{ flex: 1, fontSize: 12, padding: "5px 8px", borderRadius: 6, background: "var(--bg-surface-2)", border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
                  />
                  <button
                    type="button"
                    className="btn-primary btn-xs"
                    disabled={selectedQuestionOpts.length === 0 && !customQuestionAnswer.trim()}
                    onClick={() => {
                      const answers = [
                        ...selectedQuestionOpts,
                        ...(customQuestionAnswer.trim() ? [customQuestionAnswer.trim()] : []),
                      ];
                      setPendingQuestion(null);
                      void invoke("resolve_agent_question", { sessionId: session.id, decision: { answers } });
                    }}
                  >
                    Send
                  </button>
                </div>
              </div>
            )}
            {activeArtifact && (
              <div className={`agent-artifact-card ${artifactExpanded ? "expanded" : ""}`}>
                <div className="agent-artifact-head">
                  <button
                    type="button"
                    className="agent-artifact-toggle"
                    title={activeArtifact.path || activeArtifact.title}
                    onClick={() => setArtifactExpanded((v) => !v)}
                  >
                    {artifactExpanded ? (
                      <ChevronDown size={13} className="agent-artifact-chev" />
                    ) : (
                      <ChevronRight size={13} className="agent-artifact-chev" />
                    )}
                    <FileText size={13} className="agent-artifact-icon" />
                    <span className="agent-artifact-title truncate-text">
                      {activeArtifact.title}
                    </span>
                    <span className="agent-artifact-badge">
                      {activeArtifact.kind === "markdown" ? "MD" : activeArtifact.kind}
                    </span>
                  </button>
                  <div className="agent-artifact-actions">
                    <button
                      type="button"
                      className="agent-artifact-btn"
                      title={
                        copiedId === "artifact-copy"
                          ? locale === "es"
                            ? "¡Copiado!"
                            : "Copied!"
                          : locale === "es"
                            ? "Copiar contenido"
                            : "Copy content"
                      }
                      onClick={() => copyText("artifact-copy", activeArtifact.content)}
                    >
                      {copiedId === "artifact-copy" ? (
                        <>
                          <Check size={12} style={{ color: "#34d399" }} />
                          <span>{locale === "es" ? "Copiado" : "Copied"}</span>
                        </>
                      ) : (
                        <>
                          <Copy size={12} />
                          <span>{locale === "es" ? "Copiar" : "Copy"}</span>
                        </>
                      )}
                    </button>
                    <button
                      type="button"
                      className="agent-artifact-close"
                      title={locale === "es" ? "Cerrar artefacto" : "Dismiss artifact"}
                      onClick={() => { artifactsBySession.current.delete(session.id); saveAgentAuxMap("loors_agent_artifacts_v1", artifactsBySession.current); setActiveArtifact(null); }}
                    >
                      <X size={13} />
                    </button>
                  </div>
                </div>
                {artifactExpanded && (
                  <div className="agent-artifact-body">
                    {activeArtifact.kind === "html" ? (
                      <iframe
                        title={activeArtifact.title}
                        sandbox="allow-scripts"
                        srcDoc={activeArtifact.content}
                        className="agent-artifact-iframe"
                      />
                    ) : (
                      <div className="agent-artifact-md">
                        <MarkdownContent text={activeArtifact.content} />
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            {todoItems.length > 0 && (
              <div
                style={{
                  marginBottom: 8,
                  padding: "8px 12px",
                  borderRadius: 10,
                  background: "var(--bg-surface-1)",
                  border: "1px solid var(--border-subtle)",
                  fontSize: 12,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 8,
                    cursor: "pointer",
                  }}
                  onClick={() => setTodoCollapsed((v) => !v)}
                >
                  <strong style={{ fontSize: 11.5, color: "var(--text-primary)" }}>
                    Tasks ({todoItems.filter((i) => i.status === "done").length}/{todoItems.length})
                  </strong>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ width: 20, height: 20 }}
                      title="Clear checklist"
                      onClick={(e) => {
                        e.stopPropagation();
                        todosBySession.current.delete(session.id); saveAgentAuxMap("loors_agent_todos_v1", todosBySession.current);
                        setTodoItems([]);
                      }}
                    >
                      <X size={12} />
                    </button>
                  </div>
                </div>
                {!todoCollapsed && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
                    {todoItems.map((item, idx) => {
                      const done = item.status === "done";
                      const active = item.status === "in_progress";
                      const failed = item.status === "failed";
                      return (
                        <div
                          key={`${idx}-${item.title}`}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 7,
                            color: done
                              ? "var(--text-muted)"
                              : failed
                                ? "#f87171"
                                : active
                                  ? "var(--text-primary)"
                                  : "var(--text-secondary)",
                            textDecoration: done ? "line-through" : "none",
                          }}
                        >
                          <span
                            style={{
                              width: 14,
                              display: "inline-flex",
                              justifyContent: "center",
                              color: done
                                ? "#4ade80"
                                : active
                                  ? "#a78bfa"
                                  : failed
                                    ? "#f87171"
                                    : "var(--text-muted)",
                              fontWeight: 700,
                            }}
                          >
                            {done ? "✓" : active ? "●" : failed ? "✕" : "○"}
                          </span>
                          <span>{item.title}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            <div className="floating-composer">
              <MessageQueueTray
                items={messageQueue}
                isBusy={isCurrentRunning}
                onRemove={(id) =>
                  setMessageQueue((prev) => prev.filter((q) => q.id !== id))
                }
                onRunNext={() => {
                  const id = session.id;
                  if (activeSessionIdRef.current !== id || runningTurns.current.has(id)) return;
                  const next = queuedMessages.find(item => item.sessionId === id);
                  if (!next) return;
                  const target = sessionRef.current;
                  if (target.id !== id || !target.modelPath) return;
                  setMessageQueue(prev => prev.filter(item => item.id !== next.id));
                  void sendTurn(next.text, false, next.attachments ?? [], target);
                }}
              />
              <AgentComposerAttachments pendingAttachments={pendingAttachments} pinnedFiles={session.pinnedFiles} stagingFiles={!!stagingFiles} locale={locale} onRemove={att => setPendingAttachments(prev => prev.filter(x => x !== att))} onUnpin={rel => void togglePinFile(rel)} />

              {!isCurrentRunning && (
                <AgentExecutionCheckpoint checkpoint={session.executionCheckpoint} workspace={session.workspace}
                  locale={locale} disabled={!session.modelPath || Boolean(input.trim()) || pendingAttachments.length > 0}
                  onContinue={(command) => void sendTurn(command, false, [])} />
              )}
              <AgentComposerInput inputRef={composerInputRef} value={input} placeholder={isCurrentRunning ? t("agent.composerQueuePlaceholder") : t("agent.composerPlaceholder")} label={t("agent.composerPlaceholder")} onChange={setInput} onFiles={(files) => void stageExternalFiles(files)} onSend={() => void sendTurn(composerPrompt())} />

              <div className="composer-bar">
                <div className="composer-left">
                <div className="composer-model-wrap" ref={attachMenuRef}>
                  <button
                    type="button"
                    className={`composer-plus-btn ${attachMenuOpen ? "active" : ""}`}
                    aria-label={locale === "es" ? "Adjuntar archivos" : "Attach files"}
                    title={locale === "es" ? "Adjuntar archivos" : "Attach files"}
                    aria-expanded={attachMenuOpen}
                    onClick={() => setAttachMenuOpen((v) => !v)}
                  >
                    <Plus size={14} />
                  </button>

                  {attachMenuOpen && (
                    <div className="composer-pin-popover">
                      <button
                        type="button"
                        className="attach-menu-action"
                        onClick={() => {
                          setAttachMenuOpen(false);
                          void handlePickAttachFiles();
                        }}
                      >
                        <Paperclip size={13} />
                        <span>
                          {locale === "es"
                            ? "Elegir archivos…"
                            : "Choose files…"}
                        </span>
                      </button>
                      <p className="attachment-help">
                        {locale === "es"
                          ? "Arrastra archivos al chat o pega una captura con Ctrl+V."
                          : "Drop files into the chat or paste a screenshot with Ctrl+V."}
                      </p>
                      <div className="model-pop-search">
                        <Search size={13} />
                        <input
                          type="text"
                          autoFocus
                          placeholder={
                            locale === "es"
                              ? "Fijar archivo del proyecto…"
                              : "Pin a project file…"
                          }
                          value={pinQuery}
                          onChange={(e) => void handleSearchPin(e.target.value)}
                        />
                      </div>
                      <div className="composer-pin-results">
                        {pinResults.length === 0 ? (
                          <div className="muted-empty" style={{ padding: 10 }}>
                            {locale === "es"
                              ? "Sin archivos que coincidan en el proyecto."
                              : "No matching files in this project."}
                          </div>
                        ) : (
                          pinResults.map((rel) => {
                            const pinned = session.pinnedFiles.includes(rel);
                            return (
                              <button
                                key={rel}
                                type="button"
                                className={`pin-result-item ${pinned ? "is-pinned" : ""}`}
                                onClick={() => togglePinFile(rel)}
                              >
                                <FileText size={12} />
                                <span className="truncate-text" style={{ flex: 1 }}>
                                  {rel}
                                </span>
                                {pinned && <Check size={12} />}
                              </button>
                            );
                          })
                        )}
                      </div>
                    </div>
                  )}
                </div>

                  <span className="composer-sep" aria-hidden="true" />
                  <div className="composer-model-wrap" ref={modelPickerRef}>
                    <AgentModelPickerTrigger modelPath={session.modelPath} cloudProviders={cloudProviders} activeModelMeta={activeModelMeta} chooseLabel={t("agent.chooseModel")} modelPickerOpen={modelPickerOpen} onToggle={() => setModelPickerOpen(prev => !prev)} />

                    {modelPickerOpen && (
                      <div className="model-popover">
                        <AgentModelSearch query={modelPickerQuery} onQuery={setModelPickerQuery} hiddenCount={hiddenModels.length} showHidden={showHiddenModels} onToggleHidden={() => setShowHiddenModels(v => !v)} onClose={() => setModelPickerOpen(false)} label={t("agent.searchModel")} locale={locale} />

                        <AgentModelList modelPath={session.modelPath} cloudProviders={cloudProviders} localModels={localModels} groupedPickerModels={groupedPickerModels} modelPickerQuery={modelPickerQuery} pinnedModels={pinnedModels} hiddenModels={hiddenModels} showHiddenModels={showHiddenModels} collapsedModelGroups={collapsedModelGroups} locale={locale} onSelect={path => { setActiveModelPath(path); void persistSession({ ...sessionRef.current, modelPath: path }); setModelPickerOpen(false); }} onToggleGroup={key => setCollapsedModelGroups(prev => ({ ...prev, [key]: !prev[key] }))} togglePinModel={togglePinModel} toggleHideModel={toggleHideModel} />

                        <div className="model-pop-runtime-bar">
                          <span className="model-pop-runtime-label">Runtime</span>
                          <div className="model-pop-runtime-chips">
                            {(["auto", ...installedBackends] as RuntimeBackendKind[])
                              .filter((v, i, arr) => arr.indexOf(v) === i)
                              .map((bk) => (
                                <button
                                  key={bk}
                                  type="button"
                                  className={`runtime-chip-btn ${selectedBackend === bk ? "active" : ""}`}
                                  onClick={() => setSelectedBackend(bk)}
                                >
                                  {bk === "auto"
                                    ? "Auto"
                                    : bk === "vulkan"
                                      ? "Vulkan"
                                      : bk === "metal"
                                        ? "Metal"
                                        : bk.toUpperCase()}
                                </button>
                              ))}
                          </div>
                        </div>

                        {session.modelPath && !isCloudModelPath(session.modelPath) && (
                          <button
                            type="button"
                            className="model-pop-tuning"
                            onClick={() => {
                              setModelPickerOpen(false);
                              openTuningModal(session.modelPath);
                            }}
                          >
                            <Settings size={14} />
                            {locale === "es" ? "Ajustes de inferencia" : "Inference settings"}
                          </button>
                        )}
                        <div className="model-pop-footer" style={{ display: "flex", gap: 6 }}>
                          <button
                            type="button"
                            className="model-pop-hub-btn"
                            style={{ flex: 1 }}
                            onClick={() => {
                              setModelPickerOpen(false);
                              window.dispatchEvent(
                                new CustomEvent("loors:open-models-modal")
                              );
                            }}
                          >
                            {t("agent.localModelsBtn")}
                          </button>
                          <button
                            type="button"
                            className="model-pop-hub-btn"
                            style={{ flex: 1 }}
                            onClick={() => {
                              setModelPickerOpen(false);
                              window.dispatchEvent(
                                new CustomEvent("loors:open-models-modal", {
                                  detail: { tab: "cloud" },
                                })
                              );
                            }}
                          >
                            {t("agent.cloudApisBtn")}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  <ReasoningEffortSelect
                    value={session.reasoningEffort ?? "medium"}
                    modelPath={session.modelPath}
                    onChange={(next) =>
                      void persistSession({ ...session, reasoningEffort: next })
                    }
                  />
                </div>

                <div className="composer-right">
                  <AgentComposerActions running={isCurrentRunning} hasContent={Boolean(input.trim() || pendingAttachments.length)} hasModel={Boolean(session.modelPath)} locale={locale} onStop={() => void handleStop()} onSend={() => void sendTurn(composerPrompt())} />
                </div>
              </div>
            </div>

            <div className="composer-env-bar">
              <div className="composer-env-left">
                <div className="effort-pop-wrap" ref={wsMenuRef}>
                  <button
                    type="button"
                    className={`composer-env-pill ${!session.workspace || session.workspace === "." ? "is-none" : ""}`}
                    title={session.workspace || "Sin proyecto enlazado"}
                    onClick={() => setWsMenuOpen((v) => !v)}
                  >
                    <Folder size={12} />
                    <span className="truncate-text" style={{ maxWidth: 160 }}>
                      {activeWsShort}
                    </span>
                    <ChevronUp size={11} className="effort-pill-chev" />
                  </button>

                  {wsMenuOpen && (
                    <div className="effort-popover" role="menu" style={{ minWidth: 220 }}>
                      <div className="effort-popover-title">
                        {locale === "es"
                          ? "Proyecto de esta conversación"
                          : "Conversation workspace"}
                      </div>
                      <button
                        type="button"
                        role="menuitem"
                        className={`effort-pop-item ${!session.workspace || session.workspace === "." ? "active" : ""}`}
                        onClick={() => {
                          setWsMenuOpen(false);
                          void persistSession({ ...session, workspace: "", pinnedFiles: [] });
                        }}
                      >
                        <div className="effort-pop-item-text">
                          <span className="effort-pop-item-label">
                            {locale === "es" ? "Sin proyecto" : "No project"}
                          </span>
                          <span className="effort-pop-item-desc">
                            {locale === "es" ? "Conversación libre" : "General chat"}
                          </span>
                        </div>
                        {(!session.workspace || session.workspace === ".") && (
                          <Check size={12} style={{ color: "#c084fc", flexShrink: 0 }} />
                        )}
                      </button>
                      {projects.map((p) => {
                        const name = p.split(/[\\/]/).filter(Boolean).pop() || p;
                        const active = normWs(session.workspace) === normWs(p);
                        return (
                          <button
                            key={p}
                            type="button"
                            role="menuitem"
                            className={`effort-pop-item ${active ? "active" : ""}`}
                            title={p}
                            onClick={() => {
                              setWsMenuOpen(false);
                              void persistSession({ ...session, workspace: p, pinnedFiles: [] });
                            }}
                          >
                            <div className="effort-pop-item-text">
                              <span className="effort-pop-item-label truncate-text">{name}</span>
                            </div>
                            {active && (
                              <Check size={12} style={{ color: "#c084fc", flexShrink: 0 }} />
                            )}
                          </button>
                        );
                      })}
                      <button
                        type="button"
                        role="menuitem"
                        className="effort-pop-item"
                        onClick={() => {
                          setWsMenuOpen(false);
                          void handlePickWorkspace();
                        }}
                      >
                        <div className="effort-pop-item-text">
                          <span className="effort-pop-item-label" style={{ color: "#a78bfa" }}>
                            {locale === "es"
                              ? "+ Elegir otra carpeta…"
                              : "+ Choose another folder…"}
                          </span>
                        </div>
                      </button>
                    </div>
                  )}
                </div>

                <div className="effort-pop-wrap" ref={workModeMenuRef}>
                  <button
                    ref={workModeButtonRef}
                    type="button"
                    className={`composer-env-pill ${workModeMenuOpen ? "is-open" : ""}`}
                    aria-label={`${workModeLabel}: ${activeWorkMode.label}`}
                    aria-expanded={workModeMenuOpen}
                    title={workModeLabel}
                    onClick={() => {
                      setPermMenuOpen(false);
                      setWorkModeMenuOpen((open) => !open);
                    }}
                  >
                    <WorkModeIcon size={12} />
                    <span>{activeWorkMode.label}</span>
                    <ChevronUp size={11} className="effort-pill-chev" />
                  </button>
                  {workModeMenuOpen && (
                    <div className="effort-popover composer-work-mode-menu" role="group" aria-label={workModeLabel}>
                      <div className="effort-popover-title">{workModeLabel}</div>
                      {workModes.map(({ id, label, icon: Icon, description }) => (
                        <button key={id} type="button" aria-pressed={id === workMode}
                          className={`effort-pop-item ${id === workMode ? "active" : ""}`}
                          onClick={() => {
                            setWorkMode(id);
                            setWorkModeMenuOpen(false);
                            workModeButtonRef.current?.focus();
                          }}>
                          <Icon size={14} />
                          <span className="effort-pop-item-text">
                            <span className="effort-pop-item-label">{label}</span>
                            <span className="effort-pop-item-desc">{description}</span>
                          </span>
                          {id === workMode && <Check size={12} />}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="effort-pop-wrap" ref={permMenuRef}>
                  <AgentPermissionPicker mode={session.permissionMode} permMenuOpen={permMenuOpen} onToggle={() => { setWorkModeMenuOpen(false); setPermMenuOpen(v => !v); }} onSelect={mode => { setPermMenuOpen(false); void persistSession({ ...sessionRef.current, permissionMode: mode }); }} />
                </div>
              </div>

              <div className="composer-env-right">
                <div className="composer-model-wrap" ref={compactPopoverRef}>
                  <button
                    type="button"
                    className={`ctx-trigger-btn tabular-nums ${compactPopoverOpen ? "active" : ""}`}
                    aria-expanded={compactPopoverOpen}
                    aria-label={usage ? `${contextLabel} · ≈ ${formatTokens(usage.used_tokens)} tokens` : contextLabel}
                    title={usage ? `${contextLabel} · ≈ ${formatTokens(usage.used_tokens)} / ${formatTokens(effectiveMaxTokens)} tokens` : contextLabel}
                    onClick={() => setCompactPopoverOpen((v) => !v)}
                  >
                    <span className="context-gauge" aria-hidden="true" style={{
                      background: `conic-gradient(var(--context-accent) ${Math.min(100, usedPct)}%, var(--border-strong) 0)`,
                    }} />
                    {usage && effectiveMaxTokens ? `${Math.round(usedPct)}%` : contextLabel}
                  </button>

                  {compactPopoverOpen && (
                    <div className="compact-popover">
                      <div className="context-card">
                        <div className="sheet-row-top">
                          <strong>{contextLabel}</strong>
                          <span className="sheet-readout tabular-nums">
                            {usage ? `≈ ${formatTokens(usage.used_tokens)}` : "—"}
                            {effectiveMaxTokens > 0 && ` / ${formatTokens(effectiveMaxTokens)}`} tokens
                          </span>
                        </div>
                        <div className="context-track" aria-hidden="true">
                          <span className="context-fill" style={{ width: `${Math.min(100, usedPct)}%` }} />
                          {effectiveMaxTokens > 0 && (
                            <span className="context-marker" style={{ left: `${compactPct}%` }} />
                          )}
                        </div>
                        <p className="context-hint">
                          {cloudContext
                            ? (locale === "es" ? "Entrada estimada · Límite de referencia para la API." : "Estimated input · Reference limit for the API.")
                            : (locale === "es" ? "Entrada estimada de la última solicitud." : "Estimated input for the latest request.")}
                        </p>
                      </div>
                      {effectiveMaxTokens > 0 && (
                        <div className="context-card">
                          <div className="sheet-row-top">
                            <label className="sheet-label" htmlFor="session-compaction">{compactLabel}</label>
                            <span className="sheet-readout tabular-nums">{Math.round(compactPct)}%</span>
                          </div>
                          <input
                            id="session-compaction"
                            className="context-threshold-slider"
                            type="range"
                            min={1}
                            max={effectiveMaxTokens}
                            step={1}
                            value={Math.max(1, shownCompactTokens)}
                            aria-valuetext={`${Math.round(compactPct)}% · ${formatTokens(shownCompactTokens)} tokens`}
                            disabled={isCurrentRunning}
                            onChange={(event) => setCompactDraft({ key: compactKey, tokens: event.currentTarget.valueAsNumber })}
                            onPointerUp={(event) => commitCompactThreshold(event.currentTarget.valueAsNumber)}
                            onKeyUp={(event) => commitCompactThreshold(event.currentTarget.valueAsNumber)}
                            onBlur={(event) => commitCompactThreshold(event.currentTarget.valueAsNumber)}
                          />
                          <div className="context-threshold-footer">
                            <span className="context-hint tabular-nums">{formatTokens(shownCompactTokens)} tokens</span>
                            <button
                              type="button"
                              className={`sheet-pill-toggle ${inheritedCompact ? "active" : ""}`}
                              aria-pressed={inheritedCompact}
                              disabled={isCurrentRunning}
                              onClick={() => {
                                setCompactDraft(null);
                                void persistSession({ ...sessionRef.current, compactAtTokens: 0 });
                              }}
                            >
                              {modelDefaultCompact ? (locale === "es" ? "Del modelo" : "Model default") : "Auto · 75%"}
                            </button>
                          </div>
                        </div>
                      )}

                      <div className="sheet-divider" />

                      <label className="check-inline">
                        <input
                          type="checkbox"
                          checked={session.useRag}
                          onChange={(e) =>
                            void persistSession({ ...session, useRag: e.target.checked })
                          }
                        />
                        <span>
                          {locale === "es" ? "Búsqueda en el proyecto (RAG)" : "Project search (RAG)"}
                        </span>
                      </label>

                      <p className="context-hint">{locale === "es"
                        ? "Añade un mapa del proyecto y permite buscar fragmentos relevantes en sus archivos."
                        : "Adds a project map and lets the agent search for relevant snippets in its files."}</p>

                      <AgentLimitSettings
                        key={session.id}
                        limits={session.agentLimits}
                        disabled={isCurrentRunning}
                        onSave={(agentLimits) => {
                          const current = sessionRef.current;
                          if (current.id === session.id) void persistSession({ ...current, agentLimits });
                        }}
                      />

                      {skills.length > 0 && (
                        <fieldset className="context-skills" aria-describedby="context-skills-help">
                          <legend>{locale === "es" ? "Skills de la conversación" : "Conversation skills"}</legend>
                          <p id="context-skills-help" className="context-hint">
                            {locale === "es" ? "Marca las skills cuyas instrucciones quieres añadir al chat." : "Select the skills whose instructions you want to add to this chat."}
                          </p>
                          <div className="skills-checklist">
                            {skills.map((sk) => {
                              const checked = session.enabledSkills.includes(sk.path);
                              return (
                                <label key={sk.path} className="check-inline">
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => {
                                      const next = checked
                                        ? session.enabledSkills.filter((p) => p !== sk.path)
                                        : [...session.enabledSkills, sk.path];
                                      void persistSession({ ...session, enabledSkills: next });
                                    }}
                                  />
                                  <span className="truncate-text" title={sk.description}>{sk.name === "architect-loopy"
                                    ? locale === "es" ? "Guía antigua de arneses" : "Legacy harness guide"
                                    : sk.name}</span>
                                </label>
                              );
                            })}
                          </div>
                        </fieldset>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
          </div>

          {rightPanelOpen && (
            <>
              {!(browserPanel && browserExpanded) && (
              <PanelResizer
                key={rightPanelTab}
                label={locale === "es" ? "Ajustar panel derecho" : "Resize right panel"}
                side="right"
                width={activePanelWidth}
                min={browserPanel ? 420 : diffPanel ? 320 : 260}
                max={browserPanel ? 980 : 680}
                defaultWidth={browserPanel ? 580 : diffPanel ? 420 : 290}
                storageKey={browserPanel ? "loors_agent_browser_w" : diffPanel ? "loors_agent_diff_w" : "loors_agent_right_w"}
                onResize={browserPanel ? setBrowserPanelWidth : diffPanel ? setDiffPanelWidth : setRightPanelWidth}
              />
              )}
              <aside
                className={`agent-workbench-drawer ${rightPanelTab === "browser" ? "browser-mode" : ""} ${rightPanelTab === "browser" && browserExpanded ? "expanded" : ""}`}
                style={browserPanel && browserExpanded ? undefined : {
                  width: `min(${activePanelWidth}px, ${browserPanel ? 58 : diffPanel ? 46 : 34}vw)`,
                  minWidth: 0,
                }}
              >
              {rightPanelTab !== "browser" && (
              <div className="workbench-drawer-head">
                <div className="workbench-drawer-title">
                  {rightPanelTab === "files" && (
                    <>
                      <Folder size={13} />
                      <span>
                        {locale === "es" ? "Archivos" : "Files"} · {activeWsShort}
                      </span>
                    </>
                  )}
                  {rightPanelTab === "diff" && (
                    <>
                      <GitBranch size={13} />
                      <span>Git Diff · {activeWsShort}</span>
                    </>
                  )}
                </div>
              </div>
              )}

              {rightPanelTab === "files" && (
                <div className="workbench-pane">
                  <div className="workbench-search-bar">
                    <Search size={12} />
                    <input
                      type="text"
                      placeholder={
                        locale === "es"
                          ? "Filtrar archivos del proyecto…"
                          : "Filter project files…"
                      }
                      value={fileFilter}
                      onChange={(e) => {
                        const q = e.target.value;
                        setFileFilter(q);
                      }}
                    />
                  </div>

                  {selectedPreviewFile ? (
                    <div className="workbench-file-preview">
                      <div className="workbench-file-preview-head">
                        <button
                          type="button"
                          className="btn-mini"
                          onClick={() => setSelectedPreviewFile(null)}
                        >
                          ← Volver al árbol
                        </button>
                        <span className="truncate-text" style={{ flex: 1, fontSize: 11.5, fontWeight: 600 }}>
                          {selectedPreviewFile}
                        </span>
                        <button
                          type="button"
                          className={`btn-mini ${session.pinnedFiles.includes(selectedPreviewFile) ? "btn-active" : ""}`}
                          onClick={() => togglePinFile(selectedPreviewFile)}
                          title="Fijar archivo al contexto del chat"
                        >
                          <Pin size={11} />
                          <span>
                            {session.pinnedFiles.includes(selectedPreviewFile) ? "Fijado" : "Fijar"}
                          </span>
                        </button>
                      </div>
                      <pre className="workbench-code-pre">{selectedPreviewContent}</pre>
                    </div>
                  ) : (
                    <div className="workbench-files-list">
                      {(!session.workspace || session.workspace === ".") ? (
                        <div className="muted-empty" style={{ padding: 14 }}>
                          {locale === "es"
                          ? "Elige un proyecto a la izquierda para ver su árbol de archivos."
                          : "Select a project on the left to browse its file tree."}
                        </div>
                      ) : workspaceFiles.length === 0 ? (
                        <div className="muted-empty" style={{ padding: 14 }}>
                          Sin archivos que coincidan.
                        </div>
                      ) : (
                        <WorkspaceFileTree
                          files={workspaceFiles}
                          pinnedFiles={session.pinnedFiles}
                          filterActive={fileFilter.trim().length > 0}
                          onOpenFile={(rel) => void handlePreviewWorkspaceFile(rel)}
                          onTogglePin={togglePinFile}
                        />
                      )}
                    </div>
                  )}
                  <div className="workbench-drawer-foot">
                    <span>
                      {locale === "es"
                        ? `${workspaceFiles.length} archivos en ${activeWsShort}`
                        : `${workspaceFiles.length} files in ${activeWsShort}`}
                    </span>
                    {session.pinnedFiles.length > 0 && (
                      <span>
                        {locale === "es"
                          ? `${session.pinnedFiles.length} fijado(s)`
                          : `${session.pinnedFiles.length} pinned`}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {rightPanelTab === "diff" && (
                <div className="workbench-pane">
                  <div className="workbench-subbar">
                    <span>
                      {locale === "es"
                        ? "Cambios sin confirmar en el repositorio"
                        : "Uncommitted repository changes"}
                    </span>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => {
                        void refreshWorkspaceDiff(activeSessionIdRef.current, sessionRef.current.workspace);
                      }}
                    >
                      <RefreshCw size={11} />
                      <span>{locale === "es" ? "Actualizar" : "Refresh"}</span>
                    </button>
                  </div>
                  <div className="agent-diff-body">
                    {gitDiffText.trim() ? (
                      <AgentDiff diff={gitDiffText} locale={locale} />
                    ) : (
                      <div className="muted-empty" style={{ padding: 14 }}>
                        {locale === "es"
                          ? "Sin cambios pendientes en Git. Cuando el agente edite archivos verás el diff aquí automáticamente."
                          : "No pending Git changes. File diffs will appear here automatically when the agent edits files."}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {rightPanelTab === "browser" && (
                <InteractiveBrowserPane
                  sessionId={session.id}
                  agentFrame={liveBrowserFrame}
                  agentBusy={running}
                  expanded={browserExpanded}
                  onToggleExpand={() => setBrowserExpanded((v) => !v)}
                  onClosePane={() => setRightPanelOpen(false)}
                />
              )}
            </aside>
            </>
          )}
        </div>
      </div>

      {/* First-run setup modal (only shown once on fresh install until configured or skipped) */}
      {setupChecked &&
        !onboardingDismissed &&
        !(
          (installedBackends.length > 0 && localModels.length > 0) ||
          cloudProviders.some((p) => p.enabled && p.models.length > 0)
        ) && (
          <div className="modal-backdrop" onClick={dismissOnboarding}>
            <div
              className="modal-card"
              style={{ maxWidth: 460, padding: "20px 22px 18px" }}
              onClick={(e) => e.stopPropagation()}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: 12,
                  marginBottom: 18,
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}>
                    {locale === "es" ? "Bienvenido a Loors" : "Welcome to Loors"}
                  </h3>
                  <span style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.4 }}>
                    {t("onboarding.title")}
                  </span>
                </div>
                <button
                  type="button"
                  className="icon-btn"
                  style={{ flexShrink: 0, marginTop: -2, marginRight: -4 }}
                  onClick={dismissOnboarding}
                  title={locale === "es" ? "Cerrar" : "Close"}
                >
                  <X size={15} />
                </button>
              </div>

              <div className="onboarding-card" style={{ border: "none", padding: 0, background: "transparent", width: "100%", maxWidth: "none", gap: 8 }}>
                <div style={{ fontSize: 10.5, fontWeight: 600, color: "#71717a", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  {t("onboarding.optA")}
                </div>

                <button
                  type="button"
                  className={`onboarding-step ${installedBackends.length > 0 ? "is-done" : ""}`}
                  disabled={autoInstallingRt}
                  onClick={() => {
                    if (installedBackends.length > 0) {
                      dismissOnboarding();
                      window.dispatchEvent(new CustomEvent("loors:open-runtime-modal"));
                    } else {
                      void handleAutoInstallRuntime();
                    }
                  }}
                >
                  <span className="onboarding-step-num">
                    {installedBackends.length > 0 ? <Check size={12} /> : <Download size={12} />}
                  </span>
                  <div className="onboarding-step-text">
                    <strong>
                      {installedBackends.length > 0
                        ? t("onboarding.rtReady")
                        : autoInstallingRt
                          ? t("onboarding.rtInstalling")
                          : t("onboarding.rtInstallCta")}
                    </strong>
                    <span>
                      {installedBackends.length > 0
                        ? `${t("onboarding.rtInstalledPrefix")} (${installedBackends.join(", ")})`
                        : rtInstallProgress?.message
                          ? rtInstallProgress.message
                          : t("onboarding.rtInstallDesc")}
                    </span>
                  </div>
                </button>

                {autoInstallingRt && rtInstallProgress?.percent !== null && rtInstallProgress?.percent !== undefined && (
                  <div className="progress-track" style={{ marginTop: -2 }}>
                    <div
                      className="progress-fill"
                      style={{
                        width: `${Math.min(100, Math.max(0, rtInstallProgress.percent))}%`,
                      }}
                    />
                  </div>
                )}

                <button
                  type="button"
                  className={`onboarding-step ${localModels.length > 0 ? "is-done" : ""}`}
                  onClick={() => {
                    dismissOnboarding();
                    window.dispatchEvent(new CustomEvent("loors:open-models-modal"));
                  }}
                >
                  <span className="onboarding-step-num">
                    {localModels.length > 0 ? <Check size={12} /> : <Boxes size={12} />}
                  </span>
                  <div className="onboarding-step-text">
                    <strong>{t("onboarding.ggufCta")}</strong>
                    <span>
                      {localModels.length > 0
                        ? `${localModels.length} ${t("onboarding.ggufDetected")}`
                        : t("onboarding.ggufDesc")}
                    </span>
                  </div>
                </button>

                <div style={{ fontSize: 10.5, fontWeight: 600, color: "#71717a", textTransform: "uppercase", letterSpacing: "0.06em", marginTop: 8 }}>
                  {t("onboarding.optB")}
                </div>

                <button
                  type="button"
                  className="onboarding-step"
                  onClick={() => {
                    dismissOnboarding();
                    window.dispatchEvent(
                      new CustomEvent("loors:open-models-modal", {
                        detail: { tab: "cloud" },
                      })
                    );
                  }}
                >
                  <span className="onboarding-step-num">
                    <Cloud size={12} />
                  </span>
                  <div className="onboarding-step-text">
                    <strong>{t("onboarding.cloudCta")}</strong>
                    <span>{t("onboarding.cloudDesc")}</span>
                  </div>
                </button>
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  marginTop: 16,
                  paddingTop: 12,
                  borderTop: "1px solid var(--border)",
                }}
              >
                <button
                  type="button"
                  className="btn-secondary"
                  style={{ fontSize: 12, padding: "5px 12px" }}
                  onClick={dismissOnboarding}
                >
                  {locale === "es" ? "Configurar más tarde" : "Skip for now"}
                </button>
              </div>
            </div>
          </div>
        )}
    </div>
  );
}
