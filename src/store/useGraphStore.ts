import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";
import {
  NODE_BLUEPRINTS,
  type AppMode,
  type CustomNodeDef,
  type EngineEvent,
  type HumanGatePrompt,
  type LoorsFlowEdge,
  type LoorsFlowNode,
  type LoorsNodeKind,
  type LoorsWorkflowFile,
  type ModelDownloadEvent,
  type ModelDownloadItem,
  type ModelLoadProgressEvent,
  type NodeExecutionResult,
  type RunLogEntry,
  type RunStatus,
  type RuntimeBackendKind,
  type SwarmAgentLiveStatus,
  type SwarmBoardLiveMessage,
} from "../types/graph";
import { invoke } from "../lib/bridge";
import { HARNESS_TEMPLATES } from "../templates/harnesses";

const INITIAL_NODES: LoorsFlowNode[] = [
  {
    id: "input-1",
    type: "loorsNode",
    position: { x: 40, y: 180 },
    data: {
      kind: "task_input",
      title: "Task Input",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.task_input.defaultConfig },
    },
  },
  {
    id: "model-1",
    type: "loorsNode",
    position: { x: 40, y: 380 },
    data: {
      kind: "local_model",
      title: "Local Model (GGUF)",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.local_model.defaultConfig },
    },
  },
  {
    id: "agent-1",
    type: "loorsNode",
    position: { x: 340, y: 200 },
    data: {
      kind: "agent",
      title: "Agent: Coder",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.agent.defaultConfig },
    },
  },
  {
    id: "shell-1",
    type: "loorsNode",
    position: { x: 640, y: 200 },
    data: {
      kind: "shell",
      title: "Shell: Run Tests",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.shell.defaultConfig },
    },
  },
  {
    id: "cond-1",
    type: "loorsNode",
    position: { x: 920, y: 200 },
    data: {
      kind: "condition",
      title: "Tests Pass?",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.condition.defaultConfig },
    },
  },
  {
    id: "guard-1",
    type: "loorsNode",
    position: { x: 640, y: 420 },
    data: {
      kind: "loop_guard",
      title: "Loop Guard (max 5)",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.loop_guard.defaultConfig },
    },
  },
  {
    id: "out-1",
    type: "loorsNode",
    position: { x: 1200, y: 150 },
    data: {
      kind: "output",
      title: "Done",
      status: "idle",
      config: { ...NODE_BLUEPRINTS.output.defaultConfig },
    },
  },
];

const INITIAL_EDGES: LoorsFlowEdge[] = [
  {
    id: "e-in-agent",
    source: "input-1",
    sourceHandle: "exec",
    target: "agent-1",
    targetHandle: "exec",
  },
  {
    id: "e-model-agent",
    source: "model-1",
    sourceHandle: "model_ref",
    target: "agent-1",
    targetHandle: "model_ref",
    style: { stroke: "#a855f7", strokeDasharray: "5 4" },
  },
  {
    id: "e-agent-shell",
    source: "agent-1",
    sourceHandle: "exec",
    target: "shell-1",
    targetHandle: "exec",
  },
  {
    id: "e-shell-cond",
    source: "shell-1",
    sourceHandle: "exec",
    target: "cond-1",
    targetHandle: "exec",
  },
  {
    id: "e-cond-out",
    source: "cond-1",
    sourceHandle: "true",
    target: "out-1",
    targetHandle: "exec",
    style: { stroke: "#22c55e" },
    label: "pass",
  },
  {
    id: "e-cond-guard",
    source: "cond-1",
    sourceHandle: "false",
    target: "guard-1",
    targetHandle: "exec",
    style: { stroke: "#ef4444" },
    label: "fail",
  },
  {
    id: "e-guard-agent",
    source: "guard-1",
    sourceHandle: "retry",
    target: "agent-1",
    targetHandle: "feedback",
    animated: true,
    style: { stroke: "#22c55e", strokeWidth: 2 },
    label: "loop retry",
  },
];

export interface HarnessTabState {
  id: string;
  workflowName: string;
  savedLibraryName: string | null;
  dirty: boolean;
  nodes: LoorsFlowNode[];
  edges: LoorsFlowEdge[];
  selectedNodeId: string | null;
  runStatus: RunStatus;
  activeRunId: string | null;
  nodeResults: Record<string, NodeExecutionResult>;
  currentIteration: number;
  maxIterations: number | null;
  runSummary: string | null;
  activeActivity: string | null;
  logs: RunLogEntry[];
  pendingGate: HumanGatePrompt | null;
}

interface GraphStoreState {
  harnessTabs: HarnessTabState[];
  activeHarnessTabId: string;
  savedLibraryName: string | null;
  dirty: boolean;
  autoSaveHarnesses: boolean;
  setAutoSaveHarnesses: (enabled: boolean) => void;
  markSaved: (libraryName?: string) => void;
  createHarnessTab: () => void;
  switchHarnessTab: (tabId: string) => void;
  closeHarnessTab: (tabId: string) => void;
  workflowName: string;
  nodes: LoorsFlowNode[];
  edges: LoorsFlowEdge[];
  selectedNodeId: string | null;
  selectedBackend: RuntimeBackendKind;
  runStatus: RunStatus;
  activeRunId: string | null;
  nodeResults: Record<string, NodeExecutionResult>;
  currentIteration: number;
  maxIterations: number | null;
  runSummary: string | null;
  activeActivity: string | null;
  loadedModelName: string | null;
  loadedContextSize: number | null;
  modelLoadProgress: ModelLoadProgressEvent | null;
  modelDownloads: Record<string, ModelDownloadItem>;
  downloadsDockOpen: boolean;
  swarmAgents: Record<string, SwarmAgentLiveStatus>;
  swarmMessages: SwarmBoardLiveMessage[];
  logs: RunLogEntry[];
  consoleOpen: boolean;
  pendingGate: HumanGatePrompt | null;
  customNodes: CustomNodeDef[];
  appMode: AppMode;
  locale: "en" | "es";
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  rightPanelTab: "inspector" | "architect";
  dimResourceEdges: boolean;
  tuningModalModelPath: string | null;
  editingCustomNode: CustomNodeDef | null;
  architectSeedPrompt: string | null;
  agentRunning: boolean;
  agentActiveTitle: string | null;
  activeModelPath: string;
  setAppMode: (mode: AppMode) => void;
  setLocale: (locale: "en" | "es") => void;
  setLeftPanelOpen: (open: boolean) => void;
  setRightPanelOpen: (open: boolean) => void;
  setActiveModelPath: (modelPath: string) => void;
  setModelLoadProgress: (ev: ModelLoadProgressEvent | null) => void;
  setDownloadsDockOpen: (open: boolean) => void;
  upsertModelDownload: (ev: ModelDownloadEvent) => void;
  startModelDownload: (repoId: string, filename: string, totalBytesHint?: number | null) => Promise<string | null>;
  cancelModelDownload: (repoId: string, filename: string) => Promise<void>;
  dismissModelDownload: (key: string) => void;
  clearFinishedDownloads: () => void;
  setRightPanelTab: (tab: "inspector" | "architect") => void;
  setDimResourceEdges: (dim: boolean) => void;
  openTuningModal: (modelPath: string | null) => void;
  setEditingCustomNode: (def: CustomNodeDef | null) => void;
  setArchitectSeedPrompt: (prompt: string | null) => void;
  setAgentStatus: (running: boolean, title?: string | null) => void;
  syncAgentContextToHarness: (workspace: string, modelPath: string) => void;
  setCustomNodes: (defs: CustomNodeDef[]) => void;
  addCustomNodeToCanvas: (def: CustomNodeDef) => void;
  attachResourceToNode: (
    targetNodeId: string,
    resourceKind: "local_model" | "skill" | "mcp_server" | "knowledge_rag" | "workspace"
  ) => void;
  duplicateNode: (id: string) => void;
  setWorkflowName: (name: string) => void;
  setSelectedNodeId: (id: string | null) => void;
  setSelectedBackend: (backend: RuntimeBackendKind) => void;
  setConsoleOpen: (open: boolean) => void;
  clearLogs: () => void;
  clearPendingGate: () => void;
  newWorkflow: () => void;
  loadTemplate: (templateId: string, useDemo?: boolean) => void;
  prepareRun: () => void;
  applyEngineEvent: (event: EngineEvent & { tabId?: string }) => void;
  onNodesChange: (changes: NodeChange<LoorsFlowNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<LoorsFlowEdge>[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (kind: LoorsNodeKind, position?: { x: number; y: number }) => void;
  autoLayout: () => void;
  updateNodeTitle: (id: string, title: string) => void;
  updateNodeConfig: (id: string, patch: Record<string, unknown>) => void;
  deleteNode: (id: string) => void;
  deleteGraphElements: (nodeIds: string[], edgeIds: string[]) => void;
  undoGraphEdit: () => void;
  redoGraphEdit: () => void;
  exportWorkflow: () => LoorsWorkflowFile;
  importWorkflow: (
    file: LoorsWorkflowFile,
    libraryName?: string | null,
    opts?: { preserveModels?: boolean; preserveLibraryLink?: boolean; preserveSelection?: boolean; dirty?: boolean }
  ) => void;
}

function formatNow(): string {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

function makeHarnessTabId(): string {
  return `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 5)}`;
}

function snapshotActiveTab(s: GraphStoreState): HarnessTabState {
  return {
    id: s.activeHarnessTabId,
    workflowName: s.workflowName,
    savedLibraryName: s.savedLibraryName,
    dirty: s.dirty,
    nodes: s.nodes,
    edges: s.edges,
    selectedNodeId: s.selectedNodeId,
    runStatus: s.runStatus,
    activeRunId: s.activeRunId,
    nodeResults: s.nodeResults,
    currentIteration: s.currentIteration,
    maxIterations: s.maxIterations,
    runSummary: s.runSummary,
    activeActivity: s.activeActivity,
    logs: s.logs,
    pendingGate: s.pendingGate,
  };
}

function applyEventToTabState(tab: HarnessTabState, event: EngineEvent): HarnessTabState {
  switch (event.type) {
    case "run_started":
      return {
        ...tab,
        activeRunId: event.run_id,
        nodeResults: {},
        runStatus: "running",
        currentIteration: 1,
        runSummary: null,
      };
    case "node_result": {
      if (tab.activeRunId !== event.run_id || !tab.nodes.some((node) => node.id === event.node_id)
        || (tab.nodeResults[event.node_id]?.step ?? 0) >= event.step) return tab;
      return { ...tab, nodeResults: { ...tab.nodeResults, [event.node_id]: event } };
    }
    case "node_status": {
      const targetNode = tab.nodes.find((n) => n.id === event.node_id);
      const activity =
        event.status === "running"
          ? `${targetNode?.data.title ?? event.node_id}${event.message ? ` · ${event.message}` : ""}`
          : tab.activeActivity;
      return {
        ...tab,
        currentIteration: event.iteration,
        activeActivity: activity,
        nodes: tab.nodes.map((n) =>
          n.id === event.node_id
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: event.status,
                  statusMessage: event.status === "running" ? event.message : null,
                },
              }
            : n
        ),
      };
    }
    case "node_log": {
      const entry: RunLogEntry = {
        id: `${event.run_id}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: formatNow(),
        nodeId: event.node_id,
        nodeTitle: event.node_title,
        iteration: event.iteration,
        stream: event.stream,
        text: event.text,
      };
      return {
        ...tab,
        logs: [...tab.logs.slice(-500), entry],
      };
    }
    case "iteration_changed":
      return {
        ...tab,
        currentIteration: event.iteration,
        maxIterations: event.max_iterations,
      };
    case "human_gate_waiting":
      return {
        ...tab,
        pendingGate: {
          runId: event.run_id,
          nodeId: event.node_id,
          nodeTitle: event.node_title,
          iteration: event.iteration,
          message: event.message,
          gitDiff: event.git_diff,
        },
      };
    case "run_finished":
      return {
        ...tab,
        runStatus: event.outcome,
        currentIteration: event.total_iterations,
        runSummary: event.summary,
        activeActivity: null,
        pendingGate: null,
      };
    default:
      return tab;
  }
}

const INITIAL_TAB_ID = "tab-1";
const TABS_STORAGE_KEY = "loors_harness_tabs_v1";

interface PersistedHarnessTab {
  id: string;
  workflowName: string;
  savedLibraryName: string | null;
  dirty?: boolean;
  nodes: LoorsFlowNode[];
  edges: LoorsFlowEdge[];
}

function loadInitialTabsState(): { tabs: HarnessTabState[]; activeId: string } {
  const fallback: HarnessTabState = {
    id: INITIAL_TAB_ID,
    workflowName: "fix-tests-loop",
    savedLibraryName: null,
    dirty: false,
    nodes: INITIAL_NODES,
    edges: INITIAL_EDGES,
    selectedNodeId: null,
    runStatus: "idle",
    activeRunId: null,
    nodeResults: {},
    currentIteration: 0,
    maxIterations: null,
    runSummary: null,
    activeActivity: null,
    logs: [],
    pendingGate: null,
  };
  if (typeof window === "undefined") {
    return { tabs: [fallback], activeId: INITIAL_TAB_ID };
  }
  try {
    const raw = window.localStorage.getItem(TABS_STORAGE_KEY);
    if (!raw) return { tabs: [fallback], activeId: INITIAL_TAB_ID };
    const parsed = JSON.parse(raw) as {
      activeId?: string;
      tabs?: PersistedHarnessTab[];
    };
    if (!Array.isArray(parsed.tabs) || parsed.tabs.length === 0) {
      return { tabs: [fallback], activeId: INITIAL_TAB_ID };
    }
    const hydrated: HarnessTabState[] = parsed.tabs
      .filter((t) => t && typeof t.id === "string" && Array.isArray(t.nodes))
      .map((t) => ({
        id: t.id,
        workflowName: t.workflowName || "harness",
        savedLibraryName: t.savedLibraryName ?? null,
        dirty: Boolean(t.dirty),
        nodes: t.nodes.map((n) => ({
          ...n,
          type: "loorsNode" as const,
          data: { ...n.data, status: "idle" as const, statusMessage: null },
        })),
        edges: Array.isArray(t.edges) ? t.edges : [],
        selectedNodeId: null,
        runStatus: "idle",
        activeRunId: null,
        nodeResults: {},
        currentIteration: 0,
        maxIterations: null,
        runSummary: null,
        activeActivity: null,
        logs: [],
        pendingGate: null,
      }));
    if (hydrated.length === 0) {
      return { tabs: [fallback], activeId: INITIAL_TAB_ID };
    }
    const activeId = hydrated.some((t) => t.id === parsed.activeId)
      ? (parsed.activeId as string)
      : hydrated[0].id;
    return { tabs: hydrated, activeId };
  } catch {
    return { tabs: [fallback], activeId: INITIAL_TAB_ID };
  }
}

const BOOT_TABS = loadInitialTabsState();
const BOOT_ACTIVE =
  BOOT_TABS.tabs.find((t) => t.id === BOOT_TABS.activeId) ?? BOOT_TABS.tabs[0];

const initialAppMode: AppMode = typeof window !== "undefined"
  && window.localStorage.getItem("loors_app_mode") === "harness" ? "harness" : "agent";
const panelStorageKey = (mode: AppMode, side: "left" | "right") => `loors_${mode}_${side}_open`;

function readPanelOpen(mode: AppMode, side: "left" | "right") {
  const saved = typeof window === "undefined"
    ? null : window.localStorage.getItem(panelStorageKey(mode, side));
  return saved === null ? side === "left" || mode === "harness" : saved === "true";
}

type GraphSnapshot = Pick<GraphStoreState, "nodes" | "edges">;
type GraphEdit = Partial<GraphSnapshot & Pick<GraphStoreState, "selectedNodeId">>;

interface GraphHistory {
  undo: GraphSnapshot[];
  redo: GraphSnapshot[];
  dragStart?: GraphSnapshot;
}

const graphHistories = new Map<string, GraphHistory>();

function getGraphHistory(tabId: string): GraphHistory {
  let history = graphHistories.get(tabId);
  if (!history) {
    history = { undo: [], redo: [] };
    graphHistories.set(tabId, history);
  }
  return history;
}

function graphSnapshot({ nodes, edges }: GraphSnapshot): GraphSnapshot {
  return { nodes, edges };
}

function editGraph(patch: GraphEdit, before?: GraphSnapshot) {
  const state = useGraphStore.getState();
  const history = getGraphHistory(state.activeHarnessTabId);
  history.undo.push(before ?? graphSnapshot(state));
  if (history.undo.length > 100) history.undo.shift();
  history.redo = [];
  useGraphStore.setState({ ...patch, dirty: true });
}

function restoreGraphEdit(direction: "undo" | "redo") {
  const state = useGraphStore.getState();
  const history = getGraphHistory(state.activeHarnessTabId);
  if (history.dragStart) return;
  const snapshot = history[direction].pop();
  if (!snapshot) return;

  history[direction === "undo" ? "redo" : "undo"].push(graphSnapshot(state));
  const liveNodes = new Map(state.nodes.map((node) => [node.id, node]));
  const selectedEdges = new Set(state.edges.filter((edge) => edge.selected).map((edge) => edge.id));
  useGraphStore.setState({
    nodes: snapshot.nodes.map((node) => {
      const live = liveNodes.get(node.id);
      return {
        ...node,
        dragging: false,
        selected: live?.selected ?? false,
        data: {
          ...node.data,
          status: live?.data.status ?? "idle",
          statusMessage: live?.data.statusMessage ?? null,
        },
      };
    }),
    edges: snapshot.edges.map((edge) => ({ ...edge, selected: selectedEdges.has(edge.id) })),
    selectedNodeId: state.selectedNodeId && snapshot.nodes.some((node) => node.id === state.selectedNodeId)
      ? state.selectedNodeId
      : null,
    dirty: true,
  });
}

export const useGraphStore = create<GraphStoreState>((set, get) => ({
  harnessTabs: BOOT_TABS.tabs,
  activeHarnessTabId: BOOT_ACTIVE.id,
  savedLibraryName: BOOT_ACTIVE.savedLibraryName,
  dirty: BOOT_ACTIVE.dirty,
  autoSaveHarnesses:
    typeof window !== "undefined"
      ? window.localStorage.getItem("loors_auto_save_harnesses") !== "0"
      : true,
  setAutoSaveHarnesses: (enabled) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("loors_auto_save_harnesses", enabled ? "1" : "0");
    }
    set({ autoSaveHarnesses: enabled });
  },
  workflowName: BOOT_ACTIVE.workflowName,
  nodes: BOOT_ACTIVE.nodes,
  edges: BOOT_ACTIVE.edges,
  selectedNodeId: null,
  selectedBackend: "auto",
  runStatus: "idle",
  activeRunId: null,
  nodeResults: {},
  currentIteration: 0,
  maxIterations: null,
  runSummary: null,
  activeActivity: null,
  loadedModelName: null,
  loadedContextSize: null,
  modelLoadProgress: null,
  modelDownloads: {},
  downloadsDockOpen: false,
  swarmAgents: {},
  swarmMessages: [],
  logs: [],
  consoleOpen: false,
  pendingGate: null,

  markSaved: (libraryName) => {
    const s = get();
    const libName = libraryName ?? s.savedLibraryName ?? s.workflowName;
    set({
      savedLibraryName: libName,
      dirty: false,
      harnessTabs: s.harnessTabs.map((t) =>
        t.id === s.activeHarnessTabId
          ? { ...t, savedLibraryName: libName, dirty: false, workflowName: s.workflowName }
          : t
      ),
    });
  },

  createHarnessTab: () => {
    const s = get();
    const savedActive = snapshotActiveTab(s);
    const updatedTabs = s.harnessTabs.map((t) =>
      t.id === s.activeHarnessTabId ? savedActive : t
    );
    const activeModel = s.activeModelPath.trim();
    const shortModelTitle =
      activeModel.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "Local Model (GGUF)";
    const newId = makeHarnessTabId();
    const newNodes: LoorsFlowNode[] = [
      {
        id: "in-1",
        type: "loorsNode",
        position: { x: 40, y: 160 },
        data: {
          kind: "task_input",
          title: "Task Input",
          status: "idle",
          config: { ...NODE_BLUEPRINTS.task_input.defaultConfig },
        },
      },
      {
        id: "model-1",
        type: "loorsNode",
        position: { x: 40, y: 360 },
        data: {
          kind: "local_model",
          title: shortModelTitle,
          status: "idle",
          config: {
            ...NODE_BLUEPRINTS.local_model.defaultConfig,
            ...(activeModel ? { modelPath: activeModel } : {}),
          },
        },
      },
      {
        id: "agent-1",
        type: "loorsNode",
        position: { x: 360, y: 160 },
        data: {
          kind: "agent",
          title: "Agent",
          status: "idle",
          config: { ...NODE_BLUEPRINTS.agent.defaultConfig },
        },
      },
      {
        id: "out-1",
        type: "loorsNode",
        position: { x: 680, y: 160 },
        data: {
          kind: "output",
          title: "Output",
          status: "idle",
          config: { ...NODE_BLUEPRINTS.output.defaultConfig },
        },
      },
    ];
    const newEdges: LoorsFlowEdge[] = [
      { id: "e-1", source: "in-1", sourceHandle: "exec", target: "agent-1", targetHandle: "exec" },
      { id: "e-m", source: "model-1", sourceHandle: "model_ref", target: "agent-1", targetHandle: "model_ref", style: { stroke: "#a855f7", strokeDasharray: "5 4" } },
      { id: "e-2", source: "agent-1", sourceHandle: "exec", target: "out-1", targetHandle: "exec" },
    ];
    const newTab: HarnessTabState = {
      id: newId,
      workflowName: `harness-${updatedTabs.length + 1}`,
      savedLibraryName: null,
      dirty: false,
      nodes: newNodes,
      edges: newEdges,
      selectedNodeId: null,
      runStatus: "idle",
      activeRunId: null,
      nodeResults: {},
      currentIteration: 0,
      maxIterations: null,
      runSummary: null,
      activeActivity: null,
      logs: [],
      pendingGate: null,
    };
    set({
      harnessTabs: [...updatedTabs, newTab],
      activeHarnessTabId: newId,
      savedLibraryName: null,
      dirty: false,
      workflowName: newTab.workflowName,
      nodes: newTab.nodes,
      edges: newTab.edges,
      selectedNodeId: null,
      runStatus: "idle",
      activeRunId: null,
      nodeResults: {},
      currentIteration: 0,
      maxIterations: null,
      runSummary: null,
      activeActivity: null,
      logs: [],
      pendingGate: null,
    });
  },

  switchHarnessTab: (tabId) => {
    const s = get();
    if (tabId === s.activeHarnessTabId) return;
    const savedActive = snapshotActiveTab(s);
    const syncedTabs = s.harnessTabs.map((t) =>
      t.id === s.activeHarnessTabId ? savedActive : t
    );
    const target = syncedTabs.find((t) => t.id === tabId);
    if (!target) return;
    set({
      harnessTabs: syncedTabs,
      activeHarnessTabId: target.id,
      savedLibraryName: target.savedLibraryName,
      dirty: target.dirty,
      workflowName: target.workflowName,
      nodes: target.nodes,
      edges: target.edges,
      selectedNodeId: target.selectedNodeId,
      runStatus: target.runStatus,
      activeRunId: target.activeRunId,
      nodeResults: target.nodeResults,
      currentIteration: target.currentIteration,
      maxIterations: target.maxIterations,
      runSummary: target.runSummary,
      activeActivity: target.activeActivity,
      logs: target.logs,
      pendingGate: target.pendingGate,
    });
  },

  closeHarnessTab: (tabId) => {
    const s = get();
    if (s.harnessTabs.length <= 1) return;
    const savedActive = snapshotActiveTab(s);
    const syncedTabs = s.harnessTabs.map((t) =>
      t.id === s.activeHarnessTabId ? savedActive : t
    );
    const idx = syncedTabs.findIndex((t) => t.id === tabId);
    if (idx === -1) return;
    graphHistories.delete(tabId);
    const remaining = syncedTabs.filter((t) => t.id !== tabId);
    if (tabId !== s.activeHarnessTabId) {
      set({ harnessTabs: remaining });
      return;
    }
    const next = remaining[Math.max(0, idx - 1)] ?? remaining[0];
    set({
      harnessTabs: remaining,
      activeHarnessTabId: next.id,
      savedLibraryName: next.savedLibraryName,
      dirty: next.dirty,
      workflowName: next.workflowName,
      nodes: next.nodes,
      edges: next.edges,
      selectedNodeId: next.selectedNodeId,
      runStatus: next.runStatus,
      activeRunId: next.activeRunId,
      nodeResults: next.nodeResults,
      currentIteration: next.currentIteration,
      maxIterations: next.maxIterations,
      runSummary: next.runSummary,
      activeActivity: next.activeActivity,
      logs: next.logs,
      pendingGate: next.pendingGate,
    });
  },
  customNodes: [],
  appMode: initialAppMode,
  locale:
    typeof window !== "undefined" && window.localStorage.getItem("loors_locale") === "es"
      ? "es"
      : "en",
  leftPanelOpen: readPanelOpen(initialAppMode, "left"),
  rightPanelOpen: readPanelOpen(initialAppMode, "right"),
  rightPanelTab: "inspector",
  dimResourceEdges: true,
  tuningModalModelPath: null,
  editingCustomNode: null,
  architectSeedPrompt: null,
  agentRunning: false,
  agentActiveTitle: null,
  activeModelPath:
    (typeof window !== "undefined" && window.localStorage.getItem("loors_active_model_path")) ||
    "",

  setAppMode: (appMode) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("loors_app_mode", appMode);
    }
    if (get().appMode !== appMode) {
      set({
        appMode,
        leftPanelOpen: readPanelOpen(appMode, "left"),
        rightPanelOpen: readPanelOpen(appMode, "right"),
      });
    }
  },
  setLocale: (locale) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("loors_locale", locale);
    }
    set({ locale });
  },
  setLeftPanelOpen: (leftPanelOpen) => {
    if (typeof window !== "undefined") window.localStorage.setItem(panelStorageKey(get().appMode, "left"), String(leftPanelOpen));
    set({ leftPanelOpen });
  },
  setRightPanelOpen: (rightPanelOpen) => {
    if (typeof window !== "undefined") window.localStorage.setItem(panelStorageKey(get().appMode, "right"), String(rightPanelOpen));
    set({ rightPanelOpen });
  },
  setModelLoadProgress: (modelLoadProgress) => set({ modelLoadProgress }),
  setDownloadsDockOpen: (downloadsDockOpen) => set({ downloadsDockOpen }),
  upsertModelDownload: (ev) => {
    const key = `${ev.repo_id}::${ev.filename}`;
    set((s) => {
      const prev = s.modelDownloads[key];
      return {
        modelDownloads: {
          ...s.modelDownloads,
          [key]: {
            ...ev,
            total_bytes: ev.total_bytes ?? prev?.total_bytes ?? null,
            saved_path: ev.saved_path ?? prev?.saved_path ?? null,
            key,
            updatedAt: Date.now(),
          },
        },
      };
    });
  },
  startModelDownload: async (repoId, filename, totalBytesHint = null) => {
    const key = `${repoId}::${filename}`;
    const existing = get().modelDownloads[key];
    if (existing && (existing.stage === "starting" || existing.stage === "downloading")) {
      return null;
    }
    set((s) => ({
      downloadsDockOpen: true,
      modelDownloads: {
        ...s.modelDownloads,
        [key]: {
          key,
          repo_id: repoId,
          filename,
          downloaded_bytes: 0,
          total_bytes: totalBytesHint ?? null,
          percent: 0,
          speed_bps: null,
          saved_path: null,
          stage: "starting",
          message:
            get().locale === "es"
              ? `Iniciando descarga: ${filename}...`
              : `Starting download: ${filename}...`,
          updatedAt: Date.now(),
        },
      },
    }));
    try {
      const savedPath = await invoke<string>("download_hf_model", {
        repoId,
        filename,
      });
      set((s) => {
        const cur = s.modelDownloads[key];
        return {
          modelDownloads: {
            ...s.modelDownloads,
            [key]: {
              ...(cur ?? {
                key,
                repo_id: repoId,
                filename,
                downloaded_bytes: totalBytesHint ?? 0,
                total_bytes: totalBytesHint ?? null,
              }),
              percent: 100,
              speed_bps: null,
              saved_path: savedPath,
              stage: "done",
              message:
                get().locale === "es"
                  ? `Descargado: ${filename}`
                  : `Downloaded: ${filename}`,
              updatedAt: Date.now(),
            },
          },
        };
      });
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("loors:models-changed"));
      }
      if (!get().activeModelPath) {
        get().setActiveModelPath(savedPath);
      }
      return savedPath;
    } catch (err) {
      const msg = typeof err === "string" ? err : String(err);
      const isCancelled = msg.toLowerCase().includes("cancel");
      set((s) => {
        const cur = s.modelDownloads[key];
        return {
          modelDownloads: {
            ...s.modelDownloads,
            [key]: {
              ...(cur ?? {
                key,
                repo_id: repoId,
                filename,
                downloaded_bytes: 0,
                total_bytes: totalBytesHint ?? null,
                percent: null,
              }),
              speed_bps: null,
              stage: isCancelled ? "cancelled" : "error",
              message: msg,
              updatedAt: Date.now(),
            },
          },
        };
      });
      return null;
    }
  },
  cancelModelDownload: async (repoId, filename) => {
    try {
      await invoke<boolean>("cancel_model_download", { repoId, filename });
    } catch {
      // ignore
    }
  },
  dismissModelDownload: (key) => {
    set((s) => {
      const next = { ...s.modelDownloads };
      delete next[key];
      return { modelDownloads: next };
    });
  },
  clearFinishedDownloads: () => {
    set((s) => {
      const next: Record<string, ModelDownloadItem> = {};
      for (const [k, v] of Object.entries(s.modelDownloads)) {
        if (v.stage === "starting" || v.stage === "downloading") {
          next[k] = v;
        }
      }
      return { modelDownloads: next };
    });
  },
  setActiveModelPath: (modelPath) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("loors_active_model_path", modelPath);
    }
    const shortName = modelPath.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "";
    set({
      activeModelPath: modelPath,
      nodes: get().nodes.map((n) =>
        n.data.kind === "local_model" && modelPath
          ? {
              ...n,
              data: {
                ...n.data,
                title: shortName || n.data.title,
                config: { ...n.data.config, modelPath },
              },
            }
          : n
      ),
    });
  },
  setRightPanelTab: (rightPanelTab) => set({ rightPanelTab }),
  setDimResourceEdges: (dimResourceEdges) => set({ dimResourceEdges }),
  openTuningModal: (tuningModalModelPath) => set({ tuningModalModelPath }),
  setEditingCustomNode: (editingCustomNode) => set({ editingCustomNode }),
  setArchitectSeedPrompt: (architectSeedPrompt) => set({ architectSeedPrompt }),
  setAgentStatus: (agentRunning, agentActiveTitle = null) =>
    set({ agentRunning, agentActiveTitle }),
  syncAgentContextToHarness: (workspace, modelPath) =>
    set({
      nodes: get().nodes.map((n) => {
        if (n.data.kind === "task_input" && workspace && workspace !== ".") {
          return { ...n, data: { ...n.data, config: { ...n.data.config, workspace } } };
        }
        if (n.data.kind === "workspace" && workspace && workspace !== ".") {
          return { ...n, data: { ...n.data, config: { ...n.data.config, path: workspace } } };
        }
        if (n.data.kind === "local_model" && modelPath) {
          const shortName = modelPath.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "";
          return {
            ...n,
            data: {
              ...n.data,
              title: shortName || n.data.title,
              config: { ...n.data.config, modelPath },
            },
          };
        }
        return n;
      }),
    }),

  setCustomNodes: (customNodes) => set({ customNodes }),

  duplicateNode: (id) => {
    const src = get().nodes.find((n) => n.id === id);
    if (!src) return;
    const copyId = `${src.data.kind}-${Date.now().toString(36)}`;
    const copy: LoorsFlowNode = {
      ...src,
      id: copyId,
      position: { x: src.position.x + 48, y: src.position.y + 48 },
      selected: false,
      data: {
        ...src.data,
        title: `${src.data.title} (copia)`,
        status: "idle",
        statusMessage: null,
        config: { ...src.data.config },
      },
    };
    editGraph({
      nodes: [...get().nodes, copy],
      selectedNodeId: copyId,
    });
  },

  addCustomNodeToCanvas: (def) => {
    const count = get().nodes.length;
    const id = `custom-${def.id}-${Date.now().toString(36)}`;
    const fieldDefaults: Record<string, unknown> = {};
    for (const f of def.fields) {
      fieldDefaults[f.key] = f.fieldType === "number" ? Number(f.defaultValue || 0) : f.defaultValue;
    }
    const newNode: LoorsFlowNode = {
      id,
      type: "loorsNode",
      position: {
        x: 260 + (count % 4) * 50,
        y: 140 + (count % 4) * 45,
      },
      data: {
        kind: "custom",
        title: def.label,
        status: "idle",
        config: {
          customNodeId: def.id,
          accent: def.accent,
          runner: def.runner,
          commandTemplate: def.commandTemplate,
          promptTemplate: def.promptTemplate,
          httpMethod: def.httpMethod,
          httpUrl: def.httpUrl,
          httpHeaders: def.httpHeaders,
          httpBody: def.httpBody,
          branchOnExit: def.branchOnExit,
          customFields: def.fields,
          ...fieldDefaults,
        },
      },
    };
    editGraph({
      nodes: [...get().nodes, newNode],
      selectedNodeId: id,
    });
  },

  attachResourceToNode: (targetNodeId, resourceKind) => {
    const state = get();
    const target = state.nodes.find((n) => n.id === targetNodeId);
    if (!target) return;

    const handleMap: Record<
      "local_model" | "skill" | "mcp_server" | "knowledge_rag" | "workspace",
      { src: string; tgt: string }
    > = {
      local_model: { src: "model_ref", tgt: "model_ref" },
      skill: { src: "skill_ref", tgt: "skill_ref" },
      mcp_server: { src: "mcp_ref", tgt: "mcp_ref" },
      knowledge_rag: { src: "knowledge_ref", tgt: "knowledge_ref" },
      workspace: { src: "workspace_ref", tgt: "workspace_ref" },
    };
    const handles = handleMap[resourceKind];
    const existingAttached = state.edges.filter(
      (e) => e.target === targetNodeId && e.targetHandle?.endsWith("_ref")
    ).length;

    const bp = NODE_BLUEPRINTS[resourceKind];
    const id = `${resourceKind}-${Date.now().toString(36)}`;
    const config = { ...bp.defaultConfig };
    if (resourceKind === "local_model" && state.activeModelPath) {
      config.modelPath = state.activeModelPath;
    }

    const newNode: LoorsFlowNode = {
      id,
      type: "loorsNode",
      position: {
        x: Math.max(20, target.position.x - 270),
        y: target.position.y + existingAttached * 125,
      },
      data: {
        kind: resourceKind,
        title: bp.label,
        status: "idle",
        config,
      },
    };
    const newEdge: LoorsFlowEdge = {
      id: `e-${id}-${targetNodeId}`,
      source: id,
      sourceHandle: handles.src,
      target: targetNodeId,
      targetHandle: handles.tgt,
      type: "smoothstep",
    };

    editGraph({
      nodes: [...state.nodes, newNode],
      edges: [...state.edges, newEdge],
      selectedNodeId: id,
    });
  },

  setWorkflowName: (workflowName) => {
    const s = get();
    set({
      workflowName,
      dirty: true,
      harnessTabs: s.harnessTabs.map((t) =>
        t.id === s.activeHarnessTabId ? { ...t, workflowName, dirty: true } : t
      ),
    });
  },
  setSelectedNodeId: (selectedNodeId) => set({ selectedNodeId }),
  setSelectedBackend: (selectedBackend) => set({ selectedBackend }),
  setConsoleOpen: (consoleOpen) => set({ consoleOpen }),
  clearLogs: () => set({ logs: [], runSummary: null, swarmAgents: {}, swarmMessages: [] }),
  clearPendingGate: () => set({ pendingGate: null }),

  newWorkflow: () => {
    const prevNodes = get().nodes;
    const storeActiveModel = get().activeModelPath.trim();
    const prevNodeModel = String(
      prevNodes.find((n) => n.data.kind === "local_model")?.data.config.modelPath ?? ""
    ).trim();
    const effectiveModel =
      storeActiveModel ||
      (prevNodeModel && prevNodeModel !== "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        ? prevNodeModel
        : "");
    const shortModelTitle =
      effectiveModel.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "Local Model (GGUF)";
    const prevWorkspace = String(
      prevNodes.find((n) => n.data.kind === "task_input")?.data.config.workspace ?? "."
    );

    const baseNodes: LoorsFlowNode[] = [
      {
        id: "in-1",
        type: "loorsNode",
        position: { x: 40, y: 160 },
        data: {
          kind: "task_input",
          title: "Task Input",
          status: "idle",
          config: {
            workspace: prevWorkspace,
            prompt: "",
          },
        },
      },
      {
        id: "model-1",
        type: "loorsNode",
        position: { x: 40, y: 360 },
        data: {
          kind: "local_model",
          title: shortModelTitle,
          status: "idle",
          config: {
            ...NODE_BLUEPRINTS.local_model.defaultConfig,
            modelPath: effectiveModel,
            maxTokens: 0,
          },
        },
      },
      {
        id: "agent-1",
        type: "loorsNode",
        position: { x: 360, y: 160 },
        data: {
          kind: "agent",
          title: "Agent",
          status: "idle",
          config: {
            ...NODE_BLUEPRINTS.agent.defaultConfig,
            maxSteps: 0,
          },
        },
      },
      {
        id: "out-1",
        type: "loorsNode",
        position: { x: 680, y: 160 },
        data: {
          kind: "output",
          title: "Output",
          status: "idle",
          config: { label: "Completado" },
        },
      },
    ];

    const baseEdges: LoorsFlowEdge[] = [
      { id: "e-in-agent", source: "in-1", sourceHandle: "exec", target: "agent-1", targetHandle: "exec" },
      {
        id: "e-model-agent",
        source: "model-1",
        sourceHandle: "model_ref",
        target: "agent-1",
        targetHandle: "model_ref",
        style: { stroke: "#a855f7", strokeDasharray: "5 4" },
      },
      { id: "e-agent-out", source: "agent-1", sourceHandle: "exec", target: "out-1", targetHandle: "exec" },
    ];

    graphHistories.delete(get().activeHarnessTabId);
    set({
      workflowName: "nuevo-arnes",
      nodes: baseNodes,
      edges: baseEdges,
      selectedNodeId: null,
      runStatus: "idle",
      runSummary: null,
      pendingGate: null,
      harnessTabs: get().harnessTabs.map((t) =>
        t.id === get().activeHarnessTabId
          ? { ...t, workflowName: "nuevo-arnes", runStatus: "idle", pendingGate: null }
          : t
      ),
    });
  },

  loadTemplate: (templateId, useDemo = false) => {
    const tpl = HARNESS_TEMPLATES.find((t) => t.id === templateId);
    if (!tpl) return;
    const prevNodes = get().nodes;
    const storeActiveModel = get().activeModelPath.trim();
    const prevNodeModel = String(
      prevNodes.find((n) => n.data.kind === "local_model")?.data.config.modelPath ?? ""
    ).trim();
    const effectivePrevModel =
      storeActiveModel ||
      (prevNodeModel && prevNodeModel !== "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        ? prevNodeModel
        : "");
    const shortModelTitle =
      effectivePrevModel.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "";
    const prevWorkspace = String(
      prevNodes.find((n) => n.data.kind === "task_input")?.data.config.workspace ?? ""
    );
    const hasRealPrevWorkspace =
      prevWorkspace.trim().length > 0 && prevWorkspace.trim() !== ".";

    const graph = useDemo && tpl.demo ? tpl.demo : tpl;
    get().createHarnessTab();
    set({
      workflowName: useDemo && tpl.demo ? `${tpl.id}-demo` : tpl.id,
      dirty: true,
      nodes: structuredClone(graph.nodes).map((n) => {
        const nextConfig = { ...n.data.config };
        let nextTitle = n.data.title;
        if (n.data.kind === "local_model" && effectivePrevModel) {
          nextConfig.modelPath = effectivePrevModel;
          if (shortModelTitle) nextTitle = shortModelTitle;
        }
        if (n.data.kind === "task_input" && hasRealPrevWorkspace) {
          nextConfig.workspace = prevWorkspace;
        }
        return {
          ...n,
          data: { ...n.data, title: nextTitle, config: nextConfig, status: "idle" },
        };
      }),
      edges: structuredClone(graph.edges),
    });
    set((state) => ({
      harnessTabs: state.harnessTabs.map((tab) =>
        tab.id === state.activeHarnessTabId ? snapshotActiveTab(state) : tab
      ),
    }));
  },

  prepareRun: () => {
    const s = get();
    const resetNodes = s.nodes.map((n) => ({
      ...n,
      data: { ...n.data, status: "idle" as const, statusMessage: null },
    }));
    set({
      runStatus: "running",
      currentIteration: 1,
      maxIterations: null,
      runSummary: null,
      activeActivity: "Iniciando arnés...",
      logs: [],
      consoleOpen: true,
      pendingGate: null,
      nodes: resetNodes,
      harnessTabs: s.harnessTabs.map((t) =>
        t.id === s.activeHarnessTabId
          ? {
              ...snapshotActiveTab(s),
              runStatus: "running",
              currentIteration: 1,
              maxIterations: null,
              runSummary: null,
              activeActivity: "Iniciando arnés...",
              logs: [],
              pendingGate: null,
              nodes: resetNodes,
            }
          : t
      ),
    });
  },

  applyEngineEvent: (event) => {
    if (event.type === "model_state") {
      set({
        loadedModelName: event.loaded ? event.model_name : null,
        loadedContextSize: event.loaded ? event.context_size : null,
      });
      return;
    }

    if (event.type === "swarm_agent_status") {
      const targetTabId = event.tabId || get().activeHarnessTabId;
      if (targetTabId !== get().activeHarnessTabId) return;
      const k = `${event.node_id}::${event.agent_id}`;
      set((st) => ({
        swarmAgents: {
          ...st.swarmAgents,
          [k]: {
            nodeId: event.node_id,
            agentId: event.agent_id,
            agentName: event.agent_name,
            role: event.role,
            status: event.status,
            summary: event.summary,
          },
        },
      }));
      return;
    }

    if (event.type === "swarm_board_message") {
      const targetTabId = event.tabId || get().activeHarnessTabId;
      if (targetTabId !== get().activeHarnessTabId) return;
      set((st) => ({
        swarmMessages: [
          ...st.swarmMessages.slice(-300),
          {
            id: `${event.run_id}-${event.node_id}-${event.seq}`,
            nodeId: event.node_id,
            seq: event.seq,
            from: event.from,
            mentions: event.mentions,
            content: event.content,
            timestamp: event.timestamp,
          },
        ],
      }));
      return;
    }

    const s = get();
    const targetTabId = event.tabId || s.activeHarnessTabId;

    if (targetTabId === s.activeHarnessTabId) {
      const updated = applyEventToTabState(snapshotActiveTab(s), event);
      set({
        ...(event.type === "run_started" ? { swarmAgents: {}, swarmMessages: [] } : {}),
        runStatus: updated.runStatus,
        activeRunId: updated.activeRunId,
        nodeResults: updated.nodeResults,
        currentIteration: updated.currentIteration,
        maxIterations: updated.maxIterations,
        runSummary: updated.runSummary,
        activeActivity: updated.activeActivity,
        nodes: updated.nodes,
        logs: updated.logs,
        pendingGate: updated.pendingGate,
        harnessTabs: s.harnessTabs.map((t) => (t.id === targetTabId ? updated : t)),
      });
    } else {
      // Background harness tab event: update that tab cleanly without disturbing the active canvas
      set({
        harnessTabs: s.harnessTabs.map((t) =>
          t.id === targetTabId ? applyEventToTabState(t, event) : t
        ),
      });
    }
  },

  onNodesChange: (changes) => {
    const state = get();
    const nodes = applyNodeChanges(changes, state.nodes);
    const positions = changes.filter((change) => change.type === "position");
    const moved = positions.some((change) => {
      const previous = state.nodes.find((node) => node.id === change.id)?.position;
      return previous && change.position
        && (previous.x !== change.position.x || previous.y !== change.position.y);
    });
    const history = getGraphHistory(state.activeHarnessTabId);
    if (moved && positions.some((change) => change.dragging === true)) {
      history.dragStart ??= graphSnapshot(state);
    }

    if (history.dragStart && positions.some((change) => change.dragging === false)) {
      const before = history.dragStart;
      delete history.dragStart;
      const originalPositions = new Map(before.nodes.map((node) => [node.id, node.position]));
      if (nodes.some((node) => {
        const original = originalPositions.get(node.id);
        return original && (original.x !== node.position.x || original.y !== node.position.y);
      })) {
        editGraph({ nodes }, before);
        return;
      }
    } else if (changes.some((change) => change.type === "add" || change.type === "remove" || change.type === "replace")
      || (moved && !history.dragStart)) {
      editGraph({ nodes });
      return;
    }
    set({ nodes });
  },

  onEdgesChange: (changes) => {
    const edges = applyEdgeChanges(changes, get().edges);
    if (changes.some((change) => change.type !== "select")) {
      editGraph({ edges });
      return;
    }
    set({ edges });
  },

  onConnect: (connection) => {
    const isRetry =
      connection.sourceHandle === "retry" || connection.targetHandle === "feedback";
    const isModel =
      connection.sourceHandle === "model_ref" || connection.targetHandle === "model_ref";
    const isSkill =
      connection.sourceHandle === "skill_ref" || connection.targetHandle === "skill_ref";
    const isMcp =
      connection.sourceHandle === "mcp_ref" || connection.targetHandle === "mcp_ref";
    const isWorkspace =
      connection.sourceHandle === "workspace_ref" || connection.targetHandle === "workspace_ref";
    const isKnowledge =
      connection.sourceHandle === "knowledge_ref" || connection.targetHandle === "knowledge_ref";

    const nextEdge: LoorsFlowEdge = {
      ...connection,
      id: `e-${connection.source}-${connection.sourceHandle ?? "out"}-${connection.target}-${connection.targetHandle ?? "in"}-${Date.now()}`,
      animated: isRetry,
      style: isRetry
        ? { stroke: "#22c55e", strokeWidth: 2 }
        : isModel
          ? { stroke: "#a855f7", strokeDasharray: "5 4" }
          : isSkill
            ? { stroke: "#2dd4bf", strokeDasharray: "5 4" }
            : isMcp
              ? { stroke: "#fb923c", strokeDasharray: "5 4" }
              : isWorkspace
                ? { stroke: "#0ea5e9", strokeDasharray: "5 4" }
                : isKnowledge
                  ? { stroke: "#06b6d4", strokeDasharray: "5 4" }
                  : undefined,
    };

    const currentEdges = get().edges;
    const edges = addEdge(nextEdge, currentEdges);
    if (edges !== currentEdges) editGraph({ edges });
  },

  addNode: (kind, position) => {
    const bp = NODE_BLUEPRINTS[kind];
    const count = get().nodes.length;
    const id = `${kind}-${Date.now().toString(36)}`;
    const newNode: LoorsFlowNode = {
      id,
      type: "loorsNode",
      position: position ?? {
        x: 240 + (count % 4) * 50,
        y: 120 + (count % 4) * 45,
      },
      data: {
        kind,
        title: bp.label,
        status: "idle",
        config: { ...bp.defaultConfig },
      },
    };
    editGraph({
      nodes: [...get().nodes, newNode],
      selectedNodeId: id,
    });
  },

  autoLayout: () => {
    const { nodes, edges } = get();
    const resourceKinds = new Set([
      "local_model",
      "skill",
      "mcp_server",
      "workspace",
      "knowledge_rag",
    ]);
    const execNodes = nodes.filter((n) => !resourceKinds.has(n.data.kind));
    const resNodes = nodes.filter((n) => resourceKinds.has(n.data.kind));

    // BFS column assignment from task_input along non-retry edges
    const colMap = new Map<string, number>();
    const start =
      execNodes.find((n) => n.data.kind === "trigger") ??
      execNodes.find((n) => n.data.kind === "task_input") ??
      execNodes[0];
    if (start) {
      const queue: string[] = [start.id];
      colMap.set(start.id, 0);
      while (queue.length > 0) {
        const curr = queue.shift()!;
        const currCol = colMap.get(curr) ?? 0;
        for (const e of edges) {
          if (e.source === curr && e.sourceHandle !== "retry" && !colMap.has(e.target)) {
            if (!resourceKinds.has(nodes.find((n) => n.id === e.target)?.data.kind ?? "")) {
              colMap.set(e.target, currCol + 1);
              queue.push(e.target);
            }
          }
        }
      }
    }

    const colCounts = new Map<number, number>();
    const positioned = nodes.map((n) => {
      if (resourceKinds.has(n.data.kind)) {
        const idx = resNodes.findIndex((r) => r.id === n.id);
        return { ...n, position: { x: 40, y: 360 + idx * 170 } };
      }
      const col = colMap.get(n.id) ?? 1;
      const row = colCounts.get(col) ?? 0;
      colCounts.set(col, row + 1);
      return {
        ...n,
        position: { x: 40 + col * 290, y: 120 + row * 210 },
      };
    });

    if (positioned.some((node, index) =>
      node.position.x !== nodes[index].position.x || node.position.y !== nodes[index].position.y
    )) {
      editGraph({ nodes: positioned });
    }
  },

  updateNodeTitle: (id, title) => {
    const { nodes } = get();
    const target = nodes.find((node) => node.id === id);
    if (!target || target.data.title === title) return;
    editGraph({
      nodes: nodes.map((node) => node === target ? { ...node, data: { ...node.data, title } } : node),
    });
  },

  updateNodeConfig: (id, patch) => {
    const { nodes } = get();
    const target = nodes.find((node) => node.id === id);
    if (!target || Object.entries(patch).every(([key, value]) => Object.is(target.data.config[key], value))) return;
    editGraph({
      nodes: nodes.map((node) =>
        node === target
          ? {
              ...node,
              data: {
                ...node.data,
                config: { ...node.data.config, ...patch },
              },
            }
          : node
      ),
    });
  },

  deleteNode: (id) => get().deleteGraphElements([id], []),

  deleteGraphElements: (nodeIds, edgeIds) => {
    const state = get();
    const removedNodes = new Set(nodeIds);
    const removedEdges = new Set(edgeIds);
    const nodes = state.nodes.filter((node) => !removedNodes.has(node.id));
    const edges = state.edges.filter((edge) =>
      !removedEdges.has(edge.id) && !removedNodes.has(edge.source) && !removedNodes.has(edge.target)
    );
    if (nodes.length === state.nodes.length && edges.length === state.edges.length) return;
    editGraph({
      nodes,
      edges,
      selectedNodeId: state.selectedNodeId && removedNodes.has(state.selectedNodeId)
        ? null
        : state.selectedNodeId,
    });
  },

  undoGraphEdit: () => restoreGraphEdit("undo"),
  redoGraphEdit: () => restoreGraphEdit("redo"),

  exportWorkflow: () => ({
    schemaVersion: 1,
    name: get().workflowName,
    nodes: get().nodes,
    edges: get().edges,
  }),

  importWorkflow: (file, libraryName, opts) => {
    graphHistories.delete(get().activeHarnessTabId);
    const activeModel = get().activeModelPath.trim();
    const shortModelTitle =
      activeModel.split(/[\\/]/).pop()?.replace(/\.gguf$/i, "") || "";
    const resolvedName = file.name || "imported-workflow";
    const resolvedLib = opts?.preserveLibraryLink
      ? (libraryName ?? get().savedLibraryName ?? null)
      : (libraryName ?? null);
    const nextNodes = (file.nodes || []).map((n) => {
      const explicitModelPath = String(n.data?.config?.modelPath ?? "").trim();
      const shouldOverrideModel =
        !opts?.preserveModels &&
        n.data.kind === "local_model" &&
        activeModel &&
        (!explicitModelPath || explicitModelPath === "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf");
      if (shouldOverrideModel) {
        return {
          ...n,
          type: "loorsNode" as const,
          data: {
            ...n.data,
            title: shortModelTitle || n.data.title,
            config: { ...n.data.config, modelPath: activeModel },
            status: "idle" as const,
          },
        };
      }
      return {
        ...n,
        type: "loorsNode" as const,
        data: { ...n.data, status: "idle" as const },
      };
    });
    const prevSelected = get().selectedNodeId;
    const nextSelected =
      opts?.preserveSelection && prevSelected && nextNodes.some((n) => n.id === prevSelected)
        ? prevSelected
        : null;
    set({
      workflowName: resolvedName,
      savedLibraryName: resolvedLib,
      dirty: opts?.dirty ?? false,
      nodes: nextNodes,
      edges: file.edges || [],
      selectedNodeId: nextSelected,
      runStatus: "idle",
      activeRunId: null,
      nodeResults: {},
      runSummary: null,
      pendingGate: null,
      harnessTabs: get().harnessTabs.map((t) =>
        t.id === get().activeHarnessTabId
          ? {
              ...t,
              workflowName: resolvedName,
              savedLibraryName: resolvedLib,
              dirty: opts?.dirty ?? false,
              runStatus: "idle",
              activeRunId: null,
              nodeResults: {},
              pendingGate: null,
            }
          : t
      ),
    });
  },
}));

// Persist open Harness tabs to localStorage (debounced) so restarting Loors restores all open tabs
if (typeof window !== "undefined") {
  let persistTimer: number | null = null;
  useGraphStore.subscribe((state) => {
    if (persistTimer) window.clearTimeout(persistTimer);
    persistTimer = window.setTimeout(() => {
      try {
        const activeSnap = snapshotActiveTab(state);
        const allTabs: PersistedHarnessTab[] = state.harnessTabs.map((t) => {
          const src = t.id === state.activeHarnessTabId ? activeSnap : t;
          return {
            id: src.id,
            workflowName: src.workflowName,
            savedLibraryName: src.savedLibraryName,
            dirty: src.dirty,
            nodes: src.nodes,
            edges: src.edges,
          };
        });
        window.localStorage.setItem(
          TABS_STORAGE_KEY,
          JSON.stringify({ activeId: state.activeHarnessTabId, tabs: allTabs })
        );
      } catch {
        // ignore quota errors
      }
    }, 400);
  });
}
