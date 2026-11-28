use crate::engine::types::{
    EngineEvent, ExecutionContext, GraphEdge, GraphNode, HumanDecision, LogStreamKind,
    NodeStatusKind, RunOutcomeKind, WorkflowGraph,
};
use crate::hardware::RuntimeBackendKind;
use crate::inference::llama_cpp::LlamaCppManager;
use crate::inference::{
    resolve_image_path, resolve_load_config_with_profile, ChatMessage, ChatRequest, ToolCall,
    ToolCallFunction,
};
use crate::knowledge::{
    build_knowledge_tree, build_or_update_index, query_knowledge, search_bm25, KnowledgeIndex,
};
use crate::mcp::{mcp_prefixed_tool_name, McpServerConfig, McpSession};
use crate::runtime::find_installed_binary;
use crate::skills::{load_skill_file, LoadedSkill};
use crate::tools::{execute_tool_cancellable, tool_definitions};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::oneshot;

const MAX_SAFETY_STEPS: u32 = 50_000;

pub type GateSlot = Arc<Mutex<Option<oneshot::Sender<HumanDecision>>>>;
pub type EventEmitter = Arc<dyn Fn(EngineEvent) + Send + Sync>;

static LIVE_INSTRUCTIONS: std::sync::OnceLock<Mutex<Vec<String>>> = std::sync::OnceLock::new();

fn live_instructions_queue() -> &'static Mutex<Vec<String>> {
    LIVE_INSTRUCTIONS.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn push_live_instruction(text: String) {
    if let Ok(mut q) = live_instructions_queue().lock() {
        q.push(text);
    }
}

fn drain_live_instructions() -> Vec<String> {
    live_instructions_queue()
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default()
}

struct ActiveRun {
    cancel: Arc<AtomicBool>,
    pending_gate: GateSlot,
}

pub struct EngineManager {
    runs: Arc<Mutex<HashMap<String, ActiveRun>>>,
    pub llama: LlamaCppManager,
}

impl Default for EngineManager {
    fn default() -> Self {
        Self {
            runs: Arc::new(Mutex::new(HashMap::new())),
            llama: LlamaCppManager::new(),
        }
    }
}

#[derive(Clone, Default)]
struct LastRunSnapshot {
    node_outputs: HashMap<String, String>,
    last_agent_output: String,
    last_stdout: String,
    last_stderr: String,
    last_custom_output: String,
    last_http_response: String,
    last_memory: String,
    last_rag_context: String,
    last_git_diff: String,
    last_judge_reason: String,
    last_exit_code: Option<i32>,
    current_item: String,
    current_item_index: usize,
    current_item_total: usize,
}

impl LastRunSnapshot {
    fn from_ctx(ctx: &ExecutionContext) -> Self {
        Self {
            node_outputs: ctx.node_outputs.clone(),
            last_agent_output: ctx.last_agent_output.clone(),
            last_stdout: ctx.last_stdout.clone(),
            last_stderr: ctx.last_stderr.clone(),
            last_custom_output: ctx.last_custom_output.clone(),
            last_http_response: ctx.last_http_response.clone(),
            last_memory: ctx.last_memory.clone(),
            last_rag_context: ctx.last_rag_context.clone(),
            last_git_diff: ctx.last_git_diff.clone(),
            last_judge_reason: ctx.last_judge_reason.clone(),
            last_exit_code: ctx.last_exit_code,
            current_item: ctx.current_item.clone(),
            current_item_index: ctx.current_item_index,
            current_item_total: ctx.current_item_total,
        }
    }

    fn hydrate_into(&self, ctx: &mut ExecutionContext) {
        ctx.node_outputs = self.node_outputs.clone();
        ctx.last_agent_output = self.last_agent_output.clone();
        ctx.last_stdout = self.last_stdout.clone();
        ctx.last_stderr = self.last_stderr.clone();
        ctx.last_custom_output = self.last_custom_output.clone();
        ctx.last_http_response = self.last_http_response.clone();
        ctx.last_memory = self.last_memory.clone();
        ctx.last_rag_context = self.last_rag_context.clone();
        ctx.last_git_diff = self.last_git_diff.clone();
        ctx.last_judge_reason = self.last_judge_reason.clone();
        ctx.last_exit_code = self.last_exit_code;
        ctx.current_item = self.current_item.clone();
        ctx.current_item_index = self.current_item_index;
        ctx.current_item_total = self.current_item_total;
    }
}

static LAST_RUN_SNAPSHOT: OnceLock<Mutex<LastRunSnapshot>> = OnceLock::new();

fn last_run_snapshot_cache() -> &'static Mutex<LastRunSnapshot> {
    LAST_RUN_SNAPSHOT.get_or_init(|| Mutex::new(LastRunSnapshot::default()))
}

fn format_duration_ms(ms: u128) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        let secs = ms / 1000;
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

impl EngineManager {
    pub fn is_running(&self) -> bool {
        self.runs
            .lock()
            .map(|m| !m.is_empty())
            .unwrap_or(false)
    }

    pub fn start_run<F>(
        &self,
        graph: WorkflowGraph,
        backend: RuntimeBackendKind,
        emit: F,
    ) -> Result<String, String>
    where
        F: Fn(EngineEvent) + Send + Sync + 'static,
    {
        self.start_run_in_tab(None, graph, backend, None, emit)
    }

    pub fn start_run_from<F>(
        &self,
        graph: WorkflowGraph,
        backend: RuntimeBackendKind,
        start_node_id: Option<String>,
        emit: F,
    ) -> Result<String, String>
    where
        F: Fn(EngineEvent) + Send + Sync + 'static,
    {
        self.start_run_in_tab(None, graph, backend, start_node_id, emit)
    }

    pub fn start_run_in_tab<F>(
        &self,
        tab_id: Option<String>,
        graph: WorkflowGraph,
        backend: RuntimeBackendKind,
        start_node_id: Option<String>,
        emit: F,
    ) -> Result<String, String>
    where
        F: Fn(EngineEvent) + Send + Sync + 'static,
    {
        let tab_key = tab_id
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "default".to_string());

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let gate_slot: GateSlot = Arc::new(Mutex::new(None));

        {
            let mut guard = self
                .runs
                .lock()
                .map_err(|_| "Engine lock poisoned".to_string())?;
            // Cancel only a previous run on this exact tab, leaving all other tabs running in parallel
            if let Some(prev) = guard.remove(&tab_key) {
                prev.cancel.store(true, Ordering::SeqCst);
                if let Ok(mut g) = prev.pending_gate.lock() {
                    g.take();
                }
            }
            guard.insert(
                tab_key.clone(),
                ActiveRun {
                    cancel: cancel_flag.clone(),
                    pending_gate: gate_slot.clone(),
                },
            );
        }

        let _ = drain_live_instructions();

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let run_id = format!("run-{now_ms}");
        let run_id_clone = run_id.clone();
        let llama = self.llama.clone();
        let runs_map = self.runs.clone();
        let cancel_for_cleanup = cancel_flag.clone();

        // Prepare live log file in %LOCALAPPDATA%/loors/logs/latest.log
        let log_file = open_run_log_file(&run_id);

        tauri::async_runtime::spawn(async move {
            struct CleanupRun {
                runs: Arc<Mutex<HashMap<String, ActiveRun>>>,
                tab_key: String,
                cancel: Arc<AtomicBool>,
            }
            impl Drop for CleanupRun {
                fn drop(&mut self) {
                    if let Ok(mut m) = self.runs.lock() {
                        let is_same = m
                            .get(&self.tab_key)
                            .map(|r| Arc::ptr_eq(&r.cancel, &self.cancel))
                            .unwrap_or(false);
                        if is_same {
                            m.remove(&self.tab_key);
                        }
                    }
                }
            }
            let _guard = CleanupRun {
                runs: runs_map,
                tab_key,
                cancel: cancel_for_cleanup,
            };
            let log_file_arc = Arc::new(Mutex::new(log_file));
            let emitter: EventEmitter = Arc::new(move |ev| {
                write_event_to_log(&log_file_arc, &ev);
                emit(ev);
            });
            execute_graph_with_depth(
                run_id_clone,
                graph,
                backend,
                llama,
                gate_slot,
                cancel_flag,
                emitter,
                0,
                start_node_id,
            )
            .await;
        });

        Ok(run_id)
    }

    pub fn stop_run(&self) -> Result<(), String> {
        self.stop_run_for(None)
    }

    pub fn stop_run_for(&self, tab_id: Option<&str>) -> Result<(), String> {
        let mut guard = self
            .runs
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        if let Some(tid) = tab_id.map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(run) = guard.remove(tid) {
                run.cancel.store(true, Ordering::SeqCst);
                if let Ok(mut gate) = run.pending_gate.lock() {
                    gate.take();
                }
            }
        } else {
            for (_, run) in guard.drain() {
                run.cancel.store(true, Ordering::SeqCst);
                if let Ok(mut gate) = run.pending_gate.lock() {
                    gate.take();
                }
            }
        }
        let _ = drain_live_instructions();
        Ok(())
    }

    pub fn resolve_human_gate(&self, decision: HumanDecision) -> Result<(), String> {
        self.resolve_human_gate_for(None, decision)
    }

    pub fn resolve_human_gate_for(
        &self,
        tab_id: Option<&str>,
        decision: HumanDecision,
    ) -> Result<(), String> {
        let guard = self
            .runs
            .lock()
            .map_err(|_| "Engine lock poisoned".to_string())?;
        if let Some(tid) = tab_id.map(str::trim).filter(|s| !s.is_empty()) {
            let Some(run) = guard.get(tid) else {
                return Err("No hay ningún Human Gate esperando respuesta en esta pestaña.".to_string());
            };
            let mut gate = run
                .pending_gate
                .lock()
                .map_err(|_| "Gate lock poisoned".to_string())?;
            let Some(tx) = gate.take() else {
                return Err("No hay ningún Human Gate esperando respuesta.".to_string());
            };
            return tx
                .send(decision)
                .map_err(|_| "El Human Gate ya no estaba activo.".to_string());
        }
        for run in guard.values() {
            if let Ok(mut gate) = run.pending_gate.lock() {
                if let Some(tx) = gate.take() {
                    return tx
                        .send(decision)
                        .map_err(|_| "El Human Gate ya no estaba activo.".to_string());
                }
            }
        }
        Err("No hay ningún Human Gate esperando respuesta.".to_string())
    }
}

#[allow(dead_code)]
pub async fn execute_graph(
    run_id: String,
    graph: WorkflowGraph,
    backend: RuntimeBackendKind,
    llama: LlamaCppManager,
    gate_slot: GateSlot,
    cancel: Arc<AtomicBool>,
    emit: EventEmitter,
) -> RunOutcomeKind {
    execute_graph_with_depth(run_id, graph, backend, llama, gate_slot, cancel, emit, 0, None).await
}

fn execute_graph_with_depth(
    run_id: String,
    graph: WorkflowGraph,
    backend: RuntimeBackendKind,
    llama: LlamaCppManager,
    gate_slot: GateSlot,
    cancel: Arc<AtomicBool>,
    emit: EventEmitter,
    depth: u8,
    start_node_id: Option<String>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = RunOutcomeKind> + Send>> {
    Box::pin(async move {
    emit(EngineEvent::RunStarted {
        run_id: run_id.clone(),
        workflow_name: graph.name.clone(),
    });

    let node_map: HashMap<String, &GraphNode> =
        graph.nodes.iter().map(|n| (n.id.clone(), n)).collect();
    if let Err(summary) = validate_graph_connections(&graph, &node_map) {
        emit(EngineEvent::RunFinished {
            run_id,
            outcome: RunOutcomeKind::Failed,
            total_iterations: 0,
            summary,
        });
        return RunOutcomeKind::Failed;
    }

    let requested_start = start_node_id
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .and_then(|id| node_map.get(id).copied());

    let start_node = requested_start
        .or_else(|| graph.nodes.iter().find(|n| n.data.kind == "trigger"))
        .or_else(|| graph.nodes.iter().find(|n| n.data.kind == "task_input"))
        .or_else(|| graph.nodes.first());

    let Some(start) = start_node else {
        emit(EngineEvent::RunFinished {
            run_id,
            outcome: RunOutcomeKind::Failed,
            total_iterations: 0,
            summary: "El grafo no contiene nodos.".to_string(),
        });
        return RunOutcomeKind::Failed;
    };

    let mut ctx = ExecutionContext {
        iteration: 1,
        ..Default::default()
    };

    // Read global run budget from task_input or trigger node
    if let Some(entry_node) = graph
        .nodes
        .iter()
        .find(|n| n.data.kind == "task_input" || n.data.kind == "trigger")
    {
        ctx.max_llm_calls = config_u64(entry_node, "maxLlmCalls", 0) as usize;
        let explicit_secs = config_u64(entry_node, "maxRunSeconds", 0);
        ctx.max_run_secs = if explicit_secs > 0 {
            explicit_secs
        } else {
            config_u64(entry_node, "maxRunMinutes", 0).saturating_mul(60)
        };
        if requested_start.is_some() {
            ctx.workspace = config_str(entry_node, "workspace", ".");
            let p = config_str(entry_node, "prompt", "");
            if !p.is_empty() {
                ctx.initial_prompt = p.clone();
                ctx.current_prompt = p;
            }
        }
    }

    // When resuming from a specific node, hydrate cached outputs and variables from the previous run
    if requested_start.is_some() {
        if let Ok(snap) = last_run_snapshot_cache().lock() {
            snap.hydrate_into(&mut ctx);
        }
    }

    let run_started = tokio::time::Instant::now();
    let run_deadline = (ctx.max_run_secs > 0)
        .then(|| run_started.checked_add(std::time::Duration::from_secs(ctx.max_run_secs)))
        .flatten();
    let mut current_id: Option<String> = Some(start.id.clone());
    let mut safety_steps: u32 = 0;

    while let Some(node_id) = current_id {
        if cancel.load(Ordering::SeqCst) {
            emit(EngineEvent::RunFinished {
                run_id,
                outcome: RunOutcomeKind::Cancelled,
                total_iterations: ctx.iteration,
                summary: "Ejecución detenida por el usuario.".to_string(),
            });
            return RunOutcomeKind::Cancelled;
        }

        if ctx.max_run_secs > 0 && run_started.elapsed().as_secs() >= ctx.max_run_secs {
            let mins = ctx.max_run_secs / 60;
            let msg = format!("Run budget timeout reached ({mins} min).");
            emit(EngineEvent::RunFinished {
                run_id,
                outcome: RunOutcomeKind::Exhausted,
                total_iterations: ctx.iteration,
                summary: msg,
            });
            return RunOutcomeKind::Exhausted;
        }

        safety_steps += 1;
        if safety_steps > MAX_SAFETY_STEPS {
            emit(EngineEvent::RunFinished {
                run_id,
                outcome: RunOutcomeKind::Failed,
                total_iterations: ctx.iteration,
                summary: format!(
                    "Se alcanzó el límite de seguridad ({MAX_SAFETY_STEPS} pasos). Usa un nodo Loop Guard para controlar ciclos."
                ),
            });
            return RunOutcomeKind::Failed;
        }

        let Some(node) = node_map.get(&node_id).copied() else {
            emit(EngineEvent::RunFinished {
                run_id,
                outcome: RunOutcomeKind::Failed,
                total_iterations: ctx.iteration,
                summary: format!("Nodo no encontrado: {node_id}"),
            });
            return RunOutcomeKind::Failed;
        };

        emit(EngineEvent::NodeStatus {
            run_id: run_id.clone(),
            node_id: node.id.clone(),
            status: NodeStatusKind::Running,
            iteration: ctx.iteration,
            message: None,
        });

        let step_iteration = ctx.iteration;
        let node_started = std::time::Instant::now();
        let step = run_node(
            &run_id,
            node,
            &graph,
            &node_map,
            &backend,
            &llama,
            &gate_slot,
            depth,
            &mut ctx,
            &cancel,
            &emit,
        );
        let step_result = if let Some(run_deadline) = run_deadline {
            match tokio::time::timeout_at(run_deadline, step).await {
                Ok(result) => result,
                Err(_) => {
                    if let Ok(mut gate) = gate_slot.lock() {
                        gate.take();
                    }
                    NodeStepOutcome::Finish(
                        RunOutcomeKind::Exhausted,
                        "Run budget timeout reached.".into(),
                    )
                }
            }
        } else {
            step.await
        };
        let elapsed_label = format_duration_ms(node_started.elapsed().as_millis());

        match step_result {
            NodeStepOutcome::Continue(out_port) => {
                record_node_output(&mut ctx, node);
                let value = match node.data.kind.as_str() {
                    "agent" => Some(("response", ctx.last_agent_output.as_str())),
                    "shell" => Some(("stderr", ctx.last_stderr.as_str())),
                    "data_transform" => ctx.node_outputs.get(&node.id).map(|value| ("exec", value.as_str())),
                    "create_excel" | "create_csv" | "create_word" | "create_pdf" | "read_document" => {
                        ctx.node_outputs.get(&node.id).map(|value| ("exec", value.as_str()))
                    }
                    _ => None,
                };
                const PREVIEW_CHARS: usize = 2000;
                let (value_preview, value_truncated) = value.map_or((None, false), |(_, text)| {
                    let mut chars = text.chars();
                    let preview = chars.by_ref().take(PREVIEW_CHARS).collect();
                    (Some(preview), chars.next().is_some())
                });
                emit(EngineEvent::NodeResult {
                    run_id: run_id.clone(),
                    node_id: node.id.clone(),
                    status: NodeStatusKind::Success,
                    iteration: step_iteration,
                    step: safety_steps,
                    out_port: Some(out_port.clone()),
                    value_port: value.map(|(port, _)| port.into()),
                    value_preview,
                    value_truncated,
                });
                if let Ok(mut snap) = last_run_snapshot_cache().lock() {
                    *snap = LastRunSnapshot::from_ctx(&ctx);
                }
                emit(EngineEvent::NodeStatus {
                    run_id: run_id.clone(),
                    node_id: node.id.clone(),
                    status: NodeStatusKind::Success,
                    iteration: ctx.iteration,
                    message: Some(elapsed_label),
                });
                current_id = find_next_node(&graph.edges, &node.id, &out_port);
            }
            NodeStepOutcome::Finish(outcome, summary) => {
                let node_status = match outcome {
                    RunOutcomeKind::Completed => NodeStatusKind::Success,
                    RunOutcomeKind::Exhausted => NodeStatusKind::Paused,
                    RunOutcomeKind::Failed | RunOutcomeKind::Cancelled => NodeStatusKind::Error,
                };
                emit(EngineEvent::NodeResult {
                    run_id: run_id.clone(),
                    node_id: node.id.clone(),
                    status: node_status.clone(),
                    iteration: step_iteration,
                    step: safety_steps,
                    out_port: None,
                    value_port: None,
                    value_preview: None,
                    value_truncated: false,
                });
                emit(EngineEvent::NodeStatus {
                    run_id: run_id.clone(),
                    node_id: node.id.clone(),
                    status: node_status,
                    iteration: ctx.iteration,
                    message: Some(summary.clone()),
                });
                emit(EngineEvent::RunFinished {
                    run_id,
                    outcome: outcome.clone(),
                    total_iterations: ctx.iteration,
                    summary,
                });
                return outcome;
            }
        }
    }

    emit(EngineEvent::RunFinished {
        run_id,
        outcome: RunOutcomeKind::Completed,
        total_iterations: ctx.iteration,
        summary: format!("Flujo finalizado en {} iteración(es).", ctx.iteration),
    });
    RunOutcomeKind::Completed
    })
}

enum NodeStepOutcome {
    Continue(String),
    Finish(RunOutcomeKind, String),
}

async fn run_node(
    run_id: &str,
    node: &GraphNode,
    graph: &WorkflowGraph,
    node_map: &HashMap<String, &GraphNode>,
    backend: &RuntimeBackendKind,
    llama: &LlamaCppManager,
    gate_slot: &GateSlot,
    depth: u8,
    ctx: &mut ExecutionContext,
    cancel: &Arc<AtomicBool>,
    emit: &EventEmitter,
) -> NodeStepOutcome {
    let log = |iteration: u32, stream: LogStreamKind, text: String| {
        emit(EngineEvent::NodeLog {
            run_id: run_id.to_string(),
            node_id: node.id.clone(),
            node_title: node.data.title.clone(),
            iteration,
            stream,
            text,
        });
    };

    match node.data.kind.as_str() {
        "trigger" => {
            let trigger_type = config_str(node, "triggerType", "webhook");
            let payload = config_str(node, "payloadTemplate", "");
            if !payload.trim().is_empty() && ctx.initial_prompt.is_empty() {
                ctx.initial_prompt = payload.clone();
                ctx.current_prompt = payload.clone();
            }
            log(
                ctx.iteration,
                LogStreamKind::System,
                format!("Trigger ({trigger_type}) disparado."),
            );
            NodeStepOutcome::Continue("exec".to_string())
        }

        "task_input" => {
            let workspace = config_str(node, "workspace", ".");
            let prompt = config_str(node, "prompt", "");
            ctx.workspace = workspace.clone();
            let merged_prompt = if !ctx.initial_prompt.is_empty() && ctx.initial_prompt != prompt {
                format!("{prompt}\n\n{}", ctx.initial_prompt)
            } else {
                prompt.clone()
            };
            if merged_prompt.trim().is_empty()
                && node
                    .data
                    .config
                    .get("requirePrompt")
                    .and_then(|value| value.as_bool())
                    == Some(true)
            {
                return NodeStepOutcome::Finish(
                    RunOutcomeKind::Failed,
                    "Introduce el encargo en el nodo de entrada.".to_string(),
                );
            }
            ctx.initial_prompt = merged_prompt.clone();
            ctx.current_prompt = merged_prompt;
            log(
                ctx.iteration,
                LogStreamKind::System,
                format!("Workspace: {workspace} | Prompt: {prompt}"),
            );
            NodeStepOutcome::Continue("exec".to_string())
        }

        "workspace" => {
            let raw_path = config_str(node, "path", "");
            let rendered = ctx.render_template(raw_path.trim());
            let iter = ctx.iteration;
            if rendered.trim().is_empty() {
                let msg = "Nodo Workspace sin carpeta configurada.".to_string();
                log(iter, LogStreamKind::Stderr, msg.clone());
                return NodeStepOutcome::Finish(RunOutcomeKind::Failed, msg);
            }
            ctx.workspace = rendered.trim().to_string();
            log(
                iter,
                LogStreamKind::System,
                format!("📁 Workspace activo cambiado a: {}", ctx.workspace),
            );
            NodeStepOutcome::Continue("exec".to_string())
        }

        "agent" => {
            let connected_model = find_connected_model_node(graph, node_map, &node.id);
            let installed_runtime = find_installed_binary(backend.clone());
            let iter = ctx.iteration;
            let node_ws = effective_workspace(graph, node_map, node, ctx);

            if let Some(model_node) = connected_model {
                let model_path = config_str(model_node, "modelPath", "");
                let is_cloud = crate::inference::cloud::is_cloud_model(&model_path);
                if is_cloud || (installed_runtime.is_some() && Path::new(&model_path).exists()) {
                    let raw_ctx = config_u64(model_node, "contextSize", 0) as u32;
                    let (load_cfg_opt, model_profile, context_size, has_mmproj, rt_version) =
                        if is_cloud {
                            let prof = crate::inference::ModelTuningProfile::default();
                            (None, prof, if raw_ctx > 0 { raw_ctx } else { 128_000 }, true, "Cloud API".to_string())
                        } else {
                            let rt = installed_runtime.unwrap();
                            let (cfg, prof) = resolve_load_config_with_profile(
                                rt.binary_path,
                                &model_path,
                                if raw_ctx > 0 { Some(raw_ctx) } else { None },
                                opt_config_str(model_node, "mmprojPath"),
                            );
                            let ctx_sz = cfg.context_size;
                            let mm = cfg.mmproj_path.is_some();
                            (Some(cfg), prof, ctx_sz, mm, rt.version)
                        };
                    let temperature = config_f32(model_node, "temperature", model_profile.temperature);
                    let top_p = Some(config_f32(model_node, "topP", model_profile.top_p));
                    let max_tokens_raw = config_u64(model_node, "maxTokens", 0) as u32;
                    let max_tokens = if max_tokens_raw > 0 {
                        Some(max_tokens_raw)
                    } else if !is_cloud && model_profile.max_tokens > 0 {
                        Some(model_profile.max_tokens)
                    } else {
                        None
                    };
                    let min_p = Some(model_profile.min_p);
                    let repeat_penalty = Some(model_profile.repeat_penalty);
                    let seed = Some(model_profile.seed);
                    let mut system_prompt = ctx.render_template(&config_str(
                        node,
                        "systemPrompt",
                        "Eres un ingeniero de software experto.",
                    ));
                    let raw_max_steps = config_u64(node, "maxSteps", 0) as usize;
                    let auto_steps = raw_max_steps == 0;
                    let max_steps = if auto_steps { 10_000 } else { raw_max_steps };
                    let enabled_tools = config_str_list(node, "tools");

                    // 1. Load connected Skill nodes (inject vs on_demand)
                    let skill_nodes =
                        find_connected_nodes_by_handle(graph, node_map, &node.id, "skill_ref");
                    let mut on_demand_skills: HashMap<String, LoadedSkill> = HashMap::new();
                    for sn in skill_nodes {
                        let mode = config_str(sn, "mode", "inject");
                        let skill_path = config_str(sn, "skillPath", "");
                        let inline_body = config_str(sn, "inlineInstructions", "");
                        let loaded = if !skill_path.trim().is_empty() && Path::new(&skill_path).exists() {
                            load_skill_file(Path::new(&skill_path)).ok()
                        } else if !inline_body.trim().is_empty() {
                            Some(LoadedSkill {
                                name: sn.data.title.clone(),
                                description: config_str(sn, "description", "Skill inline"),
                                body: inline_body,
                                path: "inline".to_string(),
                            })
                        } else {
                            None
                        };

                        if let Some(sk) = loaded {
                            if mode == "on_demand" {
                                log(
                                    iter,
                                    LogStreamKind::System,
                                    format!("Skill '{}' registrada (on_demand).", sk.name),
                                );
                                system_prompt.push_str(&format!(
                                    "\n\n[Skill disponible bajo demanda: '{}' — {} (usa load_skill para leerla)]",
                                    sk.name, sk.description
                                ));
                                on_demand_skills.insert(sk.name.clone(), sk);
                            } else {
                                log(
                                    iter,
                                    LogStreamKind::System,
                                    format!("Skill '{}' inyectada en el system prompt.", sk.name),
                                );
                                system_prompt.push_str(&format!(
                                    "\n\n### Skill: {}\n{}\n",
                                    sk.name, sk.body
                                ));
                            }
                        }
                    }

                    // 2. Connect plugged MCP Server nodes (stdio JSON-RPC 2.0)
                    let mcp_nodes =
                        find_connected_nodes_by_handle(graph, node_map, &node.id, "mcp_ref");
                    let mut mcp_sessions: Vec<McpSession> = Vec::new();
                    let mut mcp_routes: HashMap<String, (usize, String)> = HashMap::new();
                    let mut tool_defs = tool_definitions(&enabled_tools);

                    if !on_demand_skills.is_empty() {
                        tool_defs.push(serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": "load_skill",
                                "description": "Carga las instrucciones completas de una skill disponible bajo demanda.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string", "description": "Nombre exacto de la skill" }
                                    },
                                    "required": ["name"]
                                }
                            }
                        }));
                    }

                    // Connected Knowledge / RAG nodes (via knowledge_ref handle)
                    let knowledge_nodes = find_connected_nodes_by_handle(
                        graph,
                        node_map,
                        &node.id,
                        "knowledge_ref",
                    );
                    let mut knowledge_index: Option<KnowledgeIndex> = None;
                    let mut knowledge_max_chars: usize = 6000;
                    let mut knowledge_top_k: usize = 5;
                    if let Some(kn) = knowledge_nodes.first() {
                        let k_ws = effective_workspace(graph, node_map, kn, ctx);
                        let k_sub = config_str(kn, "subpath", "");
                        knowledge_top_k = config_u64(kn, "topK", 5) as usize;
                        knowledge_max_chars = config_u64(kn, "maxChars", 6000) as usize;
                        match build_or_update_index(&k_ws, &k_sub) {
                            Ok((idx, stats)) => {
                                log(
                                    iter,
                                    LogStreamKind::System,
                                    format!(
                                        "Knowledge / RAG conectado ({} archivos, {} chunks, {} reindexados) -> herramientas search_knowledge y knowledge_tree activas.",
                                        stats.total_files, stats.total_chunks, stats.reindexed_files
                                    ),
                                );
                                knowledge_index = Some(idx);
                                system_prompt.push_str(
                                    "\n\n[Base de conocimiento RAG / Árbol conectada]\nTienes disponibles `search_knowledge(query, top_k)` para buscar fragmentos relevantes por BM25 y `knowledge_tree(path)` para ver el mapa compacto de archivos y símbolos. Úsalas primero para ahorrar contexto.",
                                );
                                tool_defs.push(serde_json::json!({
                                    "type": "function",
                                    "function": {
                                        "name": "search_knowledge",
                                        "description": "Busca fragmentos de código o documentación relevantes en el índice RAG local (BM25).",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "query": { "type": "string", "description": "Consulta técnica, símbolo o concepto a buscar" },
                                                "top_k": { "type": "integer", "description": "Número máximo de fragmentos (por defecto 5)" }
                                            },
                                            "required": ["query"]
                                        }
                                    }
                                }));
                                tool_defs.push(serde_json::json!({
                                    "type": "function",
                                    "function": {
                                        "name": "knowledge_tree",
                                        "description": "Devuelve el árbol de conocimiento compacto (archivos y símbolos principales) del repositorio o subcarpeta.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "path": { "type": "string", "description": "Subcarpeta opcional para filtrar el árbol" }
                                            }
                                        }
                                    }
                                }));
                            }
                            Err(err) => {
                                log(
                                    iter,
                                    LogStreamKind::Stderr,
                                    format!("No se pudo inicializar Knowledge / RAG: {err}"),
                                );
                            }
                        }
                    }

                    for mn in mcp_nodes {
                        let mcp_cfg = mcp_config_from_node(mn);
                        log(
                            iter,
                            LogStreamKind::System,
                            format!(
                                "Conectando servidor MCP '{}' ({} {})...",
                                mcp_cfg.name,
                                mcp_cfg.command,
                                mcp_cfg.args.join(" ")
                            ),
                        );
                        match McpSession::connect(&mcp_cfg, &node_ws).await {
                            Ok(mut session) => {
                                match session.list_tools(&mcp_cfg.allowed_tools).await {
                                    Ok(tools) => {
                                        let session_idx = mcp_sessions.len();
                                        let count = tools.len();
                                        for t in tools {
                                            let prefixed = mcp_prefixed_tool_name(
                                                &session.server_name,
                                                &t.name,
                                            );
                                            tool_defs.push(serde_json::json!({
                                                "type": "function",
                                                "function": {
                                                    "name": prefixed,
                                                    "description": format!("[MCP:{}] {}", session.server_name, t.description),
                                                    "parameters": t.input_schema
                                                }
                                            }));
                                            mcp_routes.insert(prefixed, (session_idx, t.name));
                                        }
                                        log(
                                            iter,
                                            LogStreamKind::System,
                                            format!(
                                                "MCP '{}' conectado ({count} tools disponibles).",
                                                session.server_name
                                            ),
                                        );
                                        mcp_sessions.push(session);
                                    }
                                    Err(err) => {
                                        log(iter, LogStreamKind::Stderr, err);
                                    }
                                }
                            }
                            Err(err) => {
                                log(iter, LogStreamKind::Stderr, err);
                            }
                        }
                    }

                    let model_short = if is_cloud {
                        crate::inference::cloud::parse_cloud_model(&model_path)
                            .map(|(p, m)| format!("{p}/{m}"))
                            .unwrap_or_else(|| model_path.clone())
                    } else {
                        Path::new(&model_path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&model_path)
                            .to_string()
                    };

                    let port = if let Some(load_cfg) = load_cfg_opt {
                        emit(EngineEvent::NodeStatus {
                            run_id: run_id.to_string(),
                            node_id: node.id.clone(),
                            status: NodeStatusKind::Running,
                            iteration: iter,
                            message: Some(format!("Cargando {model_short}...")),
                        });
                        log(
                            iter,
                            LogStreamKind::System,
                            format!(
                                "Cargando modelo GGUF '{}' en llama.cpp ({})...",
                                model_short, rt_version
                            ),
                        );

                        match llama.ensure_loaded_cancellable(load_cfg, Some(&cancel)).await {
                            Ok(p) => {
                                emit(EngineEvent::ModelState {
                                    loaded: true,
                                    model_name: Some(model_short.clone()),
                                    context_size: Some(context_size),
                                });
                                Some(p)
                            }
                            Err(err) if err == "__CANCELLED__" => {
                                return NodeStepOutcome::Finish(
                                    RunOutcomeKind::Cancelled,
                                    "Ejecución detenida inmediatamente por el usuario.".to_string(),
                                );
                            }
                            Err(err) => {
                                log(iter, LogStreamKind::Stderr, err.clone());
                                return NodeStepOutcome::Finish(RunOutcomeKind::Failed, err);
                            }
                        }
                    } else {
                        emit(EngineEvent::ModelState {
                            loaded: true,
                            model_name: Some(model_short.clone()),
                            context_size: Some(context_size),
                        });
                        log(
                            iter,
                            LogStreamKind::System,
                            format!("Usando proveedor Cloud/API '{model_short}'..."),
                        );
                        None
                    };

                    let image_paths: Vec<String> = opt_config_str(node, "imagePath")
                        .map(|raw| ctx.render_template(&raw))
                        .and_then(|p| resolve_image_path(&node_ws, &p))
                        .into_iter()
                        .collect();
                    if !image_paths.is_empty() {
                        log(
                            iter,
                            LogStreamKind::System,
                            format!("Adjuntando imagen multimodal: {}", image_paths[0]),
                        );
                    }

                    let mut messages = vec![
                        ChatMessage {
                            role: "system".into(),
                            content: Some(system_prompt),
                            tool_calls: None,
                            tool_call_id: None,
                            image_paths: Vec::new(),
                        },
                        ChatMessage {
                            role: "user".into(),
                            content: Some(ctx.current_prompt.clone()),
                            tool_calls: None,
                            tool_call_id: None,
                            image_paths,
                        },
                    ];

                    let mut final_output = String::new();
                    let mut last_call_sig = String::new();
                    let mut repeat_call_count: u32 = 0;

                    let compact_budget =
                        compaction_threshold(context_size, model_profile.compact_at_tokens);
                    let tool_bytes = chat_tool_bytes(&tool_defs);

                    for step in 1..=max_steps {
                        if cancel.load(Ordering::SeqCst) {
                            return NodeStepOutcome::Finish(
                                RunOutcomeKind::Cancelled,
                                "Cancelado durante Agent Node.".to_string(),
                            );
                        }

                        let step_tag = if auto_steps {
                            format!("Paso {step} (auto)")
                        } else {
                            format!("Paso {step}/{max_steps}")
                        };

                        // Inject any live instructions sent by the user from the chat drawer
                        for inst in drain_live_instructions() {
                            log(
                                iter,
                                LogStreamKind::System,
                                format!("💬 Instrucción en vivo del usuario: {inst}"),
                            );
                            ctx.current_prompt = format!("{}\n\n[Instrucción del usuario]: {}", ctx.current_prompt, inst);
                            messages.push(ChatMessage {
                                role: "user".into(),
                                content: Some(format!(
                                    "[INSTRUCCIÓN EN VIVO DEL USUARIO — PRIORIDAD ALTA]: {inst}"
                                )),
                                tool_calls: None,
                                tool_call_id: None,
                                image_paths: Vec::new(),
                            });
                        }

                        log(
                            iter,
                            LogStreamKind::System,
                            format!("Agent {step_tag} (llama.cpp local)..."),
                        );

                        if compact_agent_messages(&mut messages, tool_bytes, compact_budget) {
                            log(
                                iter,
                                LogStreamKind::System,
                                "Auto-compactación de contexto aplicada para evitar exceder ctx_size.".to_string(),
                            );
                        }

                        let mut token_line = String::new();
                        let mut thinking_line = String::new();
                        let mut notified_thinking = false;
                        let mut notified_writing = false;

                        emit(EngineEvent::NodeStatus {
                            run_id: run_id.to_string(),
                            node_id: node.id.clone(),
                            status: NodeStatusKind::Running,
                            iteration: iter,
                            message: Some(format!("{step_tag}: pensando...")),
                        });

                        if ctx.max_llm_calls > 0 && ctx.llm_call_count >= ctx.max_llm_calls {
                            let msg = format!(
                                "Run budget reached: {}/{} LLM calls.",
                                ctx.llm_call_count, ctx.max_llm_calls
                            );
                            log(iter, LogStreamKind::Stderr, msg.clone());
                            return NodeStepOutcome::Finish(RunOutcomeKind::Exhausted, msg);
                        }
                        ctx.llm_call_count += 1;

                        let agent_req = ChatRequest {
                            messages: messages.clone(),
                            tools: tool_defs.clone(),
                            temperature,
                            top_p,
                            max_tokens,
                            min_p,
                            repeat_penalty,
                            seed,
                            reasoning_effort: opt_config_str(node, "reasoningEffort"),
                        };
                        let mut handle_tok = |is_thinking: bool, tok: &str| {
                            if is_thinking {
                                if !notified_thinking {
                                    notified_thinking = true;
                                    emit(EngineEvent::NodeStatus {
                                        run_id: run_id.to_string(),
                                        node_id: node.id.clone(),
                                        status: NodeStatusKind::Running,
                                        iteration: iter,
                                        message: Some(format!("{step_tag}: razonando...")),
                                    });
                                }
                                thinking_line.push_str(tok);
                                while let Some(nl) = thinking_line.find('\n') {
                                    let line = thinking_line[..nl].trim_end().to_string();
                                    thinking_line.drain(..=nl);
                                    if !line.trim().is_empty() {
                                        log(iter, LogStreamKind::Thinking, line);
                                    }
                                }
                                if thinking_line.len() >= 140 {
                                    if let Some(sp) = thinking_line.rfind(' ') {
                                        let chunk = thinking_line[..sp].to_string();
                                        thinking_line.drain(..=sp);
                                        if !chunk.trim().is_empty() {
                                            log(iter, LogStreamKind::Thinking, chunk);
                                        }
                                    }
                                }
                            } else {
                                if !notified_writing {
                                    notified_writing = true;
                                    emit(EngineEvent::NodeStatus {
                                        run_id: run_id.to_string(),
                                        node_id: node.id.clone(),
                                        status: NodeStatusKind::Running,
                                        iteration: iter,
                                        message: Some(format!("{step_tag}: escribiendo...")),
                                    });
                                }
                                token_line.push_str(tok);
                                while let Some(nl) = token_line.find('\n') {
                                    let line = token_line[..nl].trim_end().to_string();
                                    token_line.drain(..=nl);
                                    if !line.trim().is_empty() {
                                        log(iter, LogStreamKind::Agent, line);
                                    }
                                }
                                if token_line.len() >= 140 {
                                    if let Some(sp) = token_line.rfind(' ') {
                                        let chunk = token_line[..sp].to_string();
                                        token_line.drain(..=sp);
                                        if !chunk.trim().is_empty() {
                                            log(iter, LogStreamKind::Agent, chunk);
                                        }
                                    }
                                }
                            }
                        };
                        let chat_res = if let Some(p) = port {
                            let retry_req = agent_req.clone();
                            let first = llama.chat_stream(p, agent_req, Some(cancel), &mut handle_tok).await;
                            match first {
                                Err(ref err) if err.contains("Failed to parse tool call arguments as JSON") => {
                                    let bumped_rp = (retry_req.repeat_penalty.unwrap_or(1.0) + 0.10).clamp(1.08, 1.18);
                                    let bumped_temp = retry_req.temperature.max(0.55);
                                    log(
                                        iter,
                                        LogStreamKind::System,
                                        format!(
                                            "Tool call JSON truncado por bucle de repetición; reintentando automáticamente con repeat_penalty={bumped_rp:.2}, temperature={bumped_temp:.2}..."
                                        ),
                                    );
                                    let mut adjusted = retry_req;
                                    adjusted.repeat_penalty = Some(bumped_rp);
                                    adjusted.temperature = bumped_temp;
                                    llama.chat_stream(p, adjusted, Some(cancel), &mut handle_tok).await
                                }
                                other => other,
                            }
                        } else {
                            crate::inference::cloud::chat_stream_cloud(
                                &model_path,
                                agent_req,
                                Some(cancel),
                                &mut handle_tok,
                            )
                            .await
                        };
                        let resp = match chat_res {
                            Ok(r) => r,
                            Err(err) if err == "__CANCELLED__" => {
                                return NodeStepOutcome::Finish(
                                    RunOutcomeKind::Cancelled,
                                    "Ejecución detenida inmediatamente por el usuario.".to_string(),
                                );
                            }
                            Err(err) => {
                                log(iter, LogStreamKind::Stderr, err.clone());
                                return NodeStepOutcome::Finish(RunOutcomeKind::Failed, err);
                            }
                        };

                        if !thinking_line.trim().is_empty() {
                            log(iter, LogStreamKind::Thinking, thinking_line.trim().to_string());
                        }
                        if !token_line.trim().is_empty() {
                            log(iter, LogStreamKind::Agent, token_line.trim().to_string());
                        }

                        let mut effective_tool_calls = resp.tool_calls.clone();
                        if effective_tool_calls.is_empty() && resp.content.contains("<tool_call>") {
                            let rescued = parse_inline_tool_calls(&resp.content);
                            if !rescued.is_empty() {
                                log(
                                    iter,
                                    LogStreamKind::System,
                                    format!("Auto-reparación: rescatadas {} llamadas <tool_call> desde texto.", rescued.len()),
                                );
                                effective_tool_calls = rescued;
                            }
                        }

                        if !resp.content.is_empty() && effective_tool_calls.is_empty() {
                            final_output = resp.content.clone();
                        }

                        if effective_tool_calls.is_empty() {
                            let truncated = resp.finish_reason.as_deref() == Some("length")
                                || resp.content.trim().is_empty();
                            if truncated && step < max_steps && repeat_call_count < 3 {
                                repeat_call_count += 1;
                                log(
                                    iter,
                                    LogStreamKind::System,
                                    "El turno terminó durante el bloque de razonamiento (o por límite de tokens de salida); auto-continuando para completar la acción o el informe...".to_string(),
                                );
                                if !resp.content.trim().is_empty() {
                                    messages.push(ChatMessage {
                                        role: "assistant".into(),
                                        content: Some(resp.content.clone()),
                                        tool_calls: None,
                                        tool_call_id: None,
                                        image_paths: Vec::new(),
                                    });
                                }
                                messages.push(ChatMessage {
                                    role: "user".into(),
                                    content: Some(
                                        "Tu respuesta anterior se cortó antes de emitir la llamada a herramienta o el informe final. Ejecuta ahora la herramienta necesaria (ej. read_file / write_file) o escribe la conclusión directamente."
                                            .into(),
                                    ),
                                    tool_calls: None,
                                    tool_call_id: None,
                                    image_paths: Vec::new(),
                                });
                                continue;
                            }
                            break;
                        }

                        messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: if resp.content.is_empty() {
                                None
                            } else {
                                Some(resp.content.clone())
                            },
                            tool_calls: Some(effective_tool_calls.clone()),
                            tool_call_id: None,
                            image_paths: Vec::new(),
                        });

                        let step_sig: String = effective_tool_calls
                            .iter()
                            .map(|c| format!("{}:{}", c.function.name, c.function.arguments))
                            .collect::<Vec<_>>()
                            .join("|");

                        let mut screenshot_to_attach: Option<String> = None;

                        for tc in effective_tool_calls {
                            if !tool_defs.iter().any(|tool| {
                                tool["function"]["name"].as_str() == Some(tc.function.name.as_str())
                            }) {
                                log(
                                    iter,
                                    LogStreamKind::Stderr,
                                    format!("Herramienta deshabilitada: {}", tc.function.name),
                                );
                                messages.push(ChatMessage {
                                    role: "tool".into(),
                                    content: Some(format!(
                                        "La herramienta '{}' no está habilitada para este nodo.",
                                        tc.function.name
                                    )),
                                    tool_calls: None,
                                    tool_call_id: Some(tc.id),
                                    image_paths: Vec::new(),
                                });
                                continue;
                            }

                            emit(EngineEvent::NodeStatus {
                                run_id: run_id.to_string(),
                                node_id: node.id.clone(),
                                status: NodeStatusKind::Running,
                                iteration: iter,
                                message: Some(format!("Tool: {}", tc.function.name)),
                            });
                            log(
                                iter,
                                LogStreamKind::System,
                                format!("Tool call -> {}({})", tc.function.name, truncate(&tc.function.arguments, 160)),
                            );

                            let parsed_res = if tc.function.arguments.trim().is_empty() {
                                Ok(serde_json::json!({}))
                            } else {
                                serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            };

                            let result = match parsed_res {
                                Err(parse_err) => format!(
                                    "Error de sintaxis JSON en arguments para '{}': {parse_err}. Recibido: {}. Corrige el JSON y reintenta.",
                                    tc.function.name,
                                    truncate(&tc.function.arguments, 240)
                                ),
                                Ok(parsed_args) => {
                                    if tc.function.name == "load_skill" {
                                        let name = parsed_args
                                            .get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        on_demand_skills
                                            .get(name)
                                            .map(|s| format!("# Skill: {}\n{}", s.name, s.body))
                                            .unwrap_or_else(|| format!("Skill '{name}' no encontrada."))
                                    } else if tc.function.name == "search_knowledge" {
                                        if let Some(idx) = &knowledge_index {
                                            let q = parsed_args
                                                .get("query")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("");
                                            let k = parsed_args
                                                .get("top_k")
                                                .and_then(|v| v.as_u64())
                                                .map(|v| v as usize)
                                                .unwrap_or(knowledge_top_k);
                                            search_bm25(idx, q, k, knowledge_max_chars)
                                        } else {
                                            "No hay índice Knowledge / RAG conectado.".to_string()
                                        }
                                    } else if tc.function.name == "knowledge_tree" {
                                        if let Some(idx) = &knowledge_index {
                                            let sub = parsed_args.get("path").and_then(|v| v.as_str());
                                            build_knowledge_tree(idx, sub, knowledge_max_chars)
                                        } else {
                                            "No hay índice Knowledge / RAG conectado.".to_string()
                                        }
                                    } else if let Some((idx, orig_name)) = mcp_routes.get(&tc.function.name)
                                    {
                                        match mcp_sessions[*idx].call_tool(orig_name, parsed_args).await {
                                            Ok(out) => out,
                                            Err(err) => format!("Error MCP: {err}"),
                                        }
                                    } else {
                                        let out = execute_tool_cancellable(&node_ws, &tc.function.name, &parsed_args, Some(cancel)).await;
                                        if tc.function.name == "browser_action" && has_mmproj {
                                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&out) {
                                                if let Some(shot) = v.get("screenshot").and_then(|s| s.as_str()) {
                                                    if Path::new(shot).exists() {
                                                        screenshot_to_attach = Some(shot.to_string());
                                                    }
                                                }
                                            }
                                        }
                                        out
                                    }
                                }
                            };
                            log(
                                iter,
                                LogStreamKind::Stdout,
                                format!("Tool {} => {}", tc.function.name, truncate(&result, 240)),
                            );
                            messages.push(ChatMessage {
                                role: "tool".into(),
                                content: Some(truncate(&result, 8000)),
                                tool_calls: None,
                                tool_call_id: Some(tc.id),
                                image_paths: Vec::new(),
                            });
                        }

                        if let Some(shot_path) = screenshot_to_attach {
                            messages.push(ChatMessage {
                                role: "user".into(),
                                content: Some("[Captura actual del navegador adjunta para inspección visual VLM]".into()),
                                tool_calls: None,
                                tool_call_id: None,
                                image_paths: vec![shot_path],
                            });
                        }

                        // Stall / repetitive tool-call detector (especially useful in Auto mode)
                        if step_sig == last_call_sig {
                            repeat_call_count += 1;
                        } else {
                            last_call_sig = step_sig;
                            repeat_call_count = 1;
                        }
                        if repeat_call_count >= 3 {
                            log(
                                iter,
                                LogStreamKind::System,
                                "Detector de atasco: llamada repetida 3 veces seguidas; indicando al agente cambiar de enfoque o concluir.".to_string(),
                            );
                            messages.push(ChatMessage {
                                role: "user".into(),
                                content: Some(
                                    "Aviso del motor: has repetido exactamente la misma llamada a herramienta 3 veces seguidas sin nuevos resultados. Cambia de archivo/enfoque o concluye ya tu informe final sin llamar más herramientas."
                                        .into(),
                                ),
                                tool_calls: None,
                                tool_call_id: None,
                                image_paths: Vec::new(),
                            });
                            repeat_call_count = 0;
                        }
                    }

                    // If the agent spent all max_steps calling tools without producing a final text answer,
                    // give it 1 final synthesis turn (without tools) so downstream nodes (e.g. LLM Judge) get a real summary.
                    if final_output.trim().is_empty() && !cancel.load(Ordering::SeqCst) {
                        log(
                            iter,
                            LogStreamKind::System,
                            "Límite de tool steps alcanzado — generando síntesis final...".to_string(),
                        );
                        messages.push(ChatMessage {
                            role: "user".into(),
                            content: Some("Has alcanzado el límite de pasos de herramientas. NO emitas etiquetas <tool_call> ni intentes llamar funciones; responde directamente en Markdown limpio resumiendo tus hallazgos concretos.".into()),
                            tool_calls: None,
                            tool_call_id: None,
                            image_paths: Vec::new(),
                        });
                        let synth_req = ChatRequest {
                            messages,
                            tools: Vec::new(),
                            temperature,
                            top_p,
                            max_tokens,
                            min_p,
                            repeat_penalty,
                            seed,
                            reasoning_effort: opt_config_str(node, "reasoningEffort"),
                        };
                        let synth_res = if let Some(p) = port {
                            llama.chat_stream(p, synth_req, Some(cancel), |_, _| {}).await
                        } else {
                            crate::inference::cloud::chat_stream_cloud(
                                &model_path,
                                synth_req,
                                Some(cancel),
                                |_, _| {},
                            )
                            .await
                        };
                        if let Ok(resp) = synth_res {
                            if !resp.content.trim().is_empty() {
                                final_output = resp.content.clone();
                                log(iter, LogStreamKind::Agent, truncate(&final_output, 800));
                            }
                        }
                    }

                    ctx.last_agent_output = final_output;
                    return NodeStepOutcome::Continue("exec".to_string());
                } else {
                    let msg = format!(
                        "Modelo GGUF no encontrado en '{}'. Selecciona un modelo GGUF válido en el nodo Local Model (o escanea tus modelos locales).",
                        model_path
                    );
                    log(iter, LogStreamKind::Stderr, msg.clone());
                    return NodeStepOutcome::Finish(RunOutcomeKind::Failed, msg);
                }
            } else {
                log(
                    iter,
                    LogStreamKind::System,
                    "Runtime llama.cpp no instalado aún (abre 'Runtime' para descargarlo) o sin Local Model conectado. Usando MockAgent.".to_string(),
                );
            }

            log(
                iter,
                LogStreamKind::Agent,
                format!("Contexto recibido: {}", truncate(&ctx.current_prompt, 240)),
            );
            tokio::time::sleep(Duration::from_millis(250)).await;
            if cancel.load(Ordering::SeqCst) {
                return NodeStepOutcome::Finish(
                    RunOutcomeKind::Cancelled,
                    "Cancelado durante Agent Node.".to_string(),
                );
            }
            let response = format!(
                "[MockAgent iter {}] Instala llama.cpp en Runtime y apunta el nodo Local Model a un .gguf válido para inferencia real.",
                iter
            );
            ctx.last_agent_output = response.clone();
            log(iter, LogStreamKind::Agent, response);
            NodeStepOutcome::Continue("exec".to_string())
        }

        "shell" => {
            let command_str = ctx.render_template(&config_str(node, "command", ""));
            let timeout_sec = config_u64(node, "timeoutSec", 120);
            let docker_image = config_str(node, "dockerImage", "");

            if command_str.trim().is_empty() {
                return NodeStepOutcome::Finish(
                    RunOutcomeKind::Failed,
                    "Shell Node sin comando configurado.".to_string(),
                );
            }

            let iter = ctx.iteration;
            let node_ws = effective_workspace(graph, node_map, node, ctx);
            let effective_cmd = if !docker_image.trim().is_empty() {
                let ws_abs = if node_ws.trim().is_empty() || node_ws == "." {
                    std::env::current_dir()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|_| ".".to_string())
                } else {
                    node_ws.clone()
                };
                let escaped = command_str.replace('"', "\\\"");
                format!(
                    "docker run --rm -v \"{ws_abs}:/workspace\" -w /workspace {} sh -c \"{escaped}\"",
                    docker_image.trim()
                )
            } else {
                command_str.clone()
            };

            log(
                iter,
                LogStreamKind::System,
                if !docker_image.trim().is_empty() {
                    format!("Docker Sandbox ({}) $ {}", docker_image.trim(), command_str)
                } else {
                    format!("Ejecutando en '{}': $ {}", node_ws, command_str)
                },
            );

            match run_shell_command(
                &effective_cmd,
                &node_ws,
                timeout_sec,
                cancel,
                |stream, line| log(iter, stream, line),
            )
            .await
            {
                Ok((exit_code, stdout, stderr)) => {
                    ctx.last_exit_code = Some(exit_code);
                    ctx.last_stdout = stdout;
                    ctx.last_stderr = stderr;
                    log(
                        iter,
                        LogStreamKind::System,
                        format!("Comando finalizado con exit_code = {exit_code}"),
                    );
                    NodeStepOutcome::Continue("exec".to_string())
                }
                Err(ShellRunError::Cancelled) => NodeStepOutcome::Finish(
                    RunOutcomeKind::Cancelled,
                    "Comando detenido por el usuario.".to_string(),
                ),
                Err(ShellRunError::Failed(msg)) => {
                    ctx.last_exit_code = Some(-1);
                    ctx.last_stderr = msg.clone();
                    log(iter, LogStreamKind::Stderr, msg.clone());
                    NodeStepOutcome::Finish(RunOutcomeKind::Failed, msg)
                }
            }
        }

        "condition" => {
            let expr = config_str(node, "expression", "exit_code == 0");
            let passed = evaluate_condition(&expr, ctx);
            log(
                ctx.iteration,
                LogStreamKind::System,
                format!(
                    "Evaluando '{}' (exit_code={:?}, iter={}) => {}",
                    expr, ctx.last_exit_code, ctx.iteration, passed
                ),
            );
            NodeStepOutcome::Continue(if passed { "true" } else { "false" }.to_string())
        }

        "loop_guard" => {
            let max_iterations = config_u64(node, "maxIterations", 0) as u32;
            let infinite_loop = max_iterations == 0;
            let template = config_str(
                node,
                "feedbackTemplate",
                "Intento anterior fallido:\n{{stderr}}",
            );

            let count = ctx.loop_counts.entry(node.id.clone()).or_insert(0);
            *count += 1;
            let attempts_so_far = *count;

            ctx.loop_ledger.push(format!(
                "- Ciclo #{}: Judge={} | Resumen={}",
                ctx.iteration,
                truncate(&ctx.last_judge_reason, 120),
                truncate(&ctx.last_agent_output, 160)
            ));
            if ctx.loop_ledger.len() > 25 {
                ctx.loop_ledger.remove(0);
            }

            if infinite_loop || attempts_so_far < max_iterations {
                ctx.iteration = attempts_so_far + 1;
                let feedback = ctx.render_template(&template);
                ctx.current_prompt = feedback.clone();

                emit(EngineEvent::IterationChanged {
                    run_id: run_id.to_string(),
                    iteration: ctx.iteration,
                    max_iterations,
                });
                log(
                    ctx.iteration,
                    LogStreamKind::System,
                    if infinite_loop {
                        format!(
                            "Ciclo continuo #{}: avanzando al siguiente ciclo autónomo.",
                            ctx.iteration
                        )
                    } else {
                        format!(
                            "Reintentando loop: iteración {}/{} — inyectando feedback.",
                            ctx.iteration, max_iterations
                        )
                    },
                );
                NodeStepOutcome::Continue("retry".to_string())
            } else {
                let msg = format!(
                    "Loop Guard agotado: se alcanzó el máximo de {max_iterations} iteraciones."
                );
                log(ctx.iteration, LogStreamKind::System, msg.clone());
                if find_next_node(&graph.edges, &node.id, "exhausted").is_some() {
                    NodeStepOutcome::Continue("exhausted".to_string())
                } else {
                    let summary = if ctx.last_judge_reason.is_empty() {
                        msg
                    } else {
                        format!("{msg} Última evaluación: {}", ctx.last_judge_reason)
                    };
                    NodeStepOutcome::Finish(RunOutcomeKind::Exhausted, summary)
                }
            }
        }

        "git_checkpoint" => {
            let mode = config_str(node, "mode", "diff");
            let iter = ctx.iteration;
            let node_ws = effective_workspace(graph, node_map, node, ctx);

            match mode.as_str() {
                "delivery_branch" => {
                    let plan = validate_json_output(&ctx.last_agent_output, "object", &[])
                        .and_then(|json| {
                            serde_json::from_str::<super::delivery::DeliveryPlan>(&json)
                                .map_err(|err| format!("Faltan datos para crear la rama: {err}"))
                        });
                    let plan = match plan {
                        Ok(plan) => plan,
                        Err(err) => return NodeStepOutcome::Finish(RunOutcomeKind::Failed, err),
                    };
                    let remote = config_str(node, "remote", "origin");
                    match super::delivery::prepare(Path::new(&node_ws), plan, remote.trim()).await {
                        Ok(delivery) => {
                            ctx.workspace = delivery.workspace.to_string_lossy().into_owned();
                            ctx.node_outputs.insert(
                                node.id.clone(),
                                serde_json::to_string(&delivery).expect("UTF-8 delivery workspace"),
                            );
                            log(
                                iter,
                                LogStreamKind::System,
                                format!("Rama {} creada en {}", delivery.branch, ctx.workspace),
                            );
                        }
                        Err(err) => return NodeStepOutcome::Finish(RunOutcomeKind::Failed, err),
                    }
                }
                "delivery_publish" => {
                    let branch_node = config_str(node, "branchNodeId", "");
                    let delivery = ctx.node_outputs.get(&branch_node).and_then(|value| {
                        serde_json::from_str::<super::delivery::DeliveryBranch>(value).ok()
                    });
                    let Some(delivery) = delivery else {
                        return NodeStepOutcome::Finish(
                            RunOutcomeKind::Failed,
                            "Falta el worktree de la rama de entrega.".to_string(),
                        );
                    };
                    match super::delivery::publish(&delivery).await {
                        Ok(summary) => {
                            log(iter, LogStreamKind::System, summary.clone());
                            ctx.node_outputs.insert(node.id.clone(), summary);
                        }
                        Err(err) => return NodeStepOutcome::Finish(RunOutcomeKind::Failed, err),
                    }
                }
                "rollback" => {
                    log(
                        iter,
                        LogStreamKind::System,
                        format!("Git Rollback: restaurando cambios no confirmados en '{}'...", node_ws),
                    );
                    let out = run_git(&node_ws, &["checkout", "--", "."]).await;
                    log(iter, LogStreamKind::System, out);
                    ctx.last_git_diff = capture_git_diff(&node_ws).await;
                }
                "snapshot" => {
                    let hash = run_git(&node_ws, &["stash", "create"]).await;
                    let diff = capture_git_diff(&node_ws).await;
                    ctx.last_git_diff = diff;
                    let label = if hash.trim().is_empty() {
                        "working tree limpio (sin cambios)".to_string()
                    } else {
                        format!("snapshot {}", hash.trim())
                    };
                    log(iter, LogStreamKind::System, format!("Git Checkpoint: {label}"));
                }
                _ => {
                    let diff = capture_git_diff(&node_ws).await;
                    let lines = diff.lines().count();
                    ctx.last_git_diff = diff;
                    log(
                        iter,
                        LogStreamKind::System,
                        format!("Git Diff capturado ({lines} líneas) disponible en {{{{git_diff}}}}."),
                    );
                }
            }
            NodeStepOutcome::Continue("exec".to_string())
        }

        "context_compactor" => {
            let template = config_str(
                node,
                "template",
                "Objetivo: {{prompt}}\nIteración: {{iteration}}\nÚltimo error:\n{{stderr}}\nDiff actual:\n{{git_diff}}",
            );
            let max_chars = config_u64(node, "maxChars", 4000) as usize;
            let rendered = ctx.render_template(&template);
            let compacted = truncate(&rendered, max_chars);
            ctx.current_prompt = compacted.clone();
            log(
                ctx.iteration,
                LogStreamKind::System,
                format!(
                    "Context Compactor: contexto compactado a {} caracteres.",
                    compacted.chars().count()
                ),
            );
            NodeStepOutcome::Continue("exec".to_string())
        }

        "human_gate" => {
            let prompt_msg = config_str(
                node,
                "message",
                "Revisa los cambios y decide si aprobar o pedir otra iteración.",
            );
            let iter = ctx.iteration;
            let diff = capture_git_diff(&ctx.workspace).await;
            ctx.last_git_diff = diff.clone();

            let (tx, rx) = oneshot::channel::<HumanDecision>();
            if let Ok(mut slot) = gate_slot.lock() {
                *slot = Some(tx);
            }

            emit(EngineEvent::NodeStatus {
                run_id: run_id.to_string(),
                node_id: node.id.clone(),
                status: NodeStatusKind::Paused,
                iteration: iter,
                message: Some(prompt_msg.clone()),
            });
            emit(EngineEvent::HumanGateWaiting {
                run_id: run_id.to_string(),
                node_id: node.id.clone(),
                node_title: node.data.title.clone(),
                iteration: iter,
                message: prompt_msg.clone(),
                git_diff: diff,
            });
            log(
                iter,
                LogStreamKind::System,
                format!("Human Gate en pausa: {prompt_msg}"),
            );

            match rx.await {
                Ok(decision) => {
                    if decision.approved {
                        log(iter, LogStreamKind::System, "Human Gate: APROBADO.".to_string());
                        NodeStepOutcome::Continue("approved".to_string())
                    } else {
                        let fb = decision
                            .feedback
                            .unwrap_or_else(|| "El usuario solicitó ajustes en los cambios.".to_string());
                        ctx.current_prompt = format!(
                            "Feedback humano (iteración {iter}):\n{fb}\n\nObjetivo original:\n{}",
                            ctx.initial_prompt
                        );
                        log(
                            iter,
                            LogStreamKind::System,
                            format!("Human Gate: RECHAZADO con feedback -> {}", truncate(&fb, 180)),
                        );
                        NodeStepOutcome::Continue("rejected".to_string())
                    }
                }
                Err(_) => NodeStepOutcome::Finish(
                    RunOutcomeKind::Cancelled,
                    "Human Gate cancelado.".to_string(),
                ),
            }
        }

        "knowledge_rag" => {
            let mode = config_str(node, "mode", "hybrid");
            let subpath = config_str(node, "subpath", "");
            let query_tpl = config_str(node, "queryTemplate", "{{current_prompt}}");
            let query = ctx.render_template(&query_tpl);
            let top_k = config_u64(node, "topK", 5) as usize;
            let max_chars = config_u64(node, "maxChars", 6000) as usize;
            let node_ws = effective_workspace(graph, node_map, node, ctx);
            let iter = ctx.iteration;

            match query_knowledge(&node_ws, &subpath, &mode, &query, top_k, max_chars) {
                Ok((content, stats)) => {
                    let chars = content.len();
                    ctx.last_rag_context = content;
                    log(
                        iter,
                        LogStreamKind::System,
                        format!(
                            "Knowledge / RAG ({mode}): {} archivos ({} reindexados, {} chunks) -> {} chars disponibles en {{{{rag_context}}}}.",
                            stats.total_files, stats.reindexed_files, stats.total_chunks, chars
                        ),
                    );
                    NodeStepOutcome::Continue("exec".to_string())
                }
                Err(err) => {
                    log(iter, LogStreamKind::Stderr, err.clone());
                    NodeStepOutcome::Finish(RunOutcomeKind::Failed, err)
                }
            }
        }

        "memory_file" => {
            let mode = config_str(node, "mode", "read");
            let rel_path = config_str(node, "filePath", ".loors/BACKLOG.md");
            let template = config_str(
                node,
                "template",
                "## Iteración {{iteration}}\n- Resultado: {{agent_output}}\n- Evaluación: {{judge_reason}}\n",
            );
            let iter = ctx.iteration;
            let node_ws = effective_workspace(graph, node_map, node, ctx);

            let target = match resolve_workspace_file(&node_ws, &rel_path) {
                Ok(p) => p,
                Err(err) => {
                    log(iter, LogStreamKind::Stderr, err.clone());
                    return NodeStepOutcome::Finish(RunOutcomeKind::Failed, err);
                }
            };

            match mode.as_str() {
                "write" => {
                    if let Some(parent) = target.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let rendered = ctx.render_template(&template);
                    if let Err(e) = std::fs::write(&target, &rendered) {
                        let msg = format!("Error escribiendo memoria '{}': {e}", rel_path);
                        log(iter, LogStreamKind::Stderr, msg.clone());
                        return NodeStepOutcome::Finish(RunOutcomeKind::Failed, msg);
                    }
                    ctx.last_memory = rendered;
                    log(
                        iter,
                        LogStreamKind::System,
                        format!("Memory File escrito: '{rel_path}'."),
                    );
                }
                "append" => {
                    if let Some(parent) = target.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let existing = std::fs::read_to_string(&target).unwrap_or_default();
                    let rendered = ctx.render_template(&template);
                    let combined = if existing.is_empty() {
                        rendered
                    } else {
                        format!("{existing}\n{rendered}")
                    };
                    if let Err(e) = std::fs::write(&target, &combined) {
                        let msg = format!("Error añadiendo a memoria '{}': {e}", rel_path);
                        log(iter, LogStreamKind::Stderr, msg.clone());
                        return NodeStepOutcome::Finish(RunOutcomeKind::Failed, msg);
                    }
                    ctx.last_memory = combined;
                    log(
                        iter,
                        LogStreamKind::System,
                        format!("Memory File actualizado (append): '{rel_path}'."),
                    );
                }
                _ => {
                    let content = std::fs::read_to_string(&target).unwrap_or_default();
                    ctx.last_memory = content.clone();
                    log(
                        iter,
                        LogStreamKind::System,
                        format!(
                            "Memory File leído ('{rel_path}', {} bytes) disponible en {{{{memory}}}}.",
                            content.len()
                        ),
                    );
                }
            }
            NodeStepOutcome::Continue("exec".to_string())
        }

        "llm_judge" => {
            let rubric = config_str(
                node,
                "rubric",
                "Evalúa si el cambio actual resuelve el objetivo sin regresiones. Responde JSON: {\"decision\":\"pass\"|\"fail\",\"reason\":\"...\"}",
            );
            let input_tpl = config_str(
                node,
                "inputTemplate",
                "Objetivo: {{prompt}}\nExit code: {{exit_code}}\nStderr:\n{{stderr}}\nDiff:\n{{git_diff}}\nMemoria:\n{{memory}}",
            );
            let iter = ctx.iteration;
            let mut eval_input = ctx.render_template(&input_tpl);
            if !input_tpl.contains("{{agent_output}}")
                && !ctx.last_agent_output.trim().is_empty()
                && eval_input.trim().len() < 80
            {
                eval_input.push_str(&format!(
                    "\n\nEvidencia automática del agente:\n{}",
                    truncate(&ctx.last_agent_output, 2500)
                ));
            }
            if !input_tpl.contains("{{loop_ledger}}") && !ctx.loop_ledger.is_empty() {
                eval_input.push_str(&format!(
                    "\n\nHistorial de ciclos previos:\n{}",
                    ctx.loop_ledger.join("\n")
                ));
            }

            let connected_model = find_connected_model_node(graph, node_map, &node.id);
            let installed_runtime = find_installed_binary(backend.clone());
            let strict_verdict = node
                .data
                .config
                .get("strictVerdict")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);

            let (passed, reason) = if let Some(model_node) = connected_model {
                let model_path = config_str(model_node, "modelPath", "");
                let is_cloud = crate::inference::cloud::is_cloud_model(&model_path);
                if is_cloud || (installed_runtime.is_some() && Path::new(&model_path).exists()) {
                    let judge_images: Vec<String> = opt_config_str(node, "imagePath")
                        .map(|raw| ctx.render_template(&raw))
                        .and_then(|p| resolve_image_path(&ctx.workspace, &p))
                        .into_iter()
                        .collect();

                    let (port_res, model_profile) = if is_cloud {
                        log(
                            iter,
                            LogStreamKind::System,
                            format!("LLM Judge evaluando con Cloud/API '{}'...", model_path),
                        );
                        (Ok(None), crate::inference::ModelTuningProfile::default())
                    } else {
                        let rt = installed_runtime.unwrap();
                        let raw_ctx = config_u64(model_node, "contextSize", 0) as u32;
                        let (load_cfg, prof) = resolve_load_config_with_profile(
                            rt.binary_path,
                            &model_path,
                            if raw_ctx > 0 { Some(raw_ctx) } else { None },
                            opt_config_str(model_node, "mmprojPath"),
                        );
                        log(
                            iter,
                            LogStreamKind::System,
                            format!("LLM Judge evaluando con '{}'...", model_path),
                        );
                        (llama.ensure_loaded_cancellable(load_cfg, Some(&cancel)).await.map(Some), prof)
                    };

                    match port_res {
                        Ok(port_opt) => {
                            let req = ChatRequest {
                                messages: vec![
                                    ChatMessage {
                                        role: "system".into(),
                                        content: Some(format!(
                                            "{rubric}\nResponde EXCLUSIVAMENTE un objeto JSON válido con las claves \"decision\" (\"pass\" o \"fail\") y \"reason\" (string breve)."
                                        )),
                                        tool_calls: None,
                                        tool_call_id: None,
                                        image_paths: Vec::new(),
                                    },
                                    ChatMessage {
                                        role: "user".into(),
                                        content: Some(eval_input),
                                        tool_calls: None,
                                        tool_call_id: None,
                                        image_paths: judge_images,
                                    },
                                ],
                                tools: Vec::new(),
                                temperature: 0.1,
                                top_p: Some(model_profile.top_p),
                                max_tokens: None,
                                min_p: Some(model_profile.min_p),
                                repeat_penalty: Some(model_profile.repeat_penalty),
                                seed: Some(model_profile.seed),
                                reasoning_effort: opt_config_str(node, "reasoningEffort")
                                    .or_else(|| Some("low".to_string())),
                            };
                            let judge_res = if let Some(port) = port_opt {
                                llama.chat_stream(port, req, Some(cancel), |_, _| {}).await
                            } else {
                                crate::inference::cloud::chat_stream_cloud(
                                    &model_path,
                                    req,
                                    Some(cancel),
                                    |_, _| {},
                                )
                                .await
                            };
                            match judge_res {
                                Ok(resp) => {
                                    if resp.content.trim().is_empty() {
                                        (false, "El LLM Judge devolvió una respuesta vacía (solo thinking).".to_string())
                                    } else if strict_verdict {
                                        parse_judge_verdict(&resp.content, true)
                                    } else {
                                        parse_judge_json(&resp.content)
                                    }
                                }
                                Err(err) if err == "__CANCELLED__" => {
                                    return NodeStepOutcome::Finish(
                                        RunOutcomeKind::Cancelled,
                                        "Ejecución detenida durante LLM Judge.".to_string(),
                                    );
                                }
                                Err(err) => (false, format!("Error en LLM Judge: {err}")),
                            }
                        }
                        Err(err) => (false, format!("Error cargando modelo en Judge: {err}")),
                    }
                } else {
                    let fallback_pass = !strict_verdict && ctx.last_exit_code == Some(0);
                    (
                        fallback_pass,
                        if strict_verdict {
                            "El juez requiere un modelo disponible para emitir un veredicto.".to_string()
                        } else {
                            format!(
                                "Fallback heurístico (GGUF no encontrado): exit_code={:?}",
                                ctx.last_exit_code
                            )
                        },
                    )
                }
            } else {
                let fallback_pass = !strict_verdict && ctx.last_exit_code == Some(0);
                (
                    fallback_pass,
                    if strict_verdict {
                        "El juez requiere un modelo conectado para emitir un veredicto.".to_string()
                    } else {
                        format!(
                            "Fallback heurístico (sin runtime/modelo conectado): exit_code={:?}",
                            ctx.last_exit_code
                        )
                    },
                )
            };

            ctx.last_judge_reason = reason.clone();
            log(
                iter,
                LogStreamKind::Agent,
                format!(
                    "LLM Judge => {} | {}",
                    if passed { "PASS" } else { "FAIL" },
                    reason
                ),
            );
            if !passed && find_next_node(&graph.edges, &node.id, "fail").is_none() {
                NodeStepOutcome::Finish(RunOutcomeKind::Failed, reason)
            } else {
                NodeStepOutcome::Continue(if passed { "pass" } else { "fail" }.to_string())
            }
        }

        "notify" => {
            let channel = config_str(node, "channel", "telegram");
            let bot_token = expand_env_vars(&config_str(node, "botToken", ""));
            let chat_id = expand_env_vars(&config_str(node, "chatId", ""));
            let webhook_url = expand_env_vars(&config_str(node, "webhookUrl", ""));
            let msg_tpl = config_str(node, "messageTemplate", "{{agent_output}}");
            let rendered_msg = ctx.render_template(&msg_tpl);
            let message = if rendered_msg.trim().is_empty() {
                "Ejecución de arnés Loors finalizada.".to_string()
            } else {
                rendered_msg
            };
            let iter = ctx.iteration;

            let client = match reqwest::Client::builder()
                .user_agent("loors-studio/0.1")
                .timeout(Duration::from_secs(20))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    log(iter, LogStreamKind::Stderr, format!("Error cliente HTTP: {e}"));
                    return NodeStepOutcome::Continue("exec".to_string());
                }
            };

            match channel.as_str() {
                "telegram" => {
                    if bot_token.trim().is_empty() || chat_id.trim().is_empty() {
                        log(
                            iter,
                            LogStreamKind::Stderr,
                            "Notify Telegram omitido: falta botToken o chatId.".to_string(),
                        );
                        return NodeStepOutcome::Continue("exec".to_string());
                    }
                    let url = format!("https://api.telegram.org/bot{}/sendMessage", bot_token.trim());
                    // Split into <= 3900 char chunks to respect Telegram's 4096 char limit
                    let chars: Vec<char> = message.chars().collect();
                    let mut sent_ok = true;
                    for chunk in chars.chunks(3900) {
                        let part: String = chunk.iter().collect();
                        let res = client
                            .post(&url)
                            .json(&serde_json::json!({
                                "chat_id": chat_id.trim(),
                                "text": part,
                                "disable_web_page_preview": false
                            }))
                            .send()
                            .await;
                        match res {
                            Ok(r) if r.status().is_success() => {}
                            Ok(r) => {
                                let status = r.status();
                                let body = r.text().await.unwrap_or_default();
                                log(
                                    iter,
                                    LogStreamKind::Stderr,
                                    format!("Telegram devolvió HTTP {status}: {}", truncate(&body, 220)),
                                );
                                sent_ok = false;
                            }
                            Err(e) => {
                                log(iter, LogStreamKind::Stderr, format!("Error enviando a Telegram: {e}"));
                                sent_ok = false;
                            }
                        }
                    }
                    if sent_ok {
                        log(
                            iter,
                            LogStreamKind::System,
                            format!("Mensaje enviado a Telegram (chat_id={}).", chat_id.trim()),
                        );
                    }
                }
                "discord" => {
                    if webhook_url.trim().is_empty() {
                        log(iter, LogStreamKind::Stderr, "Notify Discord omitido: falta webhookUrl.".to_string());
                    } else {
                        let content = truncate(&message, 1950);
                        match client
                            .post(webhook_url.trim())
                            .json(&serde_json::json!({ "content": content }))
                            .send()
                            .await
                        {
                            Ok(r) if r.status().is_success() => {
                                log(iter, LogStreamKind::System, "Notificación enviada a Discord Webhook.".to_string());
                            }
                            Ok(r) => {
                                log(iter, LogStreamKind::Stderr, format!("Discord Webhook devolvió HTTP {}.", r.status()));
                            }
                            Err(e) => {
                                log(iter, LogStreamKind::Stderr, format!("Error enviando a Discord: {e}"));
                            }
                        }
                    }
                }
                _ => {
                    if webhook_url.trim().is_empty() {
                        log(iter, LogStreamKind::Stderr, "Notify Webhook omitido: falta webhookUrl.".to_string());
                    } else {
                        match client
                            .post(webhook_url.trim())
                            .json(&serde_json::json!({
                                "workflow": graph.name,
                                "iteration": iter,
                                "message": message
                            }))
                            .send()
                            .await
                        {
                            Ok(r) if r.status().is_success() => {
                                log(iter, LogStreamKind::System, "Notificación enviada al Webhook.".to_string());
                            }
                            Ok(r) => {
                                log(iter, LogStreamKind::Stderr, format!("Webhook devolvió HTTP {}.", r.status()));
                            }
                            Err(e) => {
                                log(iter, LogStreamKind::Stderr, format!("Error enviando al Webhook: {e}"));
                            }
                        }
                    }
                }
            }
            NodeStepOutcome::Continue("exec".to_string())
        }

        "http_request" => {
            let method = config_str(node, "method", "GET").to_uppercase();
            let raw_url = config_str(node, "url", "");
            let url = expand_env_vars(&ctx.render_template(&raw_url));
            let headers_raw = config_str(node, "headersJson", "{}");
            let body_tpl = config_str(node, "bodyTemplate", "");
            let iter = ctx.iteration;

            if url.trim().is_empty() {
                return NodeStepOutcome::Finish(
                    RunOutcomeKind::Failed,
                    "HTTP Connector sin URL configurada.".to_string(),
                );
            }

            log(
                iter,
                LogStreamKind::System,
                format!("HTTP {method} {url}"),
            );

            let client = reqwest::Client::builder()
                .user_agent("loors-studio/0.1")
                .timeout(Duration::from_secs(30))
                .build();

            let Ok(client) = client else {
                return NodeStepOutcome::Finish(
                    RunOutcomeKind::Failed,
                    "No se pudo inicializar el cliente HTTP.".to_string(),
                );
            };

            let req_method = reqwest::Method::from_bytes(method.as_bytes())
                .unwrap_or(reqwest::Method::GET);
            let mut req = client.request(req_method, &url);

            let expanded_headers = expand_env_vars(&ctx.render_template(&headers_raw));
            if let Ok(serde_json::Value::Object(map)) =
                serde_json::from_str::<serde_json::Value>(&expanded_headers)
            {
                for (k, v) in map {
                    if let Some(val_str) = v.as_str() {
                        req = req.header(&k, val_str);
                    }
                }
            }

            if !body_tpl.trim().is_empty() && method != "GET" && method != "HEAD" {
                let rendered_body = ctx.render_template(&body_tpl);
                req = req.header("Content-Type", "application/json").body(rendered_body);
            }

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let ok = resp.status().is_success();
                    let text = resp.text().await.unwrap_or_default();
                    ctx.last_http_status = Some(status);
                    ctx.last_http_response = text.clone();
                    if ok && ctx.current_prompt.trim().is_empty() {
                        ctx.current_prompt = text.clone();
                    }
                    log(
                        iter,
                        LogStreamKind::Stdout,
                        format!("HTTP {status} ({} bytes): {}", text.len(), truncate(&text, 220)),
                    );
                    NodeStepOutcome::Continue(if ok { "ok" } else { "error" }.to_string())
                }
                Err(err) => {
                    let msg = format!("Error HTTP: {err}");
                    ctx.last_http_status = None;
                    ctx.last_http_response = msg.clone();
                    log(iter, LogStreamKind::Stderr, msg);
                    NodeStepOutcome::Continue("error".to_string())
                }
            }
        }

        "custom" => {
            let runner = config_str(node, "runner", "shell");
            let branch_on_exit = node
                .data
                .config
                .get("branchOnExit")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let iter = ctx.iteration;

            match runner.as_str() {
                "prompt" => {
                    let tpl = config_str(node, "promptTemplate", "{{current_prompt}}");
                    let rendered = render_with_node_fields(&tpl, ctx, node);
                    let connected_model = find_connected_model_node(graph, node_map, &node.id);
                    let installed_runtime = find_installed_binary(backend.clone());

                    if let Some(model_node) = connected_model {
                        let model_path = config_str(model_node, "modelPath", "");
                        let is_cloud = crate::inference::cloud::is_cloud_model(&model_path);
                        if is_cloud || (installed_runtime.is_some() && Path::new(&model_path).exists()) {
                            let (port_res, model_profile) = if is_cloud {
                                (Ok(None), crate::inference::ModelTuningProfile::default())
                            } else {
                                let rt = installed_runtime.unwrap();
                                let raw_ctx = config_u64(model_node, "contextSize", 0) as u32;
                                let (load_cfg, prof) = resolve_load_config_with_profile(
                                    rt.binary_path,
                                    &model_path,
                                    if raw_ctx > 0 { Some(raw_ctx) } else { None },
                                    opt_config_str(model_node, "mmprojPath"),
                                );
                                (llama.ensure_loaded_cancellable(load_cfg, Some(&cancel)).await.map(Some), prof)
                            };
                            let temperature = config_f32(model_node, "temperature", model_profile.temperature);
                            match port_res {
                                Ok(port_opt) => {
                                    let req = ChatRequest {
                                        messages: vec![ChatMessage {
                                            role: "user".into(),
                                            content: Some(rendered),
                                            tool_calls: None,
                                            tool_call_id: None,
                                            image_paths: Vec::new(),
                                        }],
                                        tools: Vec::new(),
                                        temperature,
                                        top_p: Some(config_f32(model_node, "topP", model_profile.top_p)),
                                        max_tokens: None,
                                        min_p: Some(model_profile.min_p),
                                        repeat_penalty: Some(model_profile.repeat_penalty),
                                        seed: Some(model_profile.seed),
                                        reasoning_effort: None,
                                    };
                                    let custom_res = if let Some(port) = port_opt {
                                        llama.chat_stream(port, req, Some(cancel), |_, _| {}).await
                                    } else {
                                        crate::inference::cloud::chat_stream_cloud(
                                            &model_path,
                                            req,
                                            Some(cancel),
                                            |_, _| {},
                                        )
                                        .await
                                    };
                                    match custom_res {
                                        Ok(resp) => {
                                            ctx.last_custom_output = resp.content.clone();
                                            ctx.last_agent_output = resp.content.clone();
                                            log(iter, LogStreamKind::Agent, resp.content);
                                            return NodeStepOutcome::Continue("ok".to_string());
                                        }
                                        Err(err) if err == "__CANCELLED__" => {
                                            return NodeStepOutcome::Finish(
                                                RunOutcomeKind::Cancelled,
                                                "Custom Prompt detenido por el usuario.".to_string(),
                                            );
                                        }
                                        Err(err) => {
                                            log(iter, LogStreamKind::Stderr, err);
                                            return NodeStepOutcome::Continue("error".to_string());
                                        }
                                    }
                                }
                                Err(err) => {
                                    log(iter, LogStreamKind::Stderr, err);
                                    return NodeStepOutcome::Continue("error".to_string());
                                }
                            }
                        }
                    }
                    log(
                        iter,
                        LogStreamKind::Agent,
                        format!("[Custom Prompt Mock] {}", truncate(&rendered, 240)),
                    );
                    ctx.last_custom_output = rendered;
                    NodeStepOutcome::Continue("ok".to_string())
                }

                "http" => {
                    let method = config_str(node, "httpMethod", "POST").to_uppercase();
                    let url = expand_env_vars(&render_with_node_fields(
                        &config_str(node, "httpUrl", ""),
                        ctx,
                        node,
                    ));
                    let headers_raw = expand_env_vars(&render_with_node_fields(
                        &config_str(node, "httpHeaders", "{}"),
                        ctx,
                        node,
                    ));
                    let body = render_with_node_fields(
                        &config_str(node, "httpBody", ""),
                        ctx,
                        node,
                    );
                    log(iter, LogStreamKind::System, format!("Custom HTTP {method} {url}"));

                    let Ok(client) = reqwest::Client::builder()
                        .user_agent("loors-studio/0.1")
                        .timeout(Duration::from_secs(30))
                        .build()
                    else {
                        return NodeStepOutcome::Continue("error".to_string());
                    };
                    let req_method = reqwest::Method::from_bytes(method.as_bytes())
                        .unwrap_or(reqwest::Method::POST);
                    let mut req = client.request(req_method, &url);
                    if let Ok(serde_json::Value::Object(map)) =
                        serde_json::from_str::<serde_json::Value>(&headers_raw)
                    {
                        for (k, v) in map {
                            if let Some(s) = v.as_str() {
                                req = req.header(&k, s);
                            }
                        }
                    }
                    if !body.trim().is_empty() && method != "GET" && method != "HEAD" {
                        req = req.header("Content-Type", "application/json").body(body);
                    }
                    match req.send().await {
                        Ok(resp) => {
                            let ok = resp.status().is_success();
                            let status = resp.status().as_u16();
                            let text = resp.text().await.unwrap_or_default();
                            ctx.last_http_status = Some(status);
                            ctx.last_http_response = text.clone();
                            ctx.last_custom_output = text.clone();
                            log(
                                iter,
                                LogStreamKind::Stdout,
                                format!("HTTP {status}: {}", truncate(&text, 220)),
                            );
                            NodeStepOutcome::Continue(if ok { "ok" } else { "error" }.to_string())
                        }
                        Err(err) => {
                            log(iter, LogStreamKind::Stderr, format!("Error HTTP: {err}"));
                            NodeStepOutcome::Continue("error".to_string())
                        }
                    }
                }

                _ => {
                    let cmd_tpl = config_str(node, "commandTemplate", "");
                    let cmd_str = render_with_node_fields(&cmd_tpl, ctx, node);
                    if cmd_str.trim().is_empty() {
                        log(iter, LogStreamKind::Stderr, "Custom node sin comando.".into());
                        return NodeStepOutcome::Continue("error".to_string());
                    }
                    let node_ws = effective_workspace(graph, node_map, node, ctx);
                    log(iter, LogStreamKind::System, format!("Custom CLI en '{node_ws}': $ {cmd_str}"));
                    match run_shell_command(&cmd_str, &node_ws, 120, cancel, |s, line| {
                        log(iter, s, line);
                    })
                    .await
                    {
                        Ok((code, stdout, stderr)) => {
                            ctx.last_exit_code = Some(code);
                            ctx.last_stdout = stdout.clone();
                            ctx.last_stderr = stderr;
                            ctx.last_custom_output = stdout;
                            if branch_on_exit {
                                NodeStepOutcome::Continue(
                                    if code == 0 { "ok" } else { "error" }.to_string(),
                                )
                            } else {
                                NodeStepOutcome::Continue("ok".to_string())
                            }
                        }
                        Err(ShellRunError::Cancelled) => NodeStepOutcome::Finish(
                            RunOutcomeKind::Cancelled,
                            "Custom Node detenido por el usuario.".to_string(),
                        ),
                        Err(ShellRunError::Failed(msg)) => {
                            log(iter, LogStreamKind::Stderr, msg);
                            NodeStepOutcome::Continue("error".to_string())
                        }
                    }
                }
            }
        }

        "subgraph" => {
            let iter = ctx.iteration;
            if depth >= 5 {
                let msg = "Profundidad máxima de sub-arneses alcanzada (5).".to_string();
                log(iter, LogStreamKind::Stderr, msg.clone());
                return NodeStepOutcome::Finish(RunOutcomeKind::Failed, msg);
            }

            let wf_path = config_str(node, "workflowPath", "");
            let input_tpl = config_str(node, "inputTemplate", "{{current_prompt}}");
            if wf_path.trim().is_empty() {
                log(
                    iter,
                    LogStreamKind::Stderr,
                    "Sub-arnés sin ruta .loors.json configurada.".into(),
                );
                return NodeStepOutcome::Continue("error".to_string());
            }

            let raw = match std::fs::read_to_string(&wf_path) {
                Ok(r) => r,
                Err(e) => {
                    log(
                        iter,
                        LogStreamKind::Stderr,
                        format!("No se pudo leer sub-arnés '{}': {e}", wf_path),
                    );
                    return NodeStepOutcome::Continue("error".to_string());
                }
            };

            let mut sub_graph: WorkflowGraph = match serde_json::from_str(&raw) {
                Ok(g) => g,
                Err(e) => {
                    log(
                        iter,
                        LogStreamKind::Stderr,
                        format!("JSON de sub-arnés inválido: {e}"),
                    );
                    return NodeStepOutcome::Continue("error".to_string());
                }
            };

            // Inject parent's rendered prompt and workspace into sub-harness task_input
            let injected_prompt = ctx.render_template(&input_tpl);
            if let Some(sub_in) = sub_graph.nodes.iter_mut().find(|n| n.data.kind == "task_input") {
                sub_in.data.config.insert(
                    "prompt".to_string(),
                    serde_json::Value::String(injected_prompt),
                );
                if !ctx.workspace.trim().is_empty() && ctx.workspace != "." {
                    sub_in.data.config.insert(
                        "workspace".to_string(),
                        serde_json::Value::String(ctx.workspace.clone()),
                    );
                }
            }

            log(
                iter,
                LogStreamKind::System,
                format!("Ejecutando sub-arnés '{}' (nivel {})...", sub_graph.name, depth + 1),
            );

            let sub_run_id = format!("{run_id}-sub{}", depth + 1);
            let sub_emit = emit.clone();
            let sub_emitter: EventEmitter = Arc::new(move |ev| {
                // Forward logs and human gates from sub-harness, keep parent run lifecycle clean
                if matches!(
                    ev,
                    EngineEvent::NodeLog { .. } | EngineEvent::HumanGateWaiting { .. }
                ) {
                    sub_emit(ev);
                }
            });
            let sub_outcome = execute_graph_with_depth(
                sub_run_id,
                sub_graph,
                backend.clone(),
                llama.clone(),
                gate_slot.clone(),
                cancel.clone(),
                sub_emitter,
                depth + 1,
                None,
            )
            .await;

            match sub_outcome {
                RunOutcomeKind::Completed => {
                    log(iter, LogStreamKind::System, "Sub-arnés completado con éxito.".into());
                    NodeStepOutcome::Continue("ok".to_string())
                }
                RunOutcomeKind::Cancelled => NodeStepOutcome::Finish(
                    RunOutcomeKind::Cancelled,
                    "Sub-arnés cancelado.".to_string(),
                ),
                other => {
                    log(
                        iter,
                        LogStreamKind::Stderr,
                        format!("Sub-arnés terminó con estado {:?}.", other),
                    );
                    NodeStepOutcome::Continue("error".to_string())
                }
            }
        }

        "parallel_fanout" => {
            let iter = ctx.iteration;
            if depth >= 4 {
                log(iter, LogStreamKind::Stderr, "Profundidad máxima de paralelismo alcanzada.".into());
                return NodeStepOutcome::Continue("merged".to_string());
            }

            let use_worktrees = node
                .data
                .config
                .get("useWorktrees")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let branch_a = find_next_node(&graph.edges, &node.id, "branch_a");
            let branch_b = find_next_node(&graph.edges, &node.id, "branch_b");

            let run_branch = |label: &'static str, start_target: Option<String>| {
                let graph_clone = graph.clone();
                let backend_clone = backend.clone();
                let llama_clone = llama.clone();
                let gate_clone = gate_slot.clone();
                let cancel_clone = cancel.clone();
                let emit_clone = emit.clone();
                let ws_orig = ctx.workspace.clone();
                let prompt_orig = ctx.current_prompt.clone();
                let run_id_owned = run_id.to_string();

                async move {
                    let Some(target_id) = start_target else {
                        return format!("=== Rama {label} ===\n(no conectada)");
                    };

                    // Optional isolated git worktree per branch
                    let mut worktree_dir: Option<std::path::PathBuf> = None;
                    let effective_ws = if use_worktrees {
                        let ts = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis();
                        let wt = std::env::temp_dir().join(format!("loors-wt-{label}-{ts}"));
                        let wt_str = wt.to_string_lossy().to_string();
                        let out = run_git(&ws_orig, &["worktree", "add", "--detach", &wt_str, "HEAD"]).await;
                        if wt.exists() {
                            worktree_dir = Some(wt);
                            wt_str
                        } else {
                            let _ = out;
                            ws_orig.clone()
                        }
                    } else {
                        ws_orig.clone()
                    };

                    let mut sub_graph = graph_clone;
                    // Replace task_input so execution starts from this branch target
                    sub_graph.nodes.retain(|n| n.data.kind != "task_input");
                    let retained: HashSet<_> = sub_graph.nodes.iter().map(|node| node.id.as_str()).collect();
                    sub_graph.edges.retain(|edge| {
                        retained.contains(edge.source.as_str()) && retained.contains(edge.target.as_str())
                    });
                    let synthetic_id = format!("__fanout_in_{label}");
                    sub_graph.nodes.push(GraphNode {
                        id: synthetic_id.clone(),
                        data: crate::engine::types::GraphNodeData {
                            kind: "task_input".into(),
                            title: format!("Fan-Out {label}"),
                            config: HashMap::from([
                                ("workspace".into(), serde_json::json!(effective_ws.clone())),
                                ("prompt".into(), serde_json::json!(prompt_orig)),
                            ]),
                        },
                    });
                    sub_graph.edges.push(GraphEdge {
                        id: format!("__e_fanout_{label}"),
                        source: synthetic_id,
                        source_handle: Some("exec".into()),
                        target: target_id,
                        target_handle: Some("exec".into()),
                    });

                    let branch_logs = Arc::new(Mutex::new(Vec::<String>::new()));
                    let logs_ref = branch_logs.clone();
                    let sub_emitter: EventEmitter = Arc::new(move |ev| {
                        if let EngineEvent::NodeLog { ref node_title, ref text, .. } = ev {
                            if let Ok(mut buf) = logs_ref.lock() {
                                buf.push(format!("[{node_title}] {text}"));
                            }
                        }
                        if matches!(ev, EngineEvent::NodeLog { .. }) {
                            emit_clone(ev);
                        }
                    });

                    let outcome = execute_graph_with_depth(
                        format!("{run_id_owned}-{label}"),
                        sub_graph,
                        backend_clone,
                        llama_clone,
                        gate_clone,
                        cancel_clone,
                        sub_emitter,
                        depth + 1,
                        None,
                    )
                    .await;

                    let diff = if worktree_dir.is_some() {
                        capture_git_diff(&effective_ws).await
                    } else {
                        String::new()
                    };

                    if let Some(ref wt) = worktree_dir {
                        let wt_str = wt.to_string_lossy().to_string();
                        let _ = run_git(&ws_orig, &["worktree", "remove", "--force", &wt_str]).await;
                        let _ = std::fs::remove_dir_all(wt);
                    }

                    let summary_logs = branch_logs
                        .lock()
                        .map(|b| b.iter().rev().take(8).cloned().collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n"))
                        .unwrap_or_default();

                    format!(
                        "=== Rama {label} (outcome: {:?}) ===\nLogs:\n{}\nDiff:\n{}",
                        outcome,
                        truncate(&summary_logs, 1500),
                        truncate(&diff, 2000)
                    )
                }
            };

            log(
                iter,
                LogStreamKind::System,
                format!(
                    "Parallel Fan-Out: ejecutando Rama A y Rama B en paralelo (git worktrees = {use_worktrees})..."
                ),
            );

            let (res_a, res_b) = tokio::join!(run_branch("A", branch_a), run_branch("B", branch_b));
            let merged_report = format!("{res_a}\n\n{res_b}");
            ctx.last_custom_output = merged_report.clone();
            ctx.current_prompt = format!(
                "Compara los resultados de las dos ramas ejecutadas en paralelo y elige o sintetiza la mejor solución:\n\n{merged_report}"
            );
            log(
                iter,
                LogStreamKind::System,
                "Parallel Fan-Out completado. Resultados combinados disponibles en {{custom_output}} y {{current_prompt}}.".into(),
            );
            NodeStepOutcome::Continue("merged".to_string())
        }

        "swarm" => {
            let iter = ctx.iteration;
            let node_ws = effective_workspace(graph, node_map, node, ctx);
            let connected_model = find_connected_model_node(graph, node_map, &node.id);
            let default_model_path = connected_model
                .map(|m| config_str(m, "modelPath", ""))
                .unwrap_or_default();

            let raw_objective = config_str(node, "objectiveTemplate", "{{current_prompt}}");
            let rendered_objective = ctx.render_template(&raw_objective);
            let raw_coord_prompt = config_str(
                node,
                "coordinatorPrompt",
                "Eres el coordinador del enjambre. Descompón el objetivo, asigna tareas claras con @worker, revisa sus entregas y llama a swarm_finish con la síntesis final.",
            );
            let rendered_coord_prompt = ctx.render_template(&raw_coord_prompt);

            let swarm_cfg = crate::engine::swarm::parse_swarm_config(
                &node.data.config,
                rendered_objective,
                rendered_coord_prompt,
            );

            if default_model_path.trim().is_empty()
                && swarm_cfg
                    .workers
                    .iter()
                    .all(|w| w.model_path.trim().is_empty())
            {
                let msg = "El nodo Swarm necesita un nodo 'local_model' conectado al puerto 'model_ref' (o un modelo configurado en sus workers).".to_string();
                log(iter, LogStreamKind::Stderr, msg.clone());
                ctx.last_swarm_output = msg;
                if find_next_node(&graph.edges, &node.id, "error").is_some() {
                    return NodeStepOutcome::Continue("error".to_string());
                }
                return NodeStepOutcome::Finish(RunOutcomeKind::Failed, ctx.last_swarm_output.clone());
            }

            let defaults = if let Some(m) = connected_model {
                let raw_ctx = config_u64(m, "contextSize", 0) as u32;
                let max_tok = config_u64(m, "maxTokens", 0) as u32;
                crate::engine::swarm::SwarmModelDefaults {
                    model_path: default_model_path.clone(),
                    context_size: if raw_ctx > 0 { Some(raw_ctx) } else { None },
                    mmproj_path: opt_config_str(m, "mmprojPath"),
                    temperature: config_f32(m, "temperature", 0.35),
                    top_p: Some(config_f32(m, "topP", 0.95)),
                    max_tokens: if max_tok > 0 { Some(max_tok) } else { None },
                    repeat_penalty: Some(1.05),
                }
            } else {
                let fallback_model = swarm_cfg
                    .workers
                    .iter()
                    .find(|w| !w.model_path.trim().is_empty())
                    .map(|w| w.model_path.clone())
                    .unwrap_or_default();
                crate::engine::swarm::SwarmModelDefaults {
                    model_path: fallback_model,
                    context_size: None,
                    mmproj_path: None,
                    temperature: 0.35,
                    top_p: Some(0.95),
                    max_tokens: None,
                    repeat_penalty: Some(1.05),
                }
            };

            let mut extra_parts = Vec::new();
            let skill_nodes =
                find_connected_nodes_by_handle(graph, node_map, &node.id, "skill_ref");
            for sn in skill_nodes {
                let skill_path = config_str(sn, "skillPath", "");
                let inline_body = config_str(sn, "inlineInstructions", "");
                if !skill_path.trim().is_empty() && Path::new(&skill_path).exists() {
                    if let Ok(sk) = load_skill_file(Path::new(&skill_path)) {
                        extra_parts.push(format!("### Skill: {}\n{}", sk.name, sk.body));
                    }
                } else if !inline_body.trim().is_empty() {
                    extra_parts.push(format!("### Skill: {}\n{}", sn.data.title, inline_body));
                }
            }
            if !ctx.last_rag_context.trim().is_empty() {
                extra_parts.push(format!("### Contexto RAG\n{}", ctx.last_rag_context));
            }
            if !ctx.last_memory.trim().is_empty() {
                extra_parts.push(format!("### Memoria\n{}", ctx.last_memory));
            }
            let extra_context = extra_parts.join("\n\n");

            log(
                iter,
                LogStreamKind::System,
                format!(
                    "Swarm iniciado: {} workers (maxRounds={}, maxConcurrency={}, worktrees={})...",
                    swarm_cfg.workers.len(),
                    swarm_cfg.max_rounds,
                    swarm_cfg.max_concurrency,
                    swarm_cfg.use_worktrees
                ),
            );

            match crate::engine::swarm::run_swarm_node(
                run_id,
                &node.id,
                &node.data.title,
                iter,
                &node_ws,
                swarm_cfg,
                defaults,
                &extra_context,
                backend.clone(),
                llama.clone(),
                cancel.clone(),
                emit.clone(),
            )
            .await
            {
                Ok(res) => {
                    ctx.last_swarm_output = res.output.clone();
                    ctx.last_swarm_board = res.board_transcript;
                    ctx.last_agent_output = res.output.clone();
                    ctx.node_outputs.insert(node.id.clone(), res.output.clone());
                    ctx.node_outputs.insert(node.data.title.clone(), res.output);
                    log(
                        iter,
                        LogStreamKind::System,
                        format!(
                            "Swarm finalizado (converged={}). Resultado disponible en {{{{swarm_output}}}} y {{{{agent_output}}}}.",
                            res.converged
                        ),
                    );
                    if res.converged {
                        NodeStepOutcome::Continue("exec".to_string())
                    } else if find_next_node(&graph.edges, &node.id, "error").is_some() {
                        NodeStepOutcome::Continue("error".to_string())
                    } else {
                        NodeStepOutcome::Finish(RunOutcomeKind::Exhausted, ctx.last_swarm_output.clone())
                    }
                }
                Err(err) if err == "__CANCELLED__" => NodeStepOutcome::Finish(
                    RunOutcomeKind::Cancelled,
                    "Ejecución detenida inmediatamente por el usuario.".to_string(),
                ),
                Err(err) => {
                    log(iter, LogStreamKind::Stderr, format!("Swarm error: {err}"));
                    ctx.last_swarm_output = err.clone();
                    if find_next_node(&graph.edges, &node.id, "error").is_some() {
                        NodeStepOutcome::Continue("error".to_string())
                    } else {
                        NodeStepOutcome::Finish(RunOutcomeKind::Failed, err)
                    }
                }
            }
        }

        "for_each" => {
            let iter = ctx.iteration;
            if !ctx.for_each_queues.contains_key(&node.id) {
                let raw_src = config_str(node, "itemsSource", "{{agent_output}}");
                let rendered = ctx.render_template(&raw_src);
                let split_mode = config_str(node, "splitMode", "auto");
                let max_items = config_u64(node, "maxItems", 0) as usize;
                let mut items = parse_for_each_items(&rendered, &split_mode);
                if max_items > 0 && items.len() > max_items {
                    items.truncate(max_items);
                }
                ctx.for_each_queues.insert(node.id.clone(), items);
                ctx.for_each_index.insert(node.id.clone(), 0);
            }

            let idx = *ctx.for_each_index.get(&node.id).unwrap_or(&0);
            let total = ctx
                .for_each_queues
                .get(&node.id)
                .map(|q| q.len())
                .unwrap_or(0);

            if idx < total {
                let item = ctx.for_each_queues[&node.id][idx].clone();
                ctx.for_each_index.insert(node.id.clone(), idx + 1);
                ctx.current_item = item.clone();
                ctx.current_item_index = idx + 1;
                ctx.current_item_total = total;
                ctx.current_prompt = item.clone();
                log(
                    iter,
                    LogStreamKind::System,
                    format!("For Each [{}/{total}]: {}", idx + 1, truncate(&item, 160)),
                );
                NodeStepOutcome::Continue("each".to_string())
            } else {
                ctx.for_each_queues.remove(&node.id);
                ctx.for_each_index.remove(&node.id);
                log(
                    iter,
                    LogStreamKind::System,
                    format!("For Each completed ({total} items)."),
                );
                NodeStepOutcome::Continue("done".to_string())
            }
        }

        "create_excel" | "create_csv" | "create_word" | "create_pdf" | "read_document" => {
            let workspace = effective_workspace(graph, node_map, node, ctx);
            let path_template = config_str(node, "path", "").replace("{{run_id}}", run_id);
            let path = ctx.render_template(&path_template);
            let kind = node.data.kind.as_str();
            let result = async {
                if kind == "read_document" {
                    let supported = Path::new(&path).extension().is_some_and(|ext| {
                        ["docx", "xlsx", "pdf", "csv"].iter().any(|format| ext.eq_ignore_ascii_case(format))
                    });
                    if !supported {
                        return Err("La ruta debe apuntar a un archivo .docx, .xlsx, .pdf o .csv.".into());
                    }
                    let input = crate::tools::resolve_safe_path(&workspace, &path)?;
                    return crate::tools::docs::read_document_or_text(&input).await;
                }
                let source = ctx.render_template(&config_str(node, "sourceTemplate", "{{agent_output}}"));
                if matches!(kind, "create_excel" | "create_csv") {
                    let rows = parse_json_output(&source, "array")?;
                    let columns: Vec<&str> = node.data.config.get("columnKeys")
                        .and_then(Value::as_str).unwrap_or_default()
                        .lines().map(str::trim).filter(|column| !column.is_empty()).collect();
                    use crate::tools::spreadsheet::{create_table, TableFormat};
                    let format = if kind == "create_csv" { TableFormat::Csv } else { TableFormat::Excel };
                    create_table(
                        &workspace, &path, rows, Some(serde_json::json!(columns)), format,
                    ).await
                } else {
                    let content = parse_json_output(&source, "object")?;
                    let format = if kind == "create_word" { "docx" } else { "pdf" };
                    crate::tools::report::create_report(&workspace, &path, format, content).await
                }
            }.await;
            match result {
                Ok(output) => {
                    ctx.node_outputs.insert(node.id.clone(), output.clone());
                    ctx.node_outputs.insert(node.data.title.clone(), output);
                    NodeStepOutcome::Continue("exec".into())
                }
                Err(error) => {
                    log(ctx.iteration, LogStreamKind::Stderr, error.clone());
                    NodeStepOutcome::Finish(RunOutcomeKind::Failed, error)
                }
            }
        }

        "data_transform" => {
            let source = config_str(node, "sourceTemplate", "{{agent_output}}");
            let root = config_str(node, "rootPointer", "");
            match transform_json(
                &ctx.render_template(&source),
                &root,
                node.data.config.get("fields"),
            ) {
                Ok(output) => {
                    ctx.node_outputs.insert(node.id.clone(), output.clone());
                    ctx.node_outputs.insert(node.data.title.clone(), output);
                    NodeStepOutcome::Continue("exec".into())
                }
                Err(error) => {
                    let message = format!("Transform Data: {error}");
                    log(ctx.iteration, LogStreamKind::Stderr, message.clone());
                    NodeStepOutcome::Finish(RunOutcomeKind::Failed, message)
                }
            }
        }

        "json_validator" => {
            let iter = ctx.iteration;
            let raw_src = config_str(node, "sourceTemplate", "{{agent_output}}");
            let rendered = ctx.render_template(&raw_src);
            let expect_type = config_str(node, "expectType", "object");
            let required_raw = config_str(node, "requiredKeys", "");
            let required_keys: Vec<String> = required_raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            match validate_json_output(&rendered, &expect_type, &required_keys) {
                Ok(clean_json) => {
                    ctx.last_custom_output = clean_json.clone();
                    ctx.node_outputs.insert(node.id.clone(), clean_json.clone());
                    log(
                        iter,
                        LogStreamKind::System,
                        format!("JSON valid ({} chars).", clean_json.len()),
                    );
                    NodeStepOutcome::Continue("valid".to_string())
                }
                Err(err_msg) => {
                    ctx.last_stderr = err_msg.clone();
                    ctx.last_judge_reason = err_msg.clone();
                    log(iter, LogStreamKind::Stderr, format!("JSON invalid: {err_msg}"));
                    if find_next_node(&graph.edges, &node.id, "invalid").is_some() {
                        NodeStepOutcome::Continue("invalid".to_string())
                    } else {
                        NodeStepOutcome::Finish(RunOutcomeKind::Failed, err_msg)
                    }
                }
            }
        }

        "output" => {
            let label = opt_config_str(node, "labelTemplate")
                .map(|template| ctx.render_template(&template))
                .unwrap_or_else(|| config_str(node, "label", "Completado"));
            let summary = format!(
                "{label} tras {} iteración(es) (exit_code={:?}).",
                ctx.iteration, ctx.last_exit_code
            );
            log(ctx.iteration, LogStreamKind::System, summary.clone());
            NodeStepOutcome::Finish(RunOutcomeKind::Completed, summary)
        }

        other => {
            log(
                ctx.iteration,
                LogStreamKind::System,
                format!("Nodo '{other}' omitido en flujo de ejecución."),
            );
            NodeStepOutcome::Continue("exec".to_string())
        }
    }
}

enum ShellRunError {
    Cancelled,
    Failed(String),
}

async fn run_shell_command<F>(
    command_str: &str,
    workspace: &str,
    timeout_sec: u64,
    cancel: &Arc<AtomicBool>,
    on_line: F,
) -> Result<(i32, String, String), ShellRunError>
where
    F: Fn(LogStreamKind, String),
{
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("powershell");
        c.args(["-NoProfile", "-Command", command_str]);
        #[cfg(target_os = "windows")]
        c.creation_flags(0x0800_0000);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command_str]);
        c
    };

    if !workspace.trim().is_empty() && workspace != "." {
        cmd.current_dir(workspace);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .map_err(|e| ShellRunError::Failed(format!("No se pudo lanzar el proceso: {e}")))?;

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(LogStreamKind, String)>();

    if let Some(out) = stdout_pipe {
        let tx_out = tx.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(out).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if tx_out.send((LogStreamKind::Stdout, line)).is_err() {
                    break;
                }
            }
        });
    }

    if let Some(err) = stderr_pipe {
        let tx_err = tx.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(err).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if tx_err.send((LogStreamKind::Stderr, line)).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_sec.max(1));
    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    let mut exit_code: Option<i32> = None;

    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill().await;
            return Err(ShellRunError::Cancelled);
        }

        if tokio::time::Instant::now() >= deadline {
            let _ = child.kill().await;
            return Err(ShellRunError::Failed(format!(
                "Timeout tras {timeout_sec}s ejecutando '{command_str}'."
            )));
        }

        tokio::select! {
            maybe_line = rx.recv() => {
                match maybe_line {
                    Some((stream, line)) => {
                        match stream {
                            LogStreamKind::Stdout => {
                                if !stdout_buf.is_empty() { stdout_buf.push('\n'); }
                                stdout_buf.push_str(&line);
                            }
                            LogStreamKind::Stderr => {
                                if !stderr_buf.is_empty() { stderr_buf.push('\n'); }
                                stderr_buf.push_str(&line);
                            }
                            _ => {}
                        }
                        on_line(stream, line);
                    }
                    None => {
                        if exit_code.is_some() {
                            break;
                        }
                        match child.wait().await {
                            Ok(status) => {
                                exit_code = Some(status.code().unwrap_or(-1));
                                break;
                            }
                            Err(e) => {
                                return Err(ShellRunError::Failed(format!("Error esperando proceso: {e}")));
                            }
                        }
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(80)) => {
                if let Ok(Some(status)) = child.try_wait() {
                    exit_code = Some(status.code().unwrap_or(-1));
                    while let Ok((stream, line)) = rx.try_recv() {
                        match stream {
                            LogStreamKind::Stdout => {
                                if !stdout_buf.is_empty() { stdout_buf.push('\n'); }
                                stdout_buf.push_str(&line);
                            }
                            LogStreamKind::Stderr => {
                                if !stderr_buf.is_empty() { stderr_buf.push('\n'); }
                                stderr_buf.push_str(&line);
                            }
                            _ => {}
                        }
                        on_line(stream, line);
                    }
                    break;
                }
            }
        }
    }

    Ok((exit_code.unwrap_or(-1), stdout_buf, stderr_buf))
}

pub fn evaluate_condition(expr: &str, ctx: &ExecutionContext) -> bool {
    let trimmed = expr.trim();
    let code = ctx.last_exit_code.unwrap_or(-1);

    if let Some(rhs) = trimmed.strip_prefix("exit_code ==") {
        if let Ok(expected) = rhs.trim().parse::<i32>() {
            return code == expected;
        }
    }
    if let Some(rhs) = trimmed.strip_prefix("exit_code !=") {
        if let Ok(expected) = rhs.trim().parse::<i32>() {
            return code != expected;
        }
    }
    if let Some(rhs) = trimmed.strip_prefix("iteration <=") {
        if let Ok(expected) = rhs.trim().parse::<u32>() {
            return ctx.iteration <= expected;
        }
    }
    if let Some(rhs) = trimmed.strip_prefix("iteration <") {
        if let Ok(expected) = rhs.trim().parse::<u32>() {
            return ctx.iteration < expected;
        }
    }
    if let Some(rhs) = trimmed.strip_prefix("iteration >=") {
        if let Ok(expected) = rhs.trim().parse::<u32>() {
            return ctx.iteration >= expected;
        }
    }
    if let Some(rhs) = trimmed.strip_prefix("stdout contains") {
        let needle = rhs.trim().trim_matches('"').trim_matches('\'');
        return ctx.last_stdout.contains(needle);
    }
    if let Some(rhs) = trimmed.strip_prefix("stderr contains") {
        let needle = rhs.trim().trim_matches('"').trim_matches('\'');
        return ctx.last_stderr.contains(needle);
    }
    if let Some(rhs) = trimmed.strip_prefix("agent_output contains") {
        let needle = rhs.trim().trim_matches('"').trim_matches('\'');
        return ctx.last_agent_output.contains(needle);
    }

    code == 0
}

fn validate_graph_connections(
    graph: &WorkflowGraph,
    nodes: &HashMap<String, &GraphNode>,
) -> Result<(), String> {
    if nodes.len() != graph.nodes.len() {
        return Err("El grafo contiene IDs de nodo duplicados.".into());
    }
    let mut flow_outputs = HashSet::new();
    for edge in &graph.edges {
        if !nodes.contains_key(&edge.source) || !nodes.contains_key(&edge.target) {
            return Err(format!("Conexión inválida: {} → {}", edge.source, edge.target));
        }
        let source = edge.source_handle.as_deref().unwrap_or("exec");
        let target = edge.target_handle.as_deref().unwrap_or("exec");
        if (source.ends_with("_ref") || target.ends_with("_ref")) && source != target {
            return Err(format!("Puertos incompatibles: {source} → {target}"));
        }
        if !source.ends_with("_ref") && !flow_outputs.insert((edge.source.as_str(), source)) {
            return Err(format!("El puerto '{source}' de '{}' tiene varias salidas.", edge.source));
        }
    }
    Ok(())
}

fn find_next_node(edges: &[GraphEdge], source_id: &str, source_handle: &str) -> Option<String> {
    edges
        .iter()
        .find(|e| {
            e.source == source_id
                && e.source_handle
                    .as_deref()
                    .map(|h| h == source_handle)
                    .unwrap_or(source_handle == "exec")
        })
        .map(|e| e.target.clone())
}

fn find_connected_model_node<'a>(
    graph: &WorkflowGraph,
    node_map: &HashMap<String, &'a GraphNode>,
    agent_id: &str,
) -> Option<&'a GraphNode> {
    let edge = graph.edges.iter().find(|e| {
        e.target == agent_id && e.target_handle.as_deref() == Some("model_ref")
    })?;
    node_map.get(&edge.source).copied()
}

fn find_connected_nodes_by_handle<'a>(
    graph: &WorkflowGraph,
    node_map: &HashMap<String, &'a GraphNode>,
    target_id: &str,
    target_handle: &str,
) -> Vec<&'a GraphNode> {
    graph
        .edges
        .iter()
        .filter(|e| e.target == target_id && e.target_handle.as_deref() == Some(target_handle))
        .filter_map(|e| node_map.get(&e.source).copied())
        .collect()
}

fn mcp_config_from_node(node: &GraphNode) -> McpServerConfig {
    let name = config_str(node, "serverName", &node.data.title);
    let command = config_str(node, "command", "");
    let args_raw = config_str(node, "args", "");
    let args: Vec<String> = args_raw
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let env_raw = config_str(node, "envJson", "{}");
    let env: HashMap<String, String> = serde_json::from_str::<HashMap<String, String>>(&env_raw)
        .unwrap_or_default();
    let allowed_raw = config_str(node, "allowedTools", "");
    let allowed_tools: Vec<String> = allowed_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    McpServerConfig {
        name,
        command,
        args,
        env,
        allowed_tools,
    }
}

fn opt_config_str(node: &GraphNode, key: &str) -> Option<String> {
    node.data
        .config
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn config_str(node: &GraphNode, key: &str, default: &str) -> String {
    node.data
        .config
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

fn effective_workspace(
    graph: &WorkflowGraph,
    node_map: &HashMap<String, &GraphNode>,
    node: &GraphNode,
    ctx: &ExecutionContext,
) -> String {
    // 1. Pluggable Workspace node connected via `workspace_ref` handle
    let ws_nodes = find_connected_nodes_by_handle(graph, node_map, &node.id, "workspace_ref");
    if let Some(wn) = ws_nodes.first() {
        let p = config_str(wn, "path", "");
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return ctx.render_template(trimmed);
        }
    }
    // 2. Per-node override field in config
    let custom = config_str(node, "workspace", "");
    let trimmed = custom.trim();
    if trimmed.is_empty() {
        ctx.workspace.clone()
    } else {
        ctx.render_template(trimmed)
    }
}

fn config_u64(node: &GraphNode, key: &str, default: u64) -> u64 {
    node.data
        .config
        .get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
}

fn config_f32(node: &GraphNode, key: &str, default: f32) -> f32 {
    node.data
        .config
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn config_str_list(node: &GraphNode, key: &str) -> Vec<String> {
    node.data
        .config
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

pub async fn capture_git_diff(workspace: &str) -> String {
    let diff = run_git(workspace, &["diff", "--no-color"]).await;
    if diff.trim().is_empty() {
        run_git(workspace, &["status", "--short"]).await
    } else {
        diff
    }
}

async fn run_git(workspace: &str, args: &[&str]) -> String {
    let mut cmd = Command::new("git");
    cmd.args(args);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x0800_0000);
    if !workspace.trim().is_empty() && workspace != "." {
        cmd.current_dir(workspace);
    }
    match cmd.output().await {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            if stdout.trim().is_empty() {
                String::from_utf8_lossy(&out.stderr).to_string()
            } else {
                stdout
            }
        }
        Err(e) => format!("git no disponible: {e}"),
    }
}

fn resolve_workspace_file(workspace: &str, rel: &str) -> Result<std::path::PathBuf, String> {
    let base = if workspace.trim().is_empty() || workspace == "." {
        std::env::current_dir().map_err(|e| format!("Error leyendo directorio actual: {e}"))?
    } else {
        std::path::PathBuf::from(workspace)
    };
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err("Memory File solo acepta rutas relativas dentro del workspace.".to_string());
    }
    for comp in rel_path.components() {
        if matches!(comp, std::path::Component::ParentDir) {
            return Err("Ruta rechazada: no se permite salir del workspace con '..'.".to_string());
        }
    }
    Ok(base.join(rel_path))
}

pub fn parse_judge_json(raw: &str) -> (bool, String) {
    parse_judge_verdict(raw, false)
}

fn parse_judge_verdict(raw: &str, strict: bool) -> (bool, String) {
    let trimmed = raw.trim();
    let candidate = if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        &trimmed[start..=end]
    } else {
        trimmed
    };

    if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
        let decision = val
            .get("decision")
            .and_then(|v| v.as_str())
            .unwrap_or("fail")
            .to_lowercase();
        let reason = val
            .get("reason")
            .and_then(|v| v.as_str())
            .filter(|reason| !strict || !reason.trim().is_empty());
        let passed = if strict {
            decision == "pass" && reason.is_some()
        } else {
            matches!(decision.as_str(), "pass" | "keep" | "done" | "approved" | "true")
        };
        let reason = reason
            .unwrap_or("Sin explicación.")
            .to_string();
        return (passed, reason);
    }

    let lower = trimmed.to_lowercase();
    let passed = !strict
        && (lower.contains("\"pass\"") || lower.contains("\"keep\"") || lower.starts_with("pass"));
    (passed, truncate(trimmed, 220))
}

fn expand_env_vars(input: &str) -> String {
    let mut out = input.to_string();
    while let Some(start) = out.find("${") {
        let Some(rel_end) = out[start + 2..].find('}') else {
            break;
        };
        let end = start + 2 + rel_end;
        let var_name = &out[start + 2..end];
        let val = std::env::var(var_name).unwrap_or_default();
        out.replace_range(start..=end, &val);
    }
    out
}

fn render_with_node_fields(template: &str, ctx: &ExecutionContext, node: &GraphNode) -> String {
    let mut out = ctx.render_template(template);
    for (k, v) in &node.data.config {
        let replacement = match v {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => continue,
        };
        out = out.replace(&format!("{{{{{k}}}}}"), &replacement);
    }
    out
}

fn open_run_log_file(run_id: &str) -> Option<(std::fs::File, std::fs::File)> {
    let dir = dirs::data_local_dir()?
        .join("loors")
        .join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let latest = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dir.join("latest.log"))
        .ok()?;
    let per_run = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dir.join(format!("{run_id}.log")))
        .ok()?;
    Some((latest, per_run))
}

fn write_event_to_log(
    log_files: &Arc<Mutex<Option<(std::fs::File, std::fs::File)>>>,
    ev: &EngineEvent,
) {
    use std::io::Write;
    let line = match ev {
        EngineEvent::RunStarted { run_id, workflow_name } => {
            format!("[START] run={run_id} workflow={workflow_name}\n")
        }
        EngineEvent::NodeStatus { node_id, status, iteration, message, .. } => {
            format!(
                "[STATUS #{iteration}] {node_id} => {:?}{}\n",
                status,
                message.as_deref().map(|m| format!(" ({m})")).unwrap_or_default()
            )
        }
        EngineEvent::NodeResult { .. } => return,
        EngineEvent::NodeLog { node_title, iteration, stream, text, .. } => {
            format!("[#{iteration} | {node_title} | {:?}] {text}\n", stream)
        }
        EngineEvent::IterationChanged { iteration, max_iterations, .. } => {
            format!("[ITER] {iteration}/{max_iterations}\n")
        }
        EngineEvent::ModelState { loaded, model_name, context_size } => {
            format!("[MODEL] loaded={loaded} name={model_name:?} ctx={context_size:?}\n")
        }
        EngineEvent::HumanGateWaiting { node_title, iteration, message, .. } => {
            format!("[HUMAN_GATE #{iteration}] {node_title}: {message}\n")
        }
        EngineEvent::SwarmAgentStatus { agent_id, status, summary, .. } => {
            format!("[SWARM_STATUS @{agent_id}] {status} — {summary}\n")
        }
        EngineEvent::SwarmBoardMessage { seq, from, mentions, content, .. } => {
            let to = if mentions.is_empty() {
                String::new()
            } else {
                format!(" -> {}", mentions.iter().map(|m| format!("@{m}")).collect::<Vec<_>>().join(" "))
            };
            format!("[SWARM_BOARD #{seq} @{from}{to}] {content}\n")
        }
        EngineEvent::RunFinished { outcome, total_iterations, summary, .. } => {
            format!("[FINISH] {:?} iters={total_iterations} — {summary}\n", outcome)
        }
    };

    if let Ok(mut guard) = log_files.lock() {
        if let Some((ref mut latest, ref mut per_run)) = *guard {
            let _ = latest.write_all(line.as_bytes());
            let _ = latest.flush();
            let _ = per_run.write_all(line.as_bytes());
            let _ = per_run.flush();
        }
    }
}

fn record_node_output(ctx: &mut ExecutionContext, node: &GraphNode) {
    let val = match node.data.kind.as_str() {
        "agent" => Some(ctx.last_agent_output.clone()),
        "shell" => Some(ctx.last_stdout.clone()),
        "llm_judge" => Some(ctx.last_judge_reason.clone()),
        "memory_file" => Some(ctx.last_memory.clone()),
        "knowledge_rag" => Some(ctx.last_rag_context.clone()),
        "http_request" => Some(ctx.last_http_response.clone()),
        "custom" | "custom_node" | "json_validator" => Some(ctx.last_custom_output.clone()),
        "for_each" => Some(ctx.current_item.clone()),
        "git_checkpoint" if !matches!(
            config_str(node, "mode", "diff").as_str(),
            "delivery_branch" | "delivery_publish"
        ) => Some(ctx.last_git_diff.clone()),
        _ => None,
    };
    if let Some(v) = val {
        ctx.node_outputs.insert(node.id.clone(), v.clone());
        ctx.node_outputs.insert(node.data.title.clone(), v);
    }
}

pub fn parse_inline_tool_calls(content: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let Ok(re) = regex::Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>") else {
        return out;
    };
    for (idx, caps) in re.captures_iter(content).enumerate() {
        let raw = caps[1].trim();
        let Ok(val) = serde_json::from_str::<serde_json::Value>(raw) else {
            continue;
        };
        let Some(name) = val.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let args_str = match val.get("arguments") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".to_string(),
        };
        out.push(ToolCall {
            id: format!("rescued_{idx}"),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args_str,
            },
        });
    }
    out
}

pub fn compaction_threshold(context_size: u32, configured: u32) -> u32 {
    match (context_size, configured) {
        (0, tokens) => tokens,
        (context, 0) => ((u64::from(context) * 3 + 2) / 4) as u32,
        (context, tokens) => context.min(tokens),
    }
}

pub fn chat_tool_bytes(tools: &[Value]) -> usize {
    if tools.is_empty() {
        return 0;
    }
    serde_json::to_vec(tools).map_or(0, |bytes| bytes.len())
}

fn chat_prompt_bytes(messages: &[ChatMessage], tool_bytes: usize) -> usize {
    tool_bytes + messages.iter().map(|message| {
        let content = message.content.as_ref().map_or(0, String::len);
        let calls = message.tool_calls.as_ref().map_or(0, |calls| {
            calls.iter().map(|call| {
                call.id.len() + call.function.name.len() + call.function.arguments.len() + 24
            }).sum::<usize>()
        });
        // Allow for role markers and separators in the chat template.
        16 + content + calls + message.tool_call_id.as_ref().map_or(0, String::len)
    }).sum::<usize>()
}

pub fn chat_prompt_tokens(messages: &[ChatMessage], tool_bytes: usize) -> usize {
    chat_prompt_bytes(messages, tool_bytes).saturating_mul(10).div_ceil(26)
}

pub fn compact_agent_messages(messages: &mut [ChatMessage], tool_bytes: usize, threshold: u32) -> bool {
    compact_tool_outputs(messages, tool_bytes, threshold, |text| truncate(text, 180))
}

pub(crate) fn compact_local_agent_messages(
    messages: &mut [ChatMessage],
    tool_bytes: usize,
    threshold: u32,
) -> bool {
    compact_tool_outputs(messages, tool_bytes, threshold, |text| local_tool_preview(text, 180))
}

pub(crate) fn local_tool_preview(text: &str, head_chars: usize) -> String {
    let Some((head_end, _)) = text.char_indices().nth(head_chars) else {
        return text.to_string();
    };
    let mut tail_start = text.len().saturating_sub(320);
    if tail_start <= head_end {
        return text.to_string();
    }
    while !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!("{}\n[…]\n{}", &text[..head_end], &text[tail_start..])
}

fn compact_tool_outputs(
    messages: &mut [ChatMessage],
    tool_bytes: usize,
    threshold: u32,
    preview: impl Fn(&str) -> String,
) -> bool {
    // Estimate one token per 2.6 UTF-8 bytes and aim for 60% of the trigger.
    let trigger_budget = (threshold as usize).saturating_mul(26) / 10;
    let target_budget = trigger_budget.saturating_mul(3) / 5;
    let mut total_bytes = chat_prompt_bytes(messages, tool_bytes);
    if threshold == 0 || total_bytes <= trigger_budget {
        return false;
    }

    let recent_start = messages.len().saturating_sub(4).max(2).min(messages.len());
    let (older, recent) = messages.split_at_mut(recent_start);
    let mut compacted_any = false;
    // Preserve the initial prompt and recent conversation. Shorten recent tool output
    // only if older output alone could not bring the prompt below the trigger.
    for (candidates, budget) in [(older, target_budget), (recent, trigger_budget)] {
        for msg in candidates.iter_mut().filter(|message| message.role == "tool") {
            if total_bytes <= budget {
                break;
            }
            if let Some(text) = &mut msg.content {
                if text.contains("[salida de herramienta compactada]") {
                    continue;
                }
                let replacement =
                    format!("{}\n[salida de herramienta compactada]", preview(text));
                if replacement.len() < text.len() {
                    total_bytes -= text.len() - replacement.len();
                    *text = replacement;
                    compacted_any = true;
                }
            }
        }
        if total_bytes <= trigger_budget {
            break;
        }
    }
    compacted_any
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max).collect();
        format!("{prefix}…")
    }
}

fn parse_for_each_items(raw: &str, split_mode: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if split_mode != "lines" {
        if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
            if end > start {
                if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&trimmed[start..=end]) {
                    return arr
                        .into_iter()
                        .map(|v| match v {
                            Value::String(s) => s,
                            other => other.to_string(),
                        })
                        .filter(|s| !s.trim().is_empty())
                        .collect();
                }
            }
        }
        if split_mode == "json_array" {
            return Vec::new();
        }
    }
    trimmed
        .lines()
        .map(|l| l.trim().trim_start_matches("- ").trim_start_matches("* ").to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn parse_json_output(raw: &str, expect_type: &str) -> Result<Value, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Empty output — expected valid JSON.".to_string());
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        Ok(value)
    } else {
        let obj_span = trimmed.find('{').zip(trimmed.rfind('}')).filter(|(s, e)| e > s);
        let arr_span = trimmed.find('[').zip(trimmed.rfind(']')).filter(|(s, e)| e > s);
        let chosen = match expect_type {
            "array" => arr_span.or(obj_span),
            "object" => obj_span.or(arr_span),
            _ => match (obj_span, arr_span) {
                (Some(o), Some(a)) => if o.0 <= a.0 { Some(o) } else { Some(a) },
                (Some(o), None) => Some(o),
                (None, Some(a)) => Some(a),
                (None, None) => None,
            },
        };
        let Some((s, e)) = chosen else {
            return Err("No JSON object or array found in output.".to_string());
        };
        serde_json::from_str::<Value>(&trimmed[s..=e])
            .map_err(|err| format!("Invalid JSON syntax: {err}"))
    }
}

fn transform_json(raw: &str, root: &str, fields: Option<&Value>) -> Result<String, String> {
    #[derive(serde::Deserialize)]
    struct Field {
        pointer: String,
        name: String,
    }

    let input = parse_json_output(raw, "any")?;
    let selected = input
        .pointer(root)
        .ok_or_else(|| format!("Root path {root:?} was not found."))?;
    let fields = match fields {
        Some(fields) => serde_json::from_value::<Vec<Field>>(fields.clone())
            .map_err(|error| format!("Invalid field rules: {error}"))?,
        None => Vec::new(),
    };
    if fields.is_empty() {
        return Ok(selected.to_string());
    }

    let mut names = HashSet::new();
    for field in &fields {
        if field.name.trim().is_empty() {
            return Err("Each field rule needs an output name.".into());
        }
        if !names.insert(&field.name) {
            return Err(format!("Duplicate output name {:?}.", field.name));
        }
        if !field.pointer.is_empty() && !field.pointer.starts_with('/') {
            return Err(format!("Field path {:?} must start with '/'.", field.pointer));
        }
    }

    let project = |value: &Value| {
        let mut output = serde_json::Map::new();
        for field in &fields {
            let value = value.pointer(&field.pointer).ok_or_else(|| {
                format!("Field path {:?} was not found for {:?}.", field.pointer, field.name)
            })?;
            output.insert(field.name.clone(), value.clone());
        }
        Ok::<_, String>(Value::Object(output))
    };
    let output = match selected {
        Value::Array(items) => Value::Array(
            items.iter().enumerate().map(|(index, item)| {
                project(item).map_err(|error| format!("Item {}: {error}", index + 1))
            }).collect::<Result<Vec<_>, _>>()?,
        ),
        value => project(value)?,
    };
    Ok(output.to_string())
}

fn validate_json_output(
    raw: &str,
    expect_type: &str,
    required_keys: &[String],
) -> Result<String, String> {
    let slice = parse_json_output(raw, expect_type)?;

    match expect_type {
        "object" if !slice.is_object() => {
            return Err("Expected a JSON object {...}.".to_string());
        }
        "array" if !slice.is_array() => {
            return Err("Expected a JSON array [...].".to_string());
        }
        _ => {}
    }

    if let Some(obj) = slice.as_object() {
        let missing: Vec<&str> = required_keys
            .iter()
            .filter(|k| match obj.get(k.as_str()) {
                None | Some(Value::Null) => true,
                Some(Value::String(s)) => s.trim().is_empty(),
                _ => false,
            })
            .map(|k| k.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(format!("Missing required JSON keys: {}", missing.join(", ")));
        }
    } else if !required_keys.is_empty() && expect_type == "object" {
        return Err("Expected JSON object with required keys.".to_string());
    }

    serde_json::to_string(&slice).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn document_nodes_pass_the_created_path_to_the_reader() {
        let workspace = tempfile::tempdir().unwrap();
        let node = |id: &str, kind: &str, config: Value| GraphNode {
            id: id.into(),
            data: crate::engine::types::GraphNodeData {
                kind: kind.into(),
                title: id.into(),
                config: serde_json::from_value(config).unwrap(),
            },
        };
        let graph = WorkflowGraph {
            name: "Document round trip".into(),
            nodes: vec![
                node("input", "task_input", json!({ "workspace": workspace.path() })),
                node("excel", "create_excel", json!({
                    "path": "datos-{{run_id}}.xlsx", "sourceTemplate": "[{\"total\":42}]"
                })),
                node("reader", "read_document", json!({ "path": "{{node:excel}}" })),
                node("csv", "create_csv", json!({
                    "path": "datos.csv", "sourceTemplate": "[{\"total\":42}]"
                })),
                node("csv_reader", "read_document", json!({ "path": "{{node:csv}}" })),
                node("end", "output", json!({})),
            ],
            edges: [
                ("input", "excel"),
                ("excel", "reader"),
                ("reader", "csv"),
                ("csv", "csv_reader"),
                ("csv_reader", "end"),
            ]
                .into_iter()
                .map(|(source, target)| GraphEdge {
                    id: format!("{source}-{target}"),
                    source: source.into(),
                    source_handle: Some("exec".into()),
                    target: target.into(),
                    target_handle: Some("exec".into()),
                }).collect(),
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let outcome = execute_graph(
            "document-nodes-test".into(),
            graph,
            RuntimeBackendKind::Cpu,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(move |event| captured.lock().unwrap().push(event)),
        ).await;
        assert!(matches!(outcome, RunOutcomeKind::Completed));
        assert!(workspace.path().join("datos-document-nodes-test.xlsx").is_file());
        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(event,
            EngineEvent::NodeResult { node_id, value_preview: Some(preview), .. }
            if node_id == "reader" && preview.contains("| total |") && preview.contains("| 42 |")
        )));
        assert!(events.iter().any(|event| matches!(event,
            EngineEvent::NodeResult { node_id, value_preview: Some(preview), .. }
            if node_id == "csv_reader" && preview == "total\r\n42\r\n"
        )));
    }

    #[tokio::test]
    async fn data_transform_projects_json_and_feeds_for_each() {
        let node = |id: &str, kind: &str, config: Value| GraphNode {
            id: id.into(),
            data: crate::engine::types::GraphNodeData {
                kind: kind.into(),
                title: id.into(),
                config: serde_json::from_value(config).unwrap(),
            },
        };
        let edge = |from: &str, port: &str, to: &str| GraphEdge {
            id: format!("{from}-{to}"),
            source: from.into(),
            source_handle: Some(port.into()),
            target: to.into(),
            target_handle: Some("exec".into()),
        };
        let graph = WorkflowGraph {
            name: "Transform data".into(),
            nodes: vec![
                node("input", "task_input", json!({})),
                node("transform", "data_transform", json!({
                    "sourceTemplate": r#"{"results":[{"id":42,"metadata":{"title":"Informe"}},{"id":43,"metadata":{"title":null}}]}"#,
                    "rootPointer": "/results",
                    "fields": [{ "pointer": "/metadata/title", "name": "label" }]
                })),
                node("iterate", "for_each", json!({
                    "itemsSource": "{{node:transform}}",
                    "splitMode": "json_array"
                })),
                node("output", "output", json!({ "labelTemplate": "{{item}}" })),
            ],
            edges: vec![
                edge("input", "exec", "transform"),
                edge("transform", "exec", "iterate"),
                edge("iterate", "each", "output"),
            ],
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let outcome = execute_graph(
            "data-transform-test".into(),
            graph,
            RuntimeBackendKind::Cpu,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(move |event| captured.lock().unwrap().push(event)),
        ).await;

        assert!(matches!(outcome, RunOutcomeKind::Completed));
        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(event,
            EngineEvent::NodeResult { node_id, value_preview: Some(preview), .. }
            if node_id == "transform" && preview == r#"[{"label":"Informe"},{"label":null}]"#
        )));
        assert!(events.iter().any(|event| matches!(event,
            EngineEvent::RunFinished { summary, .. } if summary.starts_with(r#"{"label":"Informe"}"#)
        )));
    }

    #[test]
    fn data_transform_reports_missing_fields_without_treating_null_as_missing() {
        let fields = json!([{ "pointer": "/title", "name": "title" }]);
        assert_eq!(
            transform_json(r#"[{"title":null},{}]"#, "", Some(&fields)).unwrap_err(),
            "Item 2: Field path \"/title\" was not found for \"title\"."
        );
        assert_eq!(
            transform_json(r#"{"results":[]}"#, "/results", None).unwrap(),
            "[]"
        );
    }

    #[test]
    fn resolves_compaction_threshold_in_tokens() {
        assert_eq!(compaction_threshold(4096, 0), 3072);
        assert_eq!(compaction_threshold(4096, 3500), 3500);
        assert_eq!(compaction_threshold(4096, u32::MAX), 4096);
        assert_eq!(compaction_threshold(0, 3500), 3500);
    }

    #[test]
    fn tool_schemas_can_trigger_compaction_in_a_short_history() {
        let mut messages: Vec<_> = ["system", "user", "tool"]
            .into_iter()
            .map(|role| ChatMessage {
                role: role.into(),
                content: Some(if role == "tool" { "x".repeat(1000) } else { "keep".into() }),
                tool_calls: None,
                tool_call_id: None,
                image_paths: Vec::new(),
            })
            .collect();
        assert!(!compact_agent_messages(&mut messages, 0, 500));
        assert!(compact_agent_messages(&mut messages, 500, 500));
        assert!(chat_prompt_tokens(&messages, 500) < 500);
        assert_eq!(messages[0].content.as_deref(), Some("keep"));
        assert_eq!(messages[1].content.as_deref(), Some("keep"));
        assert!(!compact_agent_messages(&mut messages, 500, 500));
    }

    #[test]
    fn local_compaction_keeps_terminal_diagnostics() {
        let output = format!(
            "exit_code: 1\n{}\nerror[E0308]: tipos incompatibles",
            "compilación de dependencias\n".repeat(500)
        );
        let mut original = vec![ChatMessage {
            role: "tool".into(),
            content: Some(local_tool_preview(&output, 7680)),
            tool_calls: None,
            tool_call_id: Some("check-1".into()),
            image_paths: Vec::new(),
        }];
        let mut local = original.clone();
        let before = chat_prompt_tokens(&original, 0);
        assert!(compact_agent_messages(&mut original, 0, 500));
        assert!(compact_local_agent_messages(&mut local, 0, 500));
        let local_tokens = chat_prompt_tokens(&local, 0);
        assert!(local_tokens < 500);
        assert!(!original[0].content.as_deref().unwrap().contains("error[E0308]"));
        let preview = local[0].content.as_deref().unwrap();
        assert!(preview.starts_with("exit_code: 1\n"));
        assert!(preview.contains("error[E0308]: tipos incompatibles"));
        assert_eq!(local[0].tool_call_id.as_deref(), Some("check-1"));
        println!("estimated prompt tokens: before={before}, current={}, local={local_tokens}",
            chat_prompt_tokens(&original, 0));
    }

    #[test]
    fn parses_for_each_json_and_lines() {
        assert_eq!(
            parse_for_each_items("Here are items:\n```json\n[\"a\", \"b\", \"c\"]\n```", "auto"),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            parse_for_each_items("- first\n- second\n", "lines"),
            vec!["first", "second"]
        );
    }

    #[test]
    fn validates_json_output_and_required_keys() {
        let ok = validate_json_output(
            "```json\n{\"title\":\"Hi\",\"url\":\"https://x.com\"}\n```",
            "object",
            &["title".to_string(), "url".to_string()],
        );
        assert!(ok.is_ok());

        let missing = validate_json_output(
            "{\"title\":\"Hi\"}",
            "object",
            &["title".to_string(), "url".to_string()],
        );
        assert!(missing.is_err());
    }
    use crate::engine::types::GraphNodeData;
    use crate::inference::compute_auto_context_size;
    use serde_json::json;

    #[tokio::test]
    async fn cyclic_loop_guard_retries_until_exhausted() {
        let graph = WorkflowGraph {
            name: "test-loop".to_string(),
            nodes: vec![
                GraphNode {
                    id: "in".into(),
                    data: GraphNodeData {
                        kind: "task_input".into(),
                        title: "Input".into(),
                        config: HashMap::from([
                            ("workspace".into(), json!(".")),
                            ("prompt".into(), json!("Fix bug")),
                        ]),
                    },
                },
                GraphNode {
                    id: "cond".into(),
                    data: GraphNodeData {
                        kind: "condition".into(),
                        title: "Always fail".into(),
                        config: HashMap::from([("expression".into(), json!("exit_code == 0"))]),
                    },
                },
                GraphNode {
                    id: "guard".into(),
                    data: GraphNodeData {
                        kind: "loop_guard".into(),
                        title: "Guard".into(),
                        config: HashMap::from([("maxIterations".into(), json!(3))]),
                    },
                },
            ],
            edges: vec![
                GraphEdge {
                    id: "e1".into(),
                    source: "in".into(),
                    source_handle: Some("exec".into()),
                    target: "cond".into(),
                    target_handle: Some("exec".into()),
                },
                GraphEdge {
                    id: "e2".into(),
                    source: "cond".into(),
                    source_handle: Some("false".into()),
                    target: "guard".into(),
                    target_handle: Some("exec".into()),
                },
                GraphEdge {
                    id: "e3".into(),
                    source: "guard".into(),
                    source_handle: Some("retry".into()),
                    target: "cond".into(),
                    target_handle: Some("exec".into()),
                },
            ],
        };

        let events = Arc::new(Mutex::new(Vec::<EngineEvent>::new()));
        let events_clone = events.clone();

        let test_emitter: EventEmitter = Arc::new(move |ev| {
            events_clone.lock().unwrap().push(ev);
        });
        let outcome = execute_graph(
            "test-run".into(),
            graph,
            RuntimeBackendKind::Auto,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            test_emitter,
        )
        .await;

        assert!(matches!(outcome, RunOutcomeKind::Exhausted));
        let collected = events.lock().unwrap();
        let iter_changes = collected
            .iter()
            .filter(|e| matches!(e, EngineEvent::IterationChanged { .. }))
            .count();
        assert_eq!(iter_changes, 2); // iter 1 -> 2 -> 3, then exhausted on 3rd attempt
    }

    #[test]
    fn judge_json_parser_extracts_decision_and_reason() {
        let (ok, reason) = parse_judge_json(
            "```json\n{\"decision\": \"keep\", \"reason\": \"Tests pasan y el diff es limpio\"}\n```",
        );
        assert!(ok);
        assert_eq!(reason, "Tests pasan y el diff es limpio");

        let (ok2, reason2) = parse_judge_json("{\"decision\":\"fail\",\"reason\":\"Falta validar null\"}");
        assert!(!ok2);
        assert_eq!(reason2, "Falta validar null");
    }

    #[test]
    fn strict_judge_requires_valid_pass_and_nonempty_reason() {
        assert!(parse_judge_verdict(r#"{"decision":"pass","reason":"Tests passed"}"#, true).0);
        for invalid in [
            "PASS",
            r#"{"decision":"pass","reason":"#,
            r#"{"decision":"pass","reason":" "}"#,
            r#"{"decision":"pass"}"#,
            r#"{"decision":"keep","reason":"Looks good"}"#,
            r#"{"decision":"fail","reason":"Tests failed"}"#,
        ] {
            assert!(!parse_judge_verdict(invalid, true).0, "{invalid}");
        }
    }

    #[tokio::test]
    #[ignore]
    async fn e2e_webkit_qwen36_real_run() {
        let hw = crate::hardware::detect_hardware_profile();
        println!("[E2E] Hardware: {:?} | Recommended: {:?}", hw.gpus, hw.recommended_backend);

        let rt = crate::runtime::get_runtime_status();
        println!("[E2E] Installed runtimes: {:?}", rt.installed);
        assert!(!rt.installed.is_empty(), "No llama.cpp runtime installed");

        let hf_repos = crate::models::search_hf_models("Qwen3.6-35B-A3B", None).await;
        println!("[E2E] HF search Qwen3.6-35B-A3B => {:?}", hf_repos.as_ref().map(|v| v.len()));

        let model_path = std::env::var("LOORS_E2E_MODEL")
            .expect("Set LOORS_E2E_MODEL to a local .gguf path before running this ignored E2E test");
        assert!(Path::new(&model_path).exists(), "Model GGUF not found at {model_path}");

        let ws = std::env::var("LOORS_E2E_WORKSPACE")
            .expect("Set LOORS_E2E_WORKSPACE to a test repo directory before running this ignored E2E test");
        assert!(Path::new(&ws).exists(), "E2E test workspace not found");

        let graph = WorkflowGraph {
            name: "webkit-605-e2e".into(),
            nodes: vec![
                GraphNode {
                    id: "in".into(),
                    data: GraphNodeData {
                        kind: "task_input".into(),
                        title: "WebKit 605 Input".into(),
                        config: HashMap::from([
                            ("workspace".into(), json!(ws)),
                            (
                                "prompt".into(),
                                json!("Audita Source/JavaScriptCore/wasm en WebKit 605. Usa list_files o find_files para ver archivos, lee 1 archivo clave con read_file (usa start_line y end_line para no leer todo) y busca un posible bug de bounds-check, integer overflow o UAF. Escribe un reporte conciso en .loors/REPORT.md con write_file."),
                            ),
                        ]),
                    },
                },
                GraphNode {
                    id: "model".into(),
                    data: GraphNodeData {
                        kind: "local_model".into(),
                        title: "Qwen3.6-35B-A3B".into(),
                        config: HashMap::from([
                            ("modelPath".into(), json!(model_path)),
                            ("contextSize".into(), json!(16384)),
                            ("gpuLayers".into(), json!(-1)),
                            ("temperature".into(), json!(0.2)),
                        ]),
                    },
                },
                GraphNode {
                    id: "skill".into(),
                    data: GraphNodeData {
                        kind: "skill".into(),
                        title: "WebKit Security Skill".into(),
                        config: HashMap::from([
                            ("mode".into(), json!("inject")),
                            (
                                "inlineInstructions".into(),
                                json!("En WebKit WASM busca unchecked integer arithmetic, raw pointers sin RefPtr, o bounds checks faltantes. Sé breve y concreto."),
                            ),
                        ]),
                    },
                },
                GraphNode {
                    id: "agent".into(),
                    data: GraphNodeData {
                        kind: "agent".into(),
                        title: "WebKit Auditor".into(),
                        config: HashMap::from([
                            (
                                "systemPrompt".into(),
                                json!("Eres un auditor de seguridad de WebKit/C++. Usa las herramientas de forma eficiente (máximo 5 pasos) y escribe el hallazgo en .loors/REPORT.md."),
                            ),
                            (
                                "tools".into(),
                                json!(["list_files", "find_files", "grep_search", "read_file", "write_file"]),
                            ),
                            ("maxSteps".into(), json!(6)),
                        ]),
                    },
                },
                GraphNode {
                    id: "judge".into(),
                    data: GraphNodeData {
                        kind: "llm_judge".into(),
                        title: "Judge Reporte".into(),
                        config: HashMap::from([
                            (
                                "rubric".into(),
                                json!("Aprueba (pass) si el auditor analizó código real de WebKit y produjo un análisis técnico concreto."),
                            ),
                            ("inputTemplate".into(), json!("Análisis del auditor:\n{{agent_output}}")),
                        ]),
                    },
                },
                GraphNode {
                    id: "out".into(),
                    data: GraphNodeData {
                        kind: "output".into(),
                        title: "Fin E2E".into(),
                        config: HashMap::from([("label".into(), json!("Auditoría WebKit completada"))]),
                    },
                },
            ],
            edges: vec![
                GraphEdge {
                    id: "e1".into(),
                    source: "in".into(),
                    source_handle: Some("exec".into()),
                    target: "agent".into(),
                    target_handle: Some("exec".into()),
                },
                GraphEdge {
                    id: "em1".into(),
                    source: "model".into(),
                    source_handle: Some("model_ref".into()),
                    target: "agent".into(),
                    target_handle: Some("model_ref".into()),
                },
                GraphEdge {
                    id: "esk".into(),
                    source: "skill".into(),
                    source_handle: Some("skill_ref".into()),
                    target: "agent".into(),
                    target_handle: Some("skill_ref".into()),
                },
                GraphEdge {
                    id: "e2".into(),
                    source: "agent".into(),
                    source_handle: Some("exec".into()),
                    target: "judge".into(),
                    target_handle: Some("exec".into()),
                },
                GraphEdge {
                    id: "em2".into(),
                    source: "model".into(),
                    source_handle: Some("model_ref".into()),
                    target: "judge".into(),
                    target_handle: Some("model_ref".into()),
                },
                GraphEdge {
                    id: "e3".into(),
                    source: "judge".into(),
                    source_handle: Some("pass".into()),
                    target: "out".into(),
                    target_handle: Some("exec".into()),
                },
            ],
        };

        let llama = LlamaCppManager::new();
        let emitter: EventEmitter = Arc::new(|ev| match ev {
            EngineEvent::NodeLog { node_title, stream, text, .. } => {
                println!("[{node_title} | {:?}] {text}", stream);
            }
            EngineEvent::RunFinished { outcome, summary, .. } => {
                println!("[FINISHED] {:?} — {summary}", outcome);
            }
            _ => {}
        });

        let outcome = execute_graph(
            "e2e-webkit".into(),
            graph,
            RuntimeBackendKind::Vulkan,
            llama.clone(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            emitter,
        )
        .await;

        llama.shutdown().await;
        println!("[E2E] Final outcome: {:?}", outcome);
        assert!(matches!(outcome, RunOutcomeKind::Completed));
    }

    #[tokio::test]
    #[ignore]
    async fn e2e_audit_qwen38_27b_full() {
        let repo_id = "unsloth/Qwen3.8-27B-GGUF";
        let filename = "Qwen3.8-27B-UD-Q4_K_M.gguf";
        let expected_path = crate::models::models_dir().join(filename);

        let model_path = if expected_path.exists() {
            println!("[AUDIT] Modelo ya descargado: {}", expected_path.display());
            expected_path.to_string_lossy().to_string()
        } else {
            println!("[AUDIT] Descargando {repo_id}/{filename} con Loors models::download_hf_model...");
            let last_pct = std::sync::atomic::AtomicU32::new(0);
            crate::models::download_hf_model(repo_id, filename, move |ev| {
                if let Some(p) = ev.percent {
                    let bucket = (p as u32) / 10;
                    if bucket > last_pct.load(Ordering::Relaxed) {
                        last_pct.store(bucket, Ordering::Relaxed);
                        println!("[DOWNLOAD] {} ({:.0}%)", ev.message, p);
                    }
                }
            })
            .await
            .expect("Error descargando Qwen3.8-27B desde Hugging Face")
        };

        let auto_ctx = compute_auto_context_size(&model_path);
        println!("[AUDIT] Auto context size calculado para este equipo: {auto_ctx} tokens");

        let ws = std::env::var("LOORS_E2E_WORKSPACE")
            .expect("Set LOORS_E2E_WORKSPACE to a test repo directory before running this ignored E2E test");
        let _ = std::fs::remove_dir_all(Path::new(&ws).join(".loors"));

        let graph = WorkflowGraph {
            name: "webkit-605-qwen38-audit".into(),
            nodes: vec![
                GraphNode {
                    id: "in".into(),
                    data: GraphNodeData {
                        kind: "task_input".into(),
                        title: "WebKit 605 Input".into(),
                        config: HashMap::from([
                            ("workspace".into(), json!(ws)),
                            (
                                "prompt".into(),
                                json!("Audita Source/JavaScriptCore/wasm y Source/JavaScriptCore/dfg en AppleWebKit 605. En cada ciclo: 1) revisa la memoria para no repetir, 2) inspecciona un archivo con grep_search/read_file, 3) escribe tu hallazgo concreto en .loors/REPORT.md usando write_file, y responde un resumen breve."),
                            ),
                        ]),
                    },
                },
                GraphNode {
                    id: "model".into(),
                    data: GraphNodeData {
                        kind: "local_model".into(),
                        title: "Qwen3.8-27B-UD-Q4_K_M".into(),
                        config: HashMap::from([
                            ("modelPath".into(), json!(model_path)),
                            ("contextSize".into(), json!(0)),
                            ("gpuLayers".into(), json!(-1)),
                            ("temperature".into(), json!(0.2)),
                            ("topP".into(), json!(0.95)),
                            ("flashAttn".into(), json!(true)),
                        ]),
                    },
                },
                GraphNode {
                    id: "skill".into(),
                    data: GraphNodeData {
                        kind: "skill".into(),
                        title: "WebKit Security Skill".into(),
                        config: HashMap::from([
                            ("mode".into(), json!("inject")),
                            (
                                "inlineInstructions".into(),
                                json!("En WebKit/JSC busca integer overflows, bounds checks faltantes o raw pointers sin RefPtr/CheckedPtr. Usa como máximo 4 tool calls antes de escribir .loors/REPORT.md con write_file."),
                            ),
                        ]),
                    },
                },
                GraphNode {
                    id: "mem_r".into(),
                    data: GraphNodeData {
                        kind: "memory_file".into(),
                        title: "Leer AUDITED.md".into(),
                        config: HashMap::from([
                            ("mode".into(), json!("read")),
                            ("filePath".into(), json!(".loors/AUDITED.md")),
                        ]),
                    },
                },
                GraphNode {
                    id: "agent".into(),
                    data: GraphNodeData {
                        kind: "agent".into(),
                        title: "WebKit Auditor".into(),
                        config: HashMap::from([
                            (
                                "systemPrompt".into(),
                                json!("Eres un auditor de seguridad C++/WebKit. Memoria previa:\n{{memory}}\nNo repitas archivos ya auditados. Escribe el reporte en .loors/REPORT.md y devuelve un resumen concreto."),
                            ),
                            (
                                "tools".into(),
                                json!(["list_files", "find_files", "grep_search", "code_symbols", "read_file", "write_file"]),
                            ),
                            ("maxSteps".into(), json!(8)),
                        ]),
                    },
                },
                GraphNode {
                    id: "judge".into(),
                    data: GraphNodeData {
                        kind: "llm_judge".into(),
                        title: "Judge Reporte".into(),
                        config: HashMap::from([
                            (
                                "rubric".into(),
                                json!("Aprueba (pass) si el auditor analizó código real de WebKit citando archivo, función y líneas concretas."),
                            ),
                            ("inputTemplate".into(), json!("Análisis del auditor:\n{{agent_output}}")),
                        ]),
                    },
                },
                GraphNode {
                    id: "mem_w".into(),
                    data: GraphNodeData {
                        kind: "memory_file".into(),
                        title: "Guardar AUDITED.md".into(),
                        config: HashMap::from([
                            ("mode".into(), json!("append")),
                            ("filePath".into(), json!(".loors/AUDITED.md")),
                            (
                                "template".into(),
                                json!("## Ciclo {{iteration}}\n- Judge: {{judge_reason}}\n- Resumen: {{agent_output}}\n"),
                            ),
                        ]),
                    },
                },
                GraphNode {
                    id: "guard".into(),
                    data: GraphNodeData {
                        kind: "loop_guard".into(),
                        title: "Loop Guard (2 ciclos)".into(),
                        config: HashMap::from([
                            ("maxIterations".into(), json!(2)),
                            (
                                "feedbackTemplate".into(),
                                json!("Ciclo {{iteration}}: audita OTRO archivo distinto en Source/JavaScriptCore/dfg o wasm. Evaluación anterior: {{judge_reason}}"),
                            ),
                        ]),
                    },
                },
                GraphNode {
                    id: "out".into(),
                    data: GraphNodeData {
                        kind: "output".into(),
                        title: "Fin Auditoría".into(),
                        config: HashMap::from([("label".into(), json!("2 ciclos de auditoría completados"))]),
                    },
                },
            ],
            edges: vec![
                GraphEdge { id: "e1".into(), source: "in".into(), source_handle: Some("exec".into()), target: "mem_r".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "em1".into(), source: "model".into(), source_handle: Some("model_ref".into()), target: "agent".into(), target_handle: Some("model_ref".into()) },
                GraphEdge { id: "em2".into(), source: "model".into(), source_handle: Some("model_ref".into()), target: "judge".into(), target_handle: Some("model_ref".into()) },
                GraphEdge { id: "esk".into(), source: "skill".into(), source_handle: Some("skill_ref".into()), target: "agent".into(), target_handle: Some("skill_ref".into()) },
                GraphEdge { id: "e2".into(), source: "mem_r".into(), source_handle: Some("exec".into()), target: "agent".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e3".into(), source: "agent".into(), source_handle: Some("exec".into()), target: "judge".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e4".into(), source: "judge".into(), source_handle: Some("pass".into()), target: "mem_w".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e5".into(), source: "judge".into(), source_handle: Some("fail".into()), target: "guard".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e6".into(), source: "mem_w".into(), source_handle: Some("exec".into()), target: "guard".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e7".into(), source: "guard".into(), source_handle: Some("retry".into()), target: "mem_r".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e8".into(), source: "guard".into(), source_handle: Some("exhausted".into()), target: "out".into(), target_handle: Some("exec".into()) },
            ],
        };

        let llama = LlamaCppManager::new();
        let log_file = open_run_log_file("audit-qwen38");
        let log_arc = Arc::new(Mutex::new(log_file));
        let log_arc_clone = log_arc.clone();

        let emitter: EventEmitter = Arc::new(move |ev| {
            write_event_to_log(&log_arc_clone, &ev);
            match &ev {
                EngineEvent::ModelState { loaded, model_name, context_size } => {
                    println!("[MODEL_STATE] loaded={loaded} name={model_name:?} ctx={context_size:?}");
                }
                EngineEvent::NodeLog { node_title, iteration, stream, text, .. } => {
                    println!("[#{iteration} | {node_title} | {:?}] {}", stream, truncate(text, 180));
                }
                EngineEvent::RunFinished { outcome, total_iterations, summary, .. } => {
                    println!("[FINISHED] {:?} iters={total_iterations} — {summary}", outcome);
                }
                _ => {}
            }
        });

        let outcome = execute_graph(
            "audit-qwen38".into(),
            graph,
            RuntimeBackendKind::Vulkan,
            llama.clone(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            emitter,
        )
        .await;

        llama.shutdown().await;
        println!("[AUDIT] Outcome final: {:?}", outcome);
        assert!(matches!(outcome, RunOutcomeKind::Completed));
    }

    #[test]
    fn renders_node_outputs_and_loop_ledger_and_rescues_inline_tool_calls() {
        let mut ctx = ExecutionContext::default();
        ctx.node_outputs.insert("wk-auditor".into(), "Hallazgo en Foo.cpp:42".into());
        ctx.loop_ledger.push("- Ciclo #1: PASS".into());
        let rendered = ctx.render_template("A={{node:wk-auditor}} | L={{loop_ledger}}");
        assert_eq!(rendered, "A=Hallazgo en Foo.cpp:42 | L=- Ciclo #1: PASS");

        let rescued = parse_inline_tool_calls(
            "Voy a leerlo\n<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.rs\"}}\n</tool_call>",
        );
        assert_eq!(rescued.len(), 1);
        assert_eq!(rescued[0].function.name, "read_file");
        assert!(rescued[0].function.arguments.contains("a.rs"));
    }

    #[tokio::test]
    async fn e2e_for_each_json_validator_retry_loop_and_metrics() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let ws = std::env::temp_dir().join(format!("loors-test-loop-{ts}"));
        std::fs::create_dir_all(&ws).unwrap();
        let ws_str = ws.to_string_lossy().to_string();

        // Worker command: emits invalid JSON (missing keys) on the first attempt for "beta",
        // then valid JSON wrapped in markdown fences for every other call.
        let worker_cmd = if cfg!(target_os = "windows") {
            "$flag = '.loors\\beta_failed'; if ('{{item}}' -eq 'beta' -and -not (Test-Path $flag)) { New-Item -ItemType Directory -Force -Path '.loors' | Out-Null; Set-Content -Path $flag -Value '1'; Write-Output '{\"item\":\"beta\"}' } else { Write-Output ('```json\n{\"item\":\"{{item}}\",\"status\":\"ok\",\"idx\":\"{{item_index}}/{{item_total}}\"}\n```') }"
        } else {
            "if [ '{{item}}' = 'beta' ] && [ ! -f .loors/beta_failed ]; then mkdir -p .loors && touch .loors/beta_failed && echo '{\"item\":\"beta\"}'; else printf '```json\\n{\"item\":\"{{item}}\",\"status\":\"ok\",\"idx\":\"{{item_index}}/{{item_total}}\"}\\n```\\n'; fi"
        };

        let gen_cmd = if cfg!(target_os = "windows") {
            "Write-Output '[\"alpha\", \"beta\", \"gamma\"]'"
        } else {
            "echo '[\"alpha\", \"beta\", \"gamma\"]'"
        };

        let graph = WorkflowGraph {
            name: "batch-json-loop-test".into(),
            nodes: vec![
                GraphNode {
                    id: "in".into(),
                    data: GraphNodeData {
                        kind: "task_input".into(),
                        title: "Input".into(),
                        config: HashMap::from([
                            ("workspace".into(), json!(ws_str)),
                            ("prompt".into(), json!("Process batch items")),
                            ("maxLlmCalls".into(), json!(10)),
                            ("maxRunMinutes".into(), json!(5)),
                        ]),
                    },
                },
                GraphNode {
                    id: "gen".into(),
                    data: GraphNodeData {
                        kind: "shell".into(),
                        title: "Generate List".into(),
                        config: HashMap::from([("command".into(), json!(gen_cmd))]),
                    },
                },
                GraphNode {
                    id: "each".into(),
                    data: GraphNodeData {
                        kind: "for_each".into(),
                        title: "For Each".into(),
                        config: HashMap::from([
                            ("itemsSource".into(), json!("{{stdout}}")),
                            ("splitMode".into(), json!("auto")),
                            ("maxItems".into(), json!(10)),
                        ]),
                    },
                },
                GraphNode {
                    id: "worker".into(),
                    data: GraphNodeData {
                        kind: "shell".into(),
                        title: "Process Item".into(),
                        config: HashMap::from([("command".into(), json!(worker_cmd))]),
                    },
                },
                GraphNode {
                    id: "val".into(),
                    data: GraphNodeData {
                        kind: "json_validator".into(),
                        title: "Validate JSON".into(),
                        config: HashMap::from([
                            ("sourceTemplate".into(), json!("{{stdout}}")),
                            ("expectType".into(), json!("object")),
                            ("requiredKeys".into(), json!("item, status, idx")),
                        ]),
                    },
                },
                GraphNode {
                    id: "guard".into(),
                    data: GraphNodeData {
                        kind: "loop_guard".into(),
                        title: "Retry Guard".into(),
                        config: HashMap::from([
                            ("maxIterations".into(), json!(5)),
                            ("feedbackTemplate".into(), json!("Retry {{item}}: {{stderr}}")),
                        ]),
                    },
                },
                GraphNode {
                    id: "mem".into(),
                    data: GraphNodeData {
                        kind: "memory_file".into(),
                        title: "Save Valid JSON".into(),
                        config: HashMap::from([
                            ("mode".into(), json!("append")),
                            ("filePath".into(), json!(".loors/results.jsonl")),
                            ("template".into(), json!("{{custom_output}}")),
                        ]),
                    },
                },
                GraphNode {
                    id: "out".into(),
                    data: GraphNodeData {
                        kind: "output".into(),
                        title: "Done".into(),
                        config: HashMap::from([("label".into(), json!("Batch complete"))]),
                    },
                },
            ],
            edges: vec![
                GraphEdge { id: "e1".into(), source: "in".into(), source_handle: Some("exec".into()), target: "gen".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e2".into(), source: "gen".into(), source_handle: Some("exec".into()), target: "each".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e3".into(), source: "each".into(), source_handle: Some("each".into()), target: "worker".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e4".into(), source: "each".into(), source_handle: Some("done".into()), target: "out".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e5".into(), source: "worker".into(), source_handle: Some("exec".into()), target: "val".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e6".into(), source: "val".into(), source_handle: Some("valid".into()), target: "mem".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e7".into(), source: "val".into(), source_handle: Some("invalid".into()), target: "guard".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e8".into(), source: "guard".into(), source_handle: Some("retry".into()), target: "worker".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e9".into(), source: "mem".into(), source_handle: Some("exec".into()), target: "each".into(), target_handle: Some("exec".into()) },
            ],
        };

        let events = Arc::new(Mutex::new(Vec::<EngineEvent>::new()));
        let ev_clone = events.clone();
        let emitter: EventEmitter = Arc::new(move |ev| {
            ev_clone.lock().unwrap().push(ev);
        });

        let outcome = execute_graph(
            "test-batch-loop".into(),
            graph,
            RuntimeBackendKind::Cpu,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            emitter,
        )
        .await;

        assert!(matches!(outcome, RunOutcomeKind::Completed));

        // 1. Verify all 3 items were processed and persisted as valid JSON
        let saved = std::fs::read_to_string(ws.join(".loors/results.jsonl"))
            .expect("results.jsonl should exist");
        let lines: Vec<&str> = saved.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 3, "Expected 3 validated items, got:\n{saved}");
        let parsed: Vec<Value> = lines
            .iter()
            .map(|l| serde_json::from_str::<Value>(l).expect("Each saved line must be valid JSON"))
            .collect();
        assert_eq!(parsed[0]["item"], "alpha");
        assert_eq!(parsed[0]["idx"], "1/3");
        assert_eq!(parsed[1]["item"], "beta");
        assert_eq!(parsed[1]["idx"], "2/3");
        assert_eq!(parsed[2]["item"], "gamma");
        assert_eq!(parsed[2]["idx"], "3/3");

        // 2. Verify JSON validator caught the invalid attempt and loop_guard triggered retry
        let recorded = events.lock().unwrap();
        let saw_invalid_log = recorded.iter().any(|ev| {
            matches!(ev, EngineEvent::NodeLog { node_id, text, .. } if node_id == "val" && text.contains("JSON invalid"))
        });
        assert!(saw_invalid_log, "Expected json_validator to log an invalid JSON error for beta");

        // A retry reports the iteration that chose it, before the guard advances the counter.
        let retry = recorded.iter().find_map(|event| match event {
            EngineEvent::NodeResult { node_id, iteration, step, out_port: Some(port), .. }
                if node_id == "guard" && port == "retry" => Some((*iteration, *step)),
            _ => None,
        }).expect("Expected a recorded retry decision");
        assert!(recorded.iter().any(|event| matches!(event,
            EngineEvent::NodeResult { node_id, iteration, step, .. }
                if node_id == "worker" && *iteration == retry.0 + 1 && *step > retry.1
        )));

        // 3. Verify per-node latency metrics were emitted on Success status events
        let success_durations: Vec<(String, String)> = recorded
            .iter()
            .filter_map(|ev| match ev {
                EngineEvent::NodeStatus {
                    node_id,
                    status: NodeStatusKind::Success,
                    message: Some(msg),
                    ..
                } if msg.ends_with("ms") || msg.ends_with('s') => Some((node_id.clone(), msg.clone())),
                _ => None,
            })
            .collect();
        for expected_node in ["in", "gen", "each", "worker", "val", "mem"] {
            assert!(
                success_durations.iter().any(|(id, _)| id == expected_node),
                "Expected latency metric on Success for node '{expected_node}', got: {success_durations:?}"
            );
        }

        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn resume_from_node_reuses_last_run_snapshot_without_rerunning_upstream() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let ws = std::env::temp_dir().join(format!("loors-test-resume-{ts}"));
        std::fs::create_dir_all(&ws).unwrap();
        let ws_str = ws.to_string_lossy().to_string();

        let gen_cmd = if cfg!(target_os = "windows") {
            "Write-Output '{\"status\":\"cached\",\"score\":99}'"
        } else {
            "echo '{\"status\":\"cached\",\"score\":99}'"
        };

        let make_graph = |upstream_cmd: &str| WorkflowGraph {
            name: "resume-test".into(),
            nodes: vec![
                GraphNode {
                    id: "in".into(),
                    data: GraphNodeData {
                        kind: "task_input".into(),
                        title: "Input".into(),
                        config: HashMap::from([
                            ("workspace".into(), json!(ws_str.clone())),
                            ("prompt".into(), json!("Resume test")),
                        ]),
                    },
                },
                GraphNode {
                    id: "gen".into(),
                    data: GraphNodeData {
                        kind: "shell".into(),
                        title: "Upstream".into(),
                        config: HashMap::from([("command".into(), json!(upstream_cmd))]),
                    },
                },
                GraphNode {
                    id: "val".into(),
                    data: GraphNodeData {
                        kind: "json_validator".into(),
                        title: "Validate".into(),
                        config: HashMap::from([
                            ("sourceTemplate".into(), json!("{{stdout}}")),
                            ("expectType".into(), json!("object")),
                            ("requiredKeys".into(), json!("status, score")),
                        ]),
                    },
                },
                GraphNode {
                    id: "out".into(),
                    data: GraphNodeData {
                        kind: "output".into(),
                        title: "Done".into(),
                        config: HashMap::from([("label".into(), json!("Resumed ok"))]),
                    },
                },
            ],
            edges: vec![
                GraphEdge { id: "e1".into(), source: "in".into(), source_handle: Some("exec".into()), target: "gen".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e2".into(), source: "gen".into(), source_handle: Some("exec".into()), target: "val".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e3".into(), source: "val".into(), source_handle: Some("valid".into()), target: "out".into(), target_handle: Some("exec".into()) },
            ],
        };

        // Run 1: full execution populates LastRunSnapshot
        let outcome1 = execute_graph(
            "resume-run-1".into(),
            make_graph(gen_cmd),
            RuntimeBackendKind::Cpu,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(|_| {}),
        )
        .await;
        assert!(matches!(outcome1, RunOutcomeKind::Completed));

        // Run 2: resume directly at "val" — upstream "gen" must NOT run
        let ran_nodes = Arc::new(Mutex::new(Vec::<String>::new()));
        let ran_clone = ran_nodes.clone();
        let emitter: EventEmitter = Arc::new(move |ev| {
            if let EngineEvent::NodeStatus {
                node_id,
                status: NodeStatusKind::Running,
                ..
            } = ev
            {
                ran_clone.lock().unwrap().push(node_id);
            }
        });

        let outcome2 = execute_graph_with_depth(
            "resume-run-2".into(),
            make_graph("exit 1"),
            RuntimeBackendKind::Cpu,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            emitter,
            0,
            Some("val".into()),
        )
        .await;

        assert!(matches!(outcome2, RunOutcomeKind::Completed));
        let executed = ran_nodes.lock().unwrap();
        assert!(!executed.contains(&"in".to_string()), "Upstream 'in' should not run on resume");
        assert!(!executed.contains(&"gen".to_string()), "Upstream 'gen' should not run on resume");
        assert_eq!(*executed, vec!["val".to_string(), "out".to_string()]);

        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn global_run_budget_timeout_stops_loop_as_exhausted() {
        let slow_cmd = if cfg!(target_os = "windows") {
            "Start-Sleep -Milliseconds 1200; Write-Output 'done'"
        } else {
            "sleep 1.2 && echo done"
        };

        let graph = WorkflowGraph {
            name: "budget-timeout-test".into(),
            nodes: vec![
                GraphNode {
                    id: "in".into(),
                    data: GraphNodeData {
                        kind: "task_input".into(),
                        title: "Input".into(),
                        config: HashMap::from([
                            ("workspace".into(), json!(".")),
                            ("prompt".into(), json!("Timeout test")),
                            ("maxRunSeconds".into(), json!(1)),
                        ]),
                    },
                },
                GraphNode {
                    id: "slow".into(),
                    data: GraphNodeData {
                        kind: "shell".into(),
                        title: "Slow Step".into(),
                        config: HashMap::from([("command".into(), json!(slow_cmd))]),
                    },
                },
                GraphNode {
                    id: "out".into(),
                    data: GraphNodeData {
                        kind: "output".into(),
                        title: "Never Reached".into(),
                        config: HashMap::from([("label".into(), json!("Done"))]),
                    },
                },
            ],
            edges: vec![
                GraphEdge { id: "e1".into(), source: "in".into(), source_handle: Some("exec".into()), target: "slow".into(), target_handle: Some("exec".into()) },
                GraphEdge { id: "e2".into(), source: "slow".into(), source_handle: Some("exec".into()), target: "out".into(), target_handle: Some("exec".into()) },
            ],
        };

        let finish_summary = Arc::new(Mutex::new(String::new()));
        let slow_step_succeeded = Arc::new(AtomicBool::new(false));
        let sum_clone = finish_summary.clone();
        let success_clone = slow_step_succeeded.clone();
        let emitter: EventEmitter = Arc::new(move |ev| {
            if matches!(&ev, EngineEvent::NodeStatus { node_id, status: NodeStatusKind::Success, .. } if node_id == "slow") {
                success_clone.store(true, Ordering::SeqCst);
            }
            if let EngineEvent::RunFinished { summary, .. } = ev {
                *sum_clone.lock().unwrap() = summary;
            }
        });

        let outcome = execute_graph(
            "budget-timeout".into(),
            graph,
            RuntimeBackendKind::Cpu,
            LlamaCppManager::new(),
            Arc::new(Mutex::new(None)),
            Arc::new(AtomicBool::new(false)),
            emitter,
        )
        .await;

        assert!(matches!(outcome, RunOutcomeKind::Exhausted));
        assert!(!slow_step_succeeded.load(Ordering::SeqCst));
        assert!(
            finish_summary.lock().unwrap().contains("Run budget timeout reached"),
            "Expected timeout message, got: {}",
            finish_summary.lock().unwrap()
        );
    }

    #[test]
    fn graph_validation_rejects_ambiguous_outputs_and_allows_shared_resources() {
        let mut graph: WorkflowGraph = serde_json::from_value(json!({
            "nodes": [
                {"id":"a","data":{"kind":"model","title":"A"}},
                {"id":"b","data":{"kind":"agent","title":"B"}},
                {"id":"c","data":{"kind":"agent","title":"C"}}
            ],
            "edges": [
                {"id":"1","source":"a","sourceHandle":"model_ref","target":"b","targetHandle":"model_ref"},
                {"id":"2","source":"a","sourceHandle":"model_ref","target":"c","targetHandle":"model_ref"}
            ]
        })).unwrap();
        let nodes = graph.nodes.iter().map(|node| (node.id.clone(), node)).collect();
        assert!(validate_graph_connections(&graph, &nodes).is_ok());
        for edge in &mut graph.edges {
            edge.source_handle = Some("exec".into());
            edge.target_handle = Some("exec".into());
        }
        assert!(validate_graph_connections(&graph, &nodes).unwrap_err().contains("varias salidas"));
        graph.edges.pop();
        graph.edges[0].target = "missing".into();
        assert!(validate_graph_connections(&graph, &nodes).unwrap_err().contains("missing"));
    }
}
