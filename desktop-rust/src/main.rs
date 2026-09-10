#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod protocol;
mod worker;

use std::{path::PathBuf, sync::Arc, time::Duration};

use eframe::egui::{
    self, Align, Color32, FontData, FontDefinitions, FontFamily, Frame, Layout, Margin, RichText,
    ScrollArea, Stroke, TextEdit, Vec2,
};
use serde_json::{json, Value};

use protocol::WorkerEvent;
use worker::WorkerClient;

const SESSION_ID: &str = "desktop-default";

#[derive(Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Clone, Default)]
struct TraceItem {
    kind: String,
    detail: String,
}

#[derive(Clone, Default)]
struct CitationItem {
    title: String,
    source: String,
    snippet: String,
    score: f64,
    retrieval_method: String,
}

#[derive(Clone, Default)]
struct KnowledgeDocument {
    document_id: String,
    display_name: String,
    category: String,
    size_bytes: u64,
}

struct CampusDesktopApp {
    worker: Option<WorkerClient>,
    messages: Vec<ChatMessage>,
    traces: Vec<TraceItem>,
    citations: Vec<CitationItem>,
    knowledge_documents: Vec<KnowledgeDocument>,
    knowledge_stats: Value,
    metrics: Value,
    health: Value,
    input: String,
    planner: String,
    status: String,
    busy: bool,
    knowledge_open: bool,
    pending_delete: Option<KnowledgeDocument>,
    clear_confirmation: bool,
}

impl CampusDesktopApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_fonts(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let (worker, status) = match WorkerClient::spawn() {
            Ok(worker) => (Some(worker), "正在初始化 Agent 与本地知识库…".to_owned()),
            Err(error) => (None, format!("启动失败：{error}")),
        };
        Self {
            worker,
            messages: vec![ChatMessage {
                role: "assistant".to_owned(),
                content: "你好，我是 Campus Agent。可以查询课程、校园办事资料和只读校园服务状态。"
                    .to_owned(),
            }],
            traces: Vec::new(),
            citations: Vec::new(),
            knowledge_documents: Vec::new(),
            knowledge_stats: Value::Null,
            metrics: Value::Null,
            health: Value::Null,
            input: String::new(),
            planner: "rule".to_owned(),
            status,
            busy: false,
            knowledge_open: false,
            pending_delete: None,
            clear_confirmation: false,
        }
    }

    fn send_command(&mut self, command: &str, payload: Value) {
        let result = self
            .worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Agent Worker 尚未运行"))
            .and_then(|worker| worker.send(command, payload));
        if let Err(error) = result {
            self.status = format!("操作失败：{error}");
            self.busy = false;
        }
    }

    fn send_chat(&mut self) {
        let query = self.input.trim().to_owned();
        if query.is_empty() || self.busy {
            return;
        }
        self.messages.push(ChatMessage {
            role: "user".to_owned(),
            content: query.clone(),
        });
        self.input.clear();
        self.traces.clear();
        self.citations.clear();
        self.busy = true;
        self.status = "Agent 正在执行 LangGraph 工作流…".to_owned();
        self.send_command(
            "chat",
            json!({
                "query": query,
                "session_id": SESSION_ID,
                "planner": self.planner,
            }),
        );
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        let events: Vec<WorkerEvent> = self
            .worker
            .as_ref()
            .map(|worker| worker.try_events().collect())
            .unwrap_or_default();
        if !events.is_empty() {
            ctx.request_repaint();
        }
        for event in events {
            self.handle_event(event);
        }
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn handle_event(&mut self, event: WorkerEvent) {
        match event.kind.as_str() {
            "ready" | "status" => {
                self.health = event.payload;
                self.status = "Agent 已就绪".to_owned();
                if event.kind == "ready" {
                    self.send_command("history", json!({ "session_id": SESSION_ID }));
                    self.send_command("list_knowledge", json!({}));
                }
            }
            "history" => {
                if let Some(items) = event.payload.get("messages").and_then(Value::as_array) {
                    if !items.is_empty() {
                        self.messages = parse_history(items);
                    }
                }
            }
            "history_cleared" => {
                self.messages = vec![ChatMessage {
                    role: "assistant".to_owned(),
                    content: "会话已经清空。你可以开始一个新的校园事务问题。".to_owned(),
                }];
                self.traces.clear();
                self.citations.clear();
                self.status = "会话已清空".to_owned();
            }
            "trace" => {
                self.traces.push(TraceItem {
                    kind: string_field(&event.payload, "type"),
                    detail: string_field(&event.payload, "detail"),
                });
            }
            "result" => {
                let answer = string_field(&event.payload, "answer");
                if !answer.is_empty() {
                    self.messages.push(ChatMessage {
                        role: "assistant".to_owned(),
                        content: answer,
                    });
                }
                self.citations = parse_citations(&event.payload);
                self.metrics = event.payload.get("metrics").cloned().unwrap_or(Value::Null);
                let elapsed = event
                    .payload
                    .get("elapsed_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                self.status = format!("回答完成 · {elapsed} ms");
            }
            "done" => {
                self.busy = false;
            }
            "knowledge" => {
                self.apply_knowledge_payload(&event.payload);
                self.status = "知识库资料已载入".to_owned();
            }
            "knowledge_changed" => {
                if let Some(stats) = event.payload.get("stats") {
                    self.knowledge_stats = stats.clone();
                }
                self.send_command("list_knowledge", json!({}));
                self.status = "知识库索引已更新".to_owned();
            }
            "error" | "fatal" => {
                let message = string_field(&event.payload, "message");
                self.status = if message.is_empty() {
                    "Agent 执行失败".to_owned()
                } else {
                    message
                };
                self.busy = false;
            }
            _ => {}
        }
    }

    fn apply_knowledge_payload(&mut self, payload: &Value) {
        self.knowledge_stats = payload.get("stats").cloned().unwrap_or(Value::Null);
        self.knowledge_documents = payload
            .get("documents")
            .and_then(Value::as_array)
            .map(|documents| {
                documents
                    .iter()
                    .map(|item| KnowledgeDocument {
                        document_id: string_field(item, "document_id"),
                        display_name: string_field(item, "display_name"),
                        category: string_field(item, "category"),
                        size_bytes: item
                            .get("size_bytes")
                            .and_then(Value::as_u64)
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
    }

    fn top_bar(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::top("top_bar")
            .frame(
                Frame::new()
                    .fill(Color32::from_rgb(247, 249, 253))
                    .inner_margin(16.0),
            )
            .show(root_ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Campus Agent")
                            .size(24.0)
                            .strong()
                            .color(Color32::from_rgb(20, 34, 63)),
                    );
                    ui.label(RichText::new("原生桌面版").color(Color32::from_rgb(74, 99, 148)));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("知识库").clicked() {
                            self.knowledge_open = true;
                            self.send_command("list_knowledge", json!({}));
                        }
                        if ui.button("清空会话").clicked() {
                            self.clear_confirmation = true;
                        }
                    });
                });
            });
    }

    fn left_panel(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::left("left_panel")
            .resizable(false)
            .exact_size(210.0)
            .frame(
                Frame::new()
                    .fill(Color32::from_rgb(241, 245, 252))
                    .inner_margin(18.0),
            )
            .show(root_ui, |ui| {
                ui.heading("运行模式");
                ui.add_space(8.0);
                ui.selectable_value(&mut self.planner, "rule".to_owned(), "Rule · 本地规则");
                ui.selectable_value(
                    &mut self.planner,
                    "deepseek".to_owned(),
                    "DeepSeek · 模型规划",
                );
                ui.add_space(20.0);
                ui.separator();
                ui.add_space(12.0);
                ui.label(RichText::new("系统状态").strong());
                ui.add_space(6.0);
                ui.label(&self.status);
                ui.add_space(10.0);

                let deepseek_configured = self
                    .health
                    .get("deepseek_configured")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                status_row(
                    ui,
                    "DeepSeek",
                    if deepseek_configured {
                        "已配置"
                    } else {
                        "未配置"
                    },
                    deepseek_configured,
                );
                let checkpoint_ok = self
                    .health
                    .pointer("/checkpoint/healthy")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                status_row(
                    ui,
                    "Checkpoint",
                    if checkpoint_ok { "正常" } else { "检查中" },
                    checkpoint_ok,
                );

                if !self.metrics.is_null() {
                    ui.add_space(20.0);
                    ui.separator();
                    ui.add_space(12.0);
                    ui.label(RichText::new("本轮指标").strong());
                    metric_row(ui, "Planner", self.metrics.get("planner"));
                    metric_row(ui, "Fallback", self.metrics.get("fallback_count"));
                    metric_row(ui, "调用次数", self.metrics.pointer("/api/calls"));
                    metric_row(ui, "Token", self.metrics.pointer("/api/total_tokens"));
                }
            });
    }

    fn right_panel(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::right("trace_panel")
            .default_size(300.0)
            .size_range(240.0..=420.0)
            .frame(
                Frame::new()
                    .fill(Color32::from_rgb(250, 251, 253))
                    .inner_margin(16.0),
            )
            .show(root_ui, |ui| {
                ui.heading("Agent 执行轨迹");
                ui.label(RichText::new("LangGraph 节点与工具事件").small().weak());
                ui.add_space(10.0);
                ScrollArea::vertical().show(ui, |ui| {
                    if self.traces.is_empty() {
                        ui.label("发送问题后，这里会显示 plan、retrieve、grade、verify 等步骤。");
                    }
                    for trace in &self.traces {
                        Frame::new()
                            .fill(Color32::WHITE)
                            .stroke(Stroke::new(1.0, Color32::from_rgb(222, 229, 241)))
                            .corner_radius(8.0)
                            .inner_margin(10.0)
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(trace.kind.to_uppercase())
                                        .strong()
                                        .color(Color32::from_rgb(48, 100, 210)),
                                );
                                ui.label(&trace.detail);
                            });
                        ui.add_space(7.0);
                    }
                });
            });
    }

    fn chat_panel(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::bottom("composer")
            .frame(Frame::new().fill(Color32::WHITE).inner_margin(16.0))
            .show(root_ui, |ui| {
                ui.horizontal(|ui| {
                    let available = (ui.available_width() - 90.0).max(200.0);
                    let response = ui.add_sized(
                        [available, 72.0],
                        TextEdit::multiline(&mut self.input)
                            .hint_text("输入校园事务问题，Ctrl+Enter 发送…")
                            .desired_rows(3),
                    );
                    let send_shortcut = response.has_focus()
                        && ui.input(|input| {
                            input.modifiers.ctrl && input.key_pressed(egui::Key::Enter)
                        });
                    if ui
                        .add_enabled(
                            !self.busy,
                            egui::Button::new("发送").min_size(Vec2::new(72.0, 46.0)),
                        )
                        .clicked()
                        || send_shortcut
                    {
                        self.send_chat();
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(Color32::from_rgb(246, 248, 252))
                    .inner_margin(20.0),
            )
            .show(root_ui, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for message in &self.messages {
                            draw_message(ui, message);
                            ui.add_space(12.0);
                        }
                        if self.busy {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("Agent 正在运行…");
                            });
                        }
                        if !self.citations.is_empty() {
                            ui.add_space(8.0);
                            ui.heading("资料来源");
                            for citation in &self.citations {
                                egui::CollapsingHeader::new(format!(
                                    "{} · {:.3} · {}",
                                    citation.title, citation.score, citation.retrieval_method
                                ))
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label(format!("来源：{}", citation.source));
                                    ui.label(&citation.snippet);
                                });
                            }
                        }
                    });
            });
    }

    fn knowledge_window(&mut self, ctx: &egui::Context) {
        let mut open = self.knowledge_open;
        egui::Window::new("知识库管理")
            .open(&mut open)
            .default_width(620.0)
            .min_width(480.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("添加资料").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("支持的资料", &["md", "txt", "pdf"])
                            .pick_file()
                        {
                            self.upload_knowledge(path);
                        }
                    }
                    if ui.button("重建索引").clicked() {
                        self.send_command("rebuild_knowledge", json!({}));
                        self.status = "正在重建知识库索引…".to_owned();
                    }
                    if ui.button("刷新").clicked() {
                        self.send_command("list_knowledge", json!({}));
                    }
                });
                ui.add_space(8.0);
                let chunks = self
                    .knowledge_stats
                    .get("chunk_count")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                let provider = self
                    .knowledge_stats
                    .get("embedding_provider")
                    .and_then(Value::as_str)
                    .unwrap_or("-");
                ui.label(format!(
                    "内置资料：{} · 上传资料：{} · Chunks：{} · Embedding：{}",
                    self.knowledge_stats
                        .get("built_in_documents")
                        .and_then(Value::as_u64)
                        .unwrap_or_default(),
                    self.knowledge_documents.len(),
                    chunks,
                    provider
                ));
                ui.separator();
                let mut delete_candidate = None;
                ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                    if self.knowledge_documents.is_empty() {
                        ui.label("暂时没有用户上传的资料。");
                    }
                    for document in &self.knowledge_documents {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(RichText::new(&document.display_name).strong());
                                ui.label(
                                    RichText::new(format!(
                                        "{} · {}",
                                        document.category,
                                        format_bytes(document.size_bytes)
                                    ))
                                    .small()
                                    .weak(),
                                );
                            });
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button("删除").clicked() {
                                    delete_candidate = Some(document.clone());
                                }
                            });
                        });
                        ui.separator();
                    }
                });
                if delete_candidate.is_some() {
                    self.pending_delete = delete_candidate;
                }
            });
        self.knowledge_open = open;
    }

    fn upload_knowledge(&mut self, path: PathBuf) {
        self.status = "正在导入资料并重建索引…".to_owned();
        self.send_command(
            "upload_knowledge",
            json!({ "path": path.to_string_lossy() }),
        );
    }

    fn confirmation_windows(&mut self, ctx: &egui::Context) {
        if self.clear_confirmation {
            egui::Window::new("确认清空会话")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("这会删除当前桌面会话的 SQLite 历史记录。是否继续？");
                    ui.horizontal(|ui| {
                        if ui.button("取消").clicked() {
                            self.clear_confirmation = false;
                        }
                        if ui.button("确认清空").clicked() {
                            self.send_command("clear_history", json!({ "session_id": SESSION_ID }));
                            self.clear_confirmation = false;
                        }
                    });
                });
        }

        if let Some(document) = self.pending_delete.clone() {
            egui::Window::new("确认删除资料")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("删除“{}”并重建索引？", document.display_name));
                    ui.horizontal(|ui| {
                        if ui.button("取消").clicked() {
                            self.pending_delete = None;
                        }
                        if ui.button("确认删除").clicked() {
                            self.send_command(
                                "delete_knowledge",
                                json!({ "document_id": document.document_id }),
                            );
                            self.pending_delete = None;
                        }
                    });
                });
        }
    }
}

impl eframe::App for CampusDesktopApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.top_bar(ui);
        self.left_panel(ui);
        self.right_panel(ui);
        self.chat_panel(ui);
        let ctx = ui.ctx().clone();
        if self.knowledge_open {
            self.knowledge_window(&ctx);
        }
        self.confirmation_windows(&ctx);
    }
}

fn draw_message(ui: &mut egui::Ui, message: &ChatMessage) {
    let is_user = message.role == "user";
    let fill = if is_user {
        Color32::from_rgb(52, 102, 220)
    } else {
        Color32::WHITE
    };
    let text_color = if is_user {
        Color32::WHITE
    } else {
        Color32::from_rgb(28, 39, 62)
    };
    ui.with_layout(
        if is_user {
            Layout::right_to_left(Align::Min)
        } else {
            Layout::left_to_right(Align::Min)
        },
        |ui| {
            Frame::new()
                .fill(fill)
                .stroke(Stroke::new(
                    1.0,
                    if is_user {
                        fill
                    } else {
                        Color32::from_rgb(220, 226, 238)
                    },
                ))
                .corner_radius(10.0)
                .inner_margin(Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_max_width(620.0);
                    ui.label(RichText::new(&message.content).color(text_color).size(15.0));
                });
        },
    );
}

fn parse_history(items: &[Value]) -> Vec<ChatMessage> {
    items
        .iter()
        .filter_map(|item| {
            let role = item.get("role")?.as_str()?.to_owned();
            let content = item.get("content")?.as_str()?.to_owned();
            Some(ChatMessage { role, content })
        })
        .collect()
}

fn parse_citations(payload: &Value) -> Vec<CitationItem> {
    payload
        .get("citations")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| CitationItem {
                    title: string_field(item, "title"),
                    source: string_field(item, "source"),
                    snippet: string_field(item, "snippet"),
                    score: item
                        .get("score")
                        .and_then(Value::as_f64)
                        .unwrap_or_default(),
                    retrieval_method: string_field(item, "retrieval_method"),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn string_field(payload: &Value, name: &str) -> String {
    payload
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn status_row(ui: &mut egui::Ui, name: &str, value: &str, positive: bool) {
    ui.horizontal(|ui| {
        ui.colored_label(
            if positive {
                Color32::from_rgb(30, 155, 94)
            } else {
                Color32::from_rgb(152, 113, 36)
            },
            "●",
        );
        ui.label(name);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(value).small().weak());
        });
    });
}

fn metric_row(ui: &mut egui::Ui, label: &str, value: Option<&Value>) {
    let display = match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Bool(flag)) => flag.to_string(),
        _ => "-".to_owned(),
    };
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(display);
        });
    });
}

fn format_bytes(size: u64) -> String {
    if size < 1024 {
        format!("{size} B")
    } else if size < 1024 * 1024 {
        format!("{:.1} KiB", size as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", size as f64 / (1024.0 * 1024.0))
    }
}

fn configure_fonts(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
    ];
    let Some(font_bytes) = candidates.iter().find_map(|path| std::fs::read(path).ok()) else {
        return;
    };
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "campus-chinese".to_owned(),
        Arc::new(FontData::from_owned(font_bytes)),
    );
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "campus-chinese".to_owned());
    }
    ctx.set_fonts(fonts);
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Campus Agent")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([980.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Campus Agent",
        options,
        Box::new(|cc| Ok(Box::new(CampusDesktopApp::new(cc)))),
    )
}
