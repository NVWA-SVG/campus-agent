use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    thread,
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use serde_json::Value;

use crate::protocol::{WorkerEvent, WorkerRequest, PROTOCOL_VERSION};

pub struct WorkerClient {
    command_tx: Sender<WorkerRequest>,
    event_rx: Receiver<WorkerEvent>,
    child: Mutex<Child>,
    request_counter: AtomicU64,
}

impl WorkerClient {
    pub fn spawn() -> Result<Self> {
        let project_root = locate_project_root()?;
        apply_runtime_defaults(&project_root);
        let mut command = worker_command(&project_root)?;
        command
            .current_dir(&project_root)
            // NDJSON is a byte-level UTF-8 protocol.  A frozen Python process on
            // Chinese Windows otherwise inherits the active ANSI code page (GBK),
            // which makes both Chinese requests and responses undecodable.
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command.spawn().context("无法启动 Python Agent Worker")?;
        let stdin = child.stdin.take().context("无法连接 Agent 输入通道")?;
        let stdout = child.stdout.take().context("无法连接 Agent 输出通道")?;
        let stderr = child.stderr.take().context("无法连接 Agent 日志通道")?;
        let (command_tx, command_rx) = unbounded::<WorkerRequest>();
        let (event_tx, event_rx) = unbounded::<WorkerEvent>();

        let writer_events = event_tx.clone();
        thread::spawn(move || {
            let mut stdin = stdin;
            for request in command_rx {
                let line = match serde_json::to_string(&request) {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = writer_events.send(WorkerEvent::local_error(format!(
                            "无法编码桌面命令：{error}"
                        )));
                        continue;
                    }
                };
                if writeln!(stdin, "{line}")
                    .and_then(|_| stdin.flush())
                    .is_err()
                {
                    let _ = writer_events.send(WorkerEvent::local_error(
                        "Agent Worker 已断开，请重新启动应用",
                    ));
                    break;
                }
            }
        });

        let stdout_events = event_tx.clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) if !line.trim().is_empty() => {
                        match serde_json::from_str::<WorkerEvent>(&line) {
                            Ok(event) if event.protocol == PROTOCOL_VERSION => {
                                let _ = stdout_events.send(event);
                            }
                            Ok(_) => {
                                let _ = stdout_events
                                    .send(WorkerEvent::local_error("Agent Worker 协议版本不兼容"));
                            }
                            Err(_) => {
                                // stdout is protocol-only. Never display arbitrary content as a
                                // trusted Agent answer.
                                let _ = stdout_events
                                    .send(WorkerEvent::local_error("Agent Worker 返回了无效消息"));
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        let _ = stdout_events.send(WorkerEvent::local_error(format!(
                            "读取 Agent 输出失败：{error}"
                        )));
                        break;
                    }
                }
            }
        });

        thread::spawn(move || {
            let log_dir = project_root.join(".campus_agent_data").join("logs");
            let _ = fs::create_dir_all(&log_dir);
            let log_path = log_dir.join("desktop-worker.log");
            if let Ok(mut log_file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                for line in BufReader::new(stderr)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    let _ = writeln!(log_file, "{line}");
                }
            } else {
                // Drain stderr even if the log path is read-only, preventing child deadlock.
                for _ in BufReader::new(stderr).lines() {}
            }
        });

        Ok(Self {
            command_tx,
            event_rx,
            child: Mutex::new(child),
            request_counter: AtomicU64::new(1),
        })
    }

    pub fn send(&self, command: &str, payload: Value) -> Result<String> {
        let sequence = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let request_id = format!("desktop-{sequence}");
        self.command_tx
            .send(WorkerRequest::new(
                request_id.clone(),
                command.to_owned(),
                payload,
            ))
            .map_err(|_| anyhow!("Agent Worker 已停止"))?;
        Ok(request_id)
    }

    pub fn try_events(&self) -> impl Iterator<Item = WorkerEvent> + '_ {
        self.event_rx.try_iter()
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        let _ = self.send("shutdown", serde_json::json!({}));
        if let Ok(mut child) = self.child.lock() {
            // Give the worker a short chance to flush SQLite/checkpoint state and
            // close cleanly. Force termination is only a bounded fallback.
            for _ in 0..20 {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    return;
                }
                thread::sleep(Duration::from_millis(25));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn locate_project_root() -> Result<PathBuf> {
    if let Some(configured) = env::var_os("CAMPUS_AGENT_PROJECT_ROOT") {
        let path = PathBuf::from(configured);
        if path.join("campus_agent").is_dir()
            || path
                .join("runtime")
                .join("campus-agent-worker")
                .join("campus-agent-worker.exe")
                .is_file()
        {
            return Ok(path);
        }
        return Err(anyhow!("CAMPUS_AGENT_PROJECT_ROOT 不是有效的项目目录"));
    }

    if let Ok(current) = env::current_dir() {
        if current.join("campus_agent").is_dir() {
            return Ok(current);
        }
    }

    if let Ok(executable) = env::current_exe() {
        if let Some(directory) = executable.parent() {
            return Ok(directory.to_path_buf());
        }
    }

    Err(anyhow!("无法确定 Campus Agent 运行目录"))
}

fn apply_runtime_defaults(project_root: &Path) {
    let development_mode = project_root.join("campus_agent").is_dir();
    let data_root = if development_mode {
        project_root.join(".campus_agent_data")
    } else {
        env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| project_root.to_path_buf())
            .join("CampusAgent")
    };

    set_default_env("CAMPUS_KNOWLEDGE_DIR", data_root.join("knowledge"));
    set_default_env("CAMPUS_VECTOR_CACHE_DIR", data_root.join("vector_cache"));
    if env::var_os("CAMPUS_BUSINESS_API_MODE").is_none() {
        env::set_var("CAMPUS_BUSINESS_API_MODE", "mock");
    }

    if env::var_os("CAMPUS_EMBEDDING_PROVIDER").is_none() {
        let model_dir = [
            project_root.join("models"),
            project_root.join(".campus_agent_data").join("models"),
        ]
        .into_iter()
        .find(|path| path.is_dir());
        if let Some(model_dir) = model_dir {
            env::set_var("CAMPUS_EMBEDDING_PROVIDER", "sentence-transformers");
            env::set_var("CAMPUS_EMBEDDING_MODEL_CACHE_DIR", model_dir);
            env::set_var("CAMPUS_EMBEDDING_LOCAL_ONLY", "true");
        } else {
            env::set_var("CAMPUS_EMBEDDING_PROVIDER", "hashing");
        }
    }
}

fn set_default_env(name: &str, value: PathBuf) {
    if env::var_os(name).is_none() {
        env::set_var(name, value);
    }
}

fn worker_command(project_root: &Path) -> Result<Command> {
    if let Some(configured) = env::var_os("CAMPUS_AGENT_WORKER_EXECUTABLE") {
        let path = PathBuf::from(configured);
        if !path.is_file() {
            return Err(anyhow!("CAMPUS_AGENT_WORKER_EXECUTABLE 不存在"));
        }
        return Ok(Command::new(path));
    }

    let packaged_candidates = [
        project_root
            .join("runtime")
            .join("campus-agent-worker")
            .join("campus-agent-worker.exe"),
        project_root.join("campus-agent-worker.exe"),
    ];
    if let Some(path) = packaged_candidates.into_iter().find(|path| path.is_file()) {
        return Ok(Command::new(path));
    }

    let virtualenv_python = project_root
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    let python = if virtualenv_python.is_file() {
        virtualenv_python
    } else if let Some(configured) = env::var_os("CAMPUS_AGENT_PYTHON") {
        PathBuf::from(configured)
    } else {
        PathBuf::from("python")
    };
    let mut command = Command::new(python);
    command.args(["-m", "campus_agent.desktop_worker"]);
    Ok(command)
}
