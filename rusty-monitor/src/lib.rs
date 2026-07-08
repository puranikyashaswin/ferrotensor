use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};
use std::io;
use std::path::PathBuf;
use std::time::Duration;
use sysinfo::System;

// ═══════════════════════════════════════════════════════════════════════════════
//                              RUSTY MONITOR - PREMIUM TUI
// ═══════════════════════════════════════════════════════════════════════════════

const LOGO: &str = r#"
██████╗ ██╗   ██╗███████╗████████╗██╗   ██╗
██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
██████╔╝██║   ██║███████╗   ██║    ╚████╔╝ 
██╔══██╗██║   ██║╚════██║   ██║     ╚██╔╝  
██║  ██║╚██████╔╝███████║   ██║      ██║   
╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   
"#;

// ─────────────────────────────────────────────────────────────────────────────
// App State
// ─────────────────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
pub enum CurrentTab {
    Dashboard,
    Training,
    Chat,
    Models,
}

#[derive(PartialEq)]
pub enum InputMode {
    Normal,
    Editing,
}

#[derive(Clone)]
pub struct TrainingConfig {
    pub model_name: String,
    pub epochs: String,
    pub batch_size: String,
    pub learning_rate: String,
}

#[derive(Clone)]
pub struct TrainingStats {
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub current_step: usize,
    pub total_steps: usize,
    pub loss: f64,
    pub accuracy: f64,
    pub eta_seconds: f64,
    pub tokens_per_sec: f64,
}

pub struct App {
    pub running: bool,
    pub active_tab: CurrentTab,
    pub sys: System,
    pub show_help: bool,

    // System history for sparklines
    pub cpu_history: Vec<u64>,
    pub ram_history: Vec<u64>,
    pub gpu_usage: f64,         // Simulated GPU
    pub gpu_memory: (f64, f64), // (used, total) in GB

    // Chat state
    pub chat_input: String,
    pub chat_history: Vec<ChatMessage>,
    pub chat_scroll: usize,

    // Training state
    pub training_config: TrainingConfig,
    pub state_source: String,
    pub active_input: usize,
    pub input_mode: InputMode,
    pub is_training: bool,
    pub training_paused: bool,
    pub loss_history: Vec<(f64, f64)>,
    pub accuracy_history: Vec<(f64, f64)>,
    pub training_stats: TrainingStats,
    pub training_step: f64,

    // Models
    pub available_models: Vec<ModelInfo>,
    pub selected_model: usize,
    pub state_path: PathBuf,
}

#[derive(Clone)]
pub struct ChatMessage {
    pub sender: String,
    pub content: String,
    pub timestamp: String,
    pub is_typing: bool,
}

#[derive(Clone)]
pub struct ModelInfo {
    pub name: String,
    pub size: String,
    pub status: String,
}

impl App {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let now = chrono::Local::now().format("%H:%M").to_string();

        Self {
            running: true,
            active_tab: CurrentTab::Dashboard,
            sys,
            show_help: false,

            cpu_history: vec![0; 60],
            ram_history: vec![0; 60],
            gpu_usage: 0.0,
            gpu_memory: (0.0, 8.0),

            chat_input: String::new(),
            chat_history: vec![
                ChatMessage {
                    sender: "System".to_string(),
                    content: "Rusty Monitor initialized. GPU backend: Metal".to_string(),
                    timestamp: now.clone(),
                    is_typing: false,
                },
                ChatMessage {
                    sender: "Llama".to_string(),
                    content: "Ready for inference. Type your prompt below.".to_string(),
                    timestamp: now,
                    is_typing: false,
                },
            ],
            chat_scroll: 0,

            training_config: TrainingConfig {
                model_name: "Llama-3-8B".to_string(),
                epochs: "3".to_string(),
                batch_size: "4".to_string(),
                learning_rate: "1e-4".to_string(),
            },
            state_source: "simulated".to_string(),
            active_input: 0,
            input_mode: InputMode::Normal,
            is_training: false,
            training_paused: false,
            loss_history: vec![],
            accuracy_history: vec![],
            training_stats: TrainingStats {
                current_epoch: 0,
                total_epochs: 3,
                current_step: 0,
                total_steps: 1000,
                loss: 0.0,
                accuracy: 0.0,
                eta_seconds: 0.0,
                tokens_per_sec: 0.0,
            },
            training_step: 0.0,

            available_models: vec![
                ModelInfo {
                    name: "Llama-3-8B".to_string(),
                    size: "16 GB".to_string(),
                    status: "Ready".to_string(),
                },
                ModelInfo {
                    name: "Llama-3-70B".to_string(),
                    size: "140 GB".to_string(),
                    status: "Not Loaded".to_string(),
                },
                ModelInfo {
                    name: "Mistral-7B".to_string(),
                    size: "14 GB".to_string(),
                    status: "Ready".to_string(),
                },
                ModelInfo {
                    name: "Phi-3-mini".to_string(),
                    size: "4 GB".to_string(),
                    status: "Ready".to_string(),
                },
                ModelInfo {
                    name: "TinyLlama-1.1B".to_string(),
                    size: "2 GB".to_string(),
                    status: "Ready".to_string(),
                },
            ],
            selected_model: 0,
            state_path: std::env::var("RUSTY_MONITOR_STATE_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("outputs/training_state.json")),
        }
    }

    pub fn on_tick(&mut self) {
        self.sys.refresh_cpu_all();
        self.sys.refresh_memory();

        // Update CPU history
        let cpus = self.sys.cpus();
        let cpu_avg = if !cpus.is_empty() {
            cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
        } else {
            0.0
        };
        self.cpu_history.push(cpu_avg as u64);
        if self.cpu_history.len() > 60 {
            self.cpu_history.remove(0);
        }

        // Update RAM history
        let ram_pct =
            (self.sys.used_memory() as f64 / self.sys.total_memory() as f64 * 100.0) as u64;
        self.ram_history.push(ram_pct);
        if self.ram_history.len() > 60 {
            self.ram_history.remove(0);
        }

        if let Ok(state) = rusty_trainer::checkpoint::TrainingState::load(&self.state_path) {
            self.state_source = "real checkpoint".to_string();
            self.is_training = state.step < state.total_steps;
            self.training_stats.current_step = state.step;
            self.training_stats.total_steps = state.total_steps.max(1);
            self.training_stats.current_epoch = state.epoch.max(1);
            self.training_stats.total_epochs =
                self.training_config.epochs.parse().unwrap_or(state.epoch.max(1));
            // best_loss is Option<f32>; training_stats.loss is f64.
            self.training_stats.loss = state
                .best_loss
                .map(|v| v as f64)
                .unwrap_or(self.training_stats.loss);
            self.training_stats.eta_seconds = 0.0;
            self.training_stats.tokens_per_sec = 0.0;
            self.training_config.batch_size = state.batch_size.to_string();
            self.training_config.learning_rate = state.lr.to_string();
            self.loss_history
                .push((state.step as f64, self.training_stats.loss));
            if self.loss_history.len() > 200 {
                self.loss_history.remove(0);
            }
        } else {
            self.state_source = "simulated fallback".to_string();
        }

        // Simulate GPU stats
        if self.is_training && !self.training_paused {
            self.gpu_usage = 85.0 + (rand::random::<f64>() - 0.5) * 20.0;
            self.gpu_memory.0 = 6.5 + (rand::random::<f64>() - 0.5) * 1.0;
        } else {
            self.gpu_usage = (self.gpu_usage * 0.9).max(5.0);
            self.gpu_memory.0 = (self.gpu_memory.0 * 0.95).max(0.5);
        }

        // Training simulation fallback when no real state file exists
        if self.is_training && !self.training_paused && !self.state_path.exists() {
            self.training_step += 1.0;

            let total_epochs: usize = self.training_config.epochs.parse().unwrap_or(3);
            let steps_per_epoch = 100;
            let total_steps = total_epochs * steps_per_epoch;

            let current_step = self.training_step as usize;
            let current_epoch = current_step / steps_per_epoch;

            // Simulated loss: exponential decay with noise
            let base_loss = 2.5 * (-self.training_step * 0.02).exp();
            let noise = (rand::random::<f64>() - 0.5) * 0.2;
            let loss = (base_loss + noise).max(0.05);

            // Simulated accuracy: inverse of loss
            let accuracy = ((1.0 - loss / 3.0) * 100.0).clamp(0.0, 99.9);

            self.loss_history.push((self.training_step, loss));
            self.accuracy_history.push((self.training_step, accuracy));

            // Keep history bounded
            if self.loss_history.len() > 200 {
                self.loss_history.remove(0);
            }
            if self.accuracy_history.len() > 200 {
                self.accuracy_history.remove(0);
            }

            // Update stats
            self.training_stats = TrainingStats {
                current_epoch: current_epoch + 1,
                total_epochs,
                current_step,
                total_steps,
                loss,
                accuracy,
                eta_seconds: ((total_steps - current_step) as f64 * 0.1).max(0.0),
                tokens_per_sec: 1200.0 + (rand::random::<f64>() - 0.5) * 200.0,
            };

            // Auto-stop when done
            if current_step >= total_steps {
                self.is_training = false;
            }
        }
    }

    pub fn next_input(&mut self) {
        self.active_input = (self.active_input + 1) % 4;
    }

    pub fn toggle_training(&mut self) {
        if self.is_training {
            self.training_paused = !self.training_paused;
        } else {
            self.is_training = true;
            self.training_paused = false;
            self.loss_history.clear();
            self.accuracy_history.clear();
            self.training_step = 0.0;
        }
    }

    pub fn stop_training(&mut self) {
        self.is_training = false;
        self.training_paused = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Loop
// ─────────────────────────────────────────────────────────────────────────────

pub fn run() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = std::time::Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::ZERO);

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    // Global keys
                    match key.code {
                        KeyCode::Char('?') => {
                            app.show_help = !app.show_help;
                            continue;
                        }
                        KeyCode::Esc if app.show_help => {
                            app.show_help = false;
                            continue;
                        }
                        _ => {}
                    }

                    if app.show_help {
                        continue;
                    }

                    match app.active_tab {
                        CurrentTab::Dashboard => match key.code {
                            KeyCode::Char('q') => app.running = false,
                            KeyCode::Tab => app.active_tab = CurrentTab::Training,
                            KeyCode::Char('1') => app.active_tab = CurrentTab::Dashboard,
                            KeyCode::Char('2') => app.active_tab = CurrentTab::Training,
                            KeyCode::Char('3') => app.active_tab = CurrentTab::Chat,
                            KeyCode::Char('4') => app.active_tab = CurrentTab::Models,
                            _ => {}
                        },
                        CurrentTab::Training => match app.input_mode {
                            InputMode::Normal => match key.code {
                                KeyCode::Char('q') => app.running = false,
                                KeyCode::Tab => app.active_tab = CurrentTab::Chat,
                                KeyCode::Char('1') => app.active_tab = CurrentTab::Dashboard,
                                KeyCode::Char('2') => app.active_tab = CurrentTab::Training,
                                KeyCode::Char('3') => app.active_tab = CurrentTab::Chat,
                                KeyCode::Char('4') => app.active_tab = CurrentTab::Models,
                                KeyCode::Down | KeyCode::Char('j') => app.next_input(),
                                KeyCode::Up | KeyCode::Char('k') => {
                                    app.active_input = if app.active_input > 0 {
                                        app.active_input - 1
                                    } else {
                                        3
                                    };
                                }
                                KeyCode::Enter => app.input_mode = InputMode::Editing,
                                KeyCode::Char('s') => app.toggle_training(),
                                KeyCode::Char('x') => app.stop_training(),
                                _ => {}
                            },
                            InputMode::Editing => match key.code {
                                KeyCode::Enter | KeyCode::Esc => app.input_mode = InputMode::Normal,
                                KeyCode::Backspace => {
                                    let val = match app.active_input {
                                        0 => &mut app.training_config.model_name,
                                        1 => &mut app.training_config.epochs,
                                        2 => &mut app.training_config.batch_size,
                                        3 => &mut app.training_config.learning_rate,
                                        _ => unreachable!(),
                                    };
                                    val.pop();
                                }
                                KeyCode::Char(c) => {
                                    let val = match app.active_input {
                                        0 => &mut app.training_config.model_name,
                                        1 => &mut app.training_config.epochs,
                                        2 => &mut app.training_config.batch_size,
                                        3 => &mut app.training_config.learning_rate,
                                        _ => unreachable!(),
                                    };
                                    val.push(c);
                                }
                                _ => {}
                            },
                        },
                        CurrentTab::Chat => match key.code {
                            KeyCode::Char('q') if app.chat_input.is_empty() => app.running = false,
                            KeyCode::Tab => app.active_tab = CurrentTab::Models,
                            KeyCode::Char('1') if app.chat_input.is_empty() => {
                                app.active_tab = CurrentTab::Dashboard
                            }
                            KeyCode::Char('2') if app.chat_input.is_empty() => {
                                app.active_tab = CurrentTab::Training
                            }
                            KeyCode::Char('3') if app.chat_input.is_empty() => {
                                app.active_tab = CurrentTab::Chat
                            }
                            KeyCode::Char('4') if app.chat_input.is_empty() => {
                                app.active_tab = CurrentTab::Models
                            }
                            KeyCode::Char(c) => app.chat_input.push(c),
                            KeyCode::Backspace => {
                                app.chat_input.pop();
                            }
                            KeyCode::Enter if !app.chat_input.is_empty() => {
                                let now = chrono::Local::now().format("%H:%M").to_string();
                                let msg = app.chat_input.drain(..).collect::<String>();
                                app.chat_history.push(ChatMessage {
                                    sender: "You".to_string(),
                                    content: msg,
                                    timestamp: now.clone(),
                                    is_typing: false,
                                });
                                // Simulate AI response
                                app.chat_history.push(ChatMessage {
                                    sender: "Llama".to_string(),
                                    content: "Processing your request...".to_string(),
                                    timestamp: now,
                                    is_typing: true,
                                });
                            }
                            KeyCode::Up => {
                                app.chat_scroll = app.chat_scroll.saturating_add(1);
                            }
                            KeyCode::Down => {
                                app.chat_scroll = app.chat_scroll.saturating_sub(1);
                            }
                            _ => {}
                        },
                        CurrentTab::Models => match key.code {
                            KeyCode::Char('q') => app.running = false,
                            KeyCode::Tab => app.active_tab = CurrentTab::Dashboard,
                            KeyCode::Char('1') => app.active_tab = CurrentTab::Dashboard,
                            KeyCode::Char('2') => app.active_tab = CurrentTab::Training,
                            KeyCode::Char('3') => app.active_tab = CurrentTab::Chat,
                            KeyCode::Char('4') => app.active_tab = CurrentTab::Models,
                            KeyCode::Down | KeyCode::Char('j') => {
                                app.selected_model =
                                    (app.selected_model + 1) % app.available_models.len();
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                app.selected_model = if app.selected_model > 0 {
                                    app.selected_model - 1
                                } else {
                                    app.available_models.len() - 1
                                };
                            }
                            KeyCode::Enter => {
                                // "Load" the selected model
                                let model = &mut app.available_models[app.selected_model];
                                model.status = "Ready".to_string();
                            }
                            _ => {}
                        },
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = std::time::Instant::now();
        }

        if !app.running {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// UI Rendering
// ─────────────────────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &App) {
    // Main layout
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer
        ])
        .split(f.area());

    // Header with tabs
    draw_header(f, main_chunks[0], app);

    // Main content
    match app.active_tab {
        CurrentTab::Dashboard => draw_dashboard(f, main_chunks[1], app),
        CurrentTab::Training => draw_training(f, main_chunks[1], app),
        CurrentTab::Chat => draw_chat(f, main_chunks[1], app),
        CurrentTab::Models => draw_models(f, main_chunks[1], app),
    }

    // Footer
    draw_footer(f, main_chunks[2], app);

    // Help overlay
    if app.show_help {
        draw_help_overlay(f);
    }
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let titles = ["[1] Dashboard", "[2] Training", "[3] Chat", "[4] Models"];
    let tabs = Tabs::new(titles.iter().map(|s| Line::from(*s)).collect::<Vec<_>>())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(0, 255, 255)))
                .title(Span::styled(
                    " RUSTY MONITOR ",
                    Style::default()
                        .fg(Color::Rgb(255, 0, 128))
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Rgb(255, 215, 0))
                .add_modifier(Modifier::BOLD),
        )
        .select(match app.active_tab {
            CurrentTab::Dashboard => 0,
            CurrentTab::Training => 1,
            CurrentTab::Chat => 2,
            CurrentTab::Models => 3,
        });
    f.render_widget(tabs, area);
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let status = if app.is_training {
        let eta = format_duration(app.training_stats.eta_seconds);
        format!(
            " TRAINING | Epoch {}/{} | Step {}/{} | Loss: {:.4} | Acc: {:.1}% | ETA: {} | {:.0} tok/s ",
            app.training_stats.current_epoch,
            app.training_stats.total_epochs,
            app.training_stats.current_step,
            app.training_stats.total_steps,
            app.training_stats.loss,
            app.training_stats.accuracy,
            eta,
            app.training_stats.tokens_per_sec
        )
    } else {
        format!(
            " GPU: {:.0}% | VRAM: {:.1}/{:.1} GB | RAM: {:.1}/{:.1} GB | [?] Help | [q] Quit ",
            app.gpu_usage,
            app.gpu_memory.0,
            app.gpu_memory.1,
            app.sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            app.sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0
        )
    };

    let style = if app.is_training {
        if app.training_paused {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::Rgb(0, 255, 128))
                .add_modifier(Modifier::BOLD)
        }
    } else {
        Style::default().fg(Color::Rgb(128, 128, 128))
    };

    let footer = Paragraph::new(status).style(style).block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
    );
    f.render_widget(footer, area);
}

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Logo + GPU
            Constraint::Min(0),    // System stats
        ])
        .split(area);

    // Top section: Logo + GPU
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);

    // ASCII Logo
    let logo = Paragraph::new(LOGO.trim())
        .style(Style::default().fg(Color::Rgb(255, 0, 128)))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        );
    f.render_widget(logo, top_chunks[0]);

    // GPU Stats
    let gpu_text = vec![
        Line::from(vec![
            Span::styled("GPU: ", Style::default().fg(Color::Gray)),
            Span::styled(
                "Apple M2 Metal",
                Style::default().fg(Color::Rgb(0, 255, 255)),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Usage: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.0}%", app.gpu_usage),
                Style::default().fg(Color::Rgb(0, 255, 128)),
            ),
        ]),
        Line::from(vec![
            Span::styled("VRAM:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1} / {:.1} GB", app.gpu_memory.0, app.gpu_memory.1),
                Style::default().fg(Color::Rgb(255, 215, 0)),
            ),
        ]),
    ];
    let gpu_block = Paragraph::new(gpu_text).block(
        Block::default()
            .title(" GPU Info ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
    );
    f.render_widget(gpu_block, top_chunks[1]);

    // Bottom: CPU and RAM sparklines
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // CPU Sparkline
    let cpu_avg = app.cpu_history.last().copied().unwrap_or(0);
    let cpu_spark = Sparkline::default()
        .block(
            Block::default()
                .title(format!(" CPU ({:>3}%) ", cpu_avg))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        )
        .data(&app.cpu_history)
        .max(100)
        .style(Style::default().fg(Color::Rgb(0, 255, 128)));
    f.render_widget(cpu_spark, bottom_chunks[0]);

    // RAM Sparkline
    let ram_pct = app.ram_history.last().copied().unwrap_or(0);
    let ram_spark = Sparkline::default()
        .block(
            Block::default()
                .title(format!(" RAM ({:>3}%) ", ram_pct))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        )
        .data(&app.ram_history)
        .max(100)
        .style(Style::default().fg(Color::Rgb(255, 0, 255)));
    f.render_widget(ram_spark, bottom_chunks[1]);
}

fn draw_training(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(32), Constraint::Min(0)])
        .split(area);

    // Left: Config Form
    let form_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(5), // Controls
            Constraint::Length(3), // Metrics source
            Constraint::Min(0),    // Progress
        ])
        .split(chunks[0]);

    let fields = [
        ("Model", &app.training_config.model_name),
        ("Epochs", &app.training_config.epochs),
        ("Batch Size", &app.training_config.batch_size),
        ("Learning Rate", &app.training_config.learning_rate),
    ];

    for (i, (label, value)) in fields.iter().enumerate() {
        let is_sel = app.active_input == i;
        let is_edit = is_sel && app.input_mode == InputMode::Editing;
        let style = if is_edit {
            Style::default().fg(Color::Rgb(0, 255, 128))
        } else if is_sel {
            Style::default().fg(Color::Rgb(0, 255, 255))
        } else {
            Style::default().fg(Color::White)
        };
        let title = if is_edit {
            format!("{} (editing)", label)
        } else {
            label.to_string()
        };
        let input = Paragraph::new(value.as_str()).style(style).block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        );
        f.render_widget(input, form_chunks[i]);
    }

    // Controls
    let ctrl_text = if app.is_training {
        if app.training_paused {
            "[s] Resume | [x] Stop"
        } else {
            "[s] Pause | [x] Stop"
        }
    } else {
        "[s] Start Training"
    };
    let controls = Paragraph::new(ctrl_text)
        .style(Style::default().fg(Color::Rgb(255, 215, 0)))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .title(" Controls ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        );
    f.render_widget(controls, form_chunks[4]);

    let source = Paragraph::new(app.state_source.as_str()).block(
        Block::default()
            .title(" Metrics Source ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
    );
    f.render_widget(source, form_chunks[5]);

    // Progress
    if app.is_training {
        let progress =
            app.training_stats.current_step as f64 / app.training_stats.total_steps as f64;
        let gauge = Gauge::default()
            .block(
                Block::default()
                    .title(" Progress ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
            )
            .gauge_style(Style::default().fg(Color::Rgb(0, 255, 128)))
            .ratio(progress.clamp(0.0, 1.0))
            .label(format!("{:.1}%", progress * 100.0));
        f.render_widget(gauge, form_chunks[6]);
    }

    // Right: Charts
    let chart_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Loss Chart
    let loss_data: Vec<(f64, f64)> = app.loss_history.clone();
    let loss_dataset = Dataset::default()
        .name("Loss")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Rgb(255, 0, 128)))
        .data(&loss_data);

    let x_max = loss_data.last().map(|d| d.0).unwrap_or(100.0).max(100.0);
    let y_max = loss_data
        .iter()
        .map(|d| d.1)
        .fold(0.0_f64, f64::max)
        .max(3.0);

    let loss_chart = Chart::new(vec![loss_dataset])
        .block(
            Block::default()
                .title(" Training Loss ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec![Span::raw("0"), Span::raw(format!("{:.0}", x_max))]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, y_max])
                .labels(vec![Span::raw("0"), Span::raw(format!("{:.1}", y_max))]),
        );
    f.render_widget(loss_chart, chart_chunks[0]);

    // Accuracy Chart
    let acc_data: Vec<(f64, f64)> = app.accuracy_history.clone();
    let acc_dataset = Dataset::default()
        .name("Accuracy")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Rgb(0, 255, 128)))
        .data(&acc_data);

    let acc_chart = Chart::new(vec![acc_dataset])
        .block(
            Block::default()
                .title(" Accuracy (%) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(vec![Span::raw("0"), Span::raw(format!("{:.0}", x_max))]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(vec![Span::raw("0"), Span::raw("100")]),
        );
    f.render_widget(acc_chart, chart_chunks[1]);
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // Messages
    let messages: Vec<ListItem> = app
        .chat_history
        .iter()
        .map(|m| {
            let prefix = if m.sender == "You" { ">" } else { "<" };
            let color = match m.sender.as_str() {
                "You" => Color::Rgb(0, 255, 255),
                "System" => Color::Rgb(128, 128, 128),
                _ => Color::Rgb(0, 255, 128),
            };
            let content = if m.is_typing {
                format!(
                    "{} {} [{}] {} ...",
                    prefix, m.sender, m.timestamp, m.content
                )
            } else {
                format!("{} {} [{}] {}", prefix, m.sender, m.timestamp, m.content)
            };
            ListItem::new(content).style(Style::default().fg(color))
        })
        .collect();

    let list = List::new(messages).block(
        Block::default()
            .title(" Chat History ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
    );
    f.render_widget(list, chunks[0]);

    // Input
    let input = Paragraph::new(app.chat_input.as_str())
        .style(Style::default().fg(Color::Rgb(255, 215, 0)))
        .block(
            Block::default()
                .title(" Message (Enter to send) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
        );
    f.render_widget(input, chunks[1]);
}

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Model list
    let items: Vec<ListItem> = app
        .available_models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let style = if i == app.selected_model {
                Style::default()
                    .fg(Color::Rgb(0, 255, 255))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            let status_color = if m.status == "Ready" {
                Color::Rgb(0, 255, 128)
            } else {
                Color::Rgb(128, 128, 128)
            };
            let line = Line::from(vec![
                Span::styled(if i == app.selected_model { "> " } else { "  " }, style),
                Span::styled(&m.name, style),
                Span::styled(format!(" ({}) ", m.size), Style::default().fg(Color::Gray)),
                Span::styled(&m.status, Style::default().fg(status_color)),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Available Models ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
    );
    f.render_widget(list, chunks[0]);

    // Model details
    let model = &app.available_models[app.selected_model];
    let details = vec![
        Line::from(vec![Span::styled(
            &model.name,
            Style::default()
                .fg(Color::Rgb(255, 215, 0))
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Size: ", Style::default().fg(Color::Gray)),
            Span::raw(&model.size),
        ]),
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::Gray)),
            Span::styled(
                &model.status,
                Style::default().fg(if model.status == "Ready" {
                    Color::Rgb(0, 255, 128)
                } else {
                    Color::Gray
                }),
            ),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "[Enter] Load Model",
            Style::default().fg(Color::Rgb(0, 255, 255)),
        )]),
    ];
    let detail_block = Paragraph::new(details).block(
        Block::default()
            .title(" Model Details ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(64, 64, 64))),
    );
    f.render_widget(detail_block, chunks[1]);
}

fn draw_help_overlay(f: &mut Frame) {
    let area = f.area();
    let block = Block::default()
        .title(" Keyboard Shortcuts ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(255, 215, 0)))
        .style(Style::default().bg(Color::Rgb(16, 16, 24)));

    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  [1-4]  ", Style::default().fg(Color::Cyan)),
            Span::raw("Switch tabs"),
        ]),
        Line::from(vec![
            Span::styled("  [Tab]  ", Style::default().fg(Color::Cyan)),
            Span::raw("Next tab"),
        ]),
        Line::from(vec![
            Span::styled("  [q]    ", Style::default().fg(Color::Cyan)),
            Span::raw("Quit"),
        ]),
        Line::from(vec![
            Span::styled("  [?]    ", Style::default().fg(Color::Cyan)),
            Span::raw("Toggle help"),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Training:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("  [s]    ", Style::default().fg(Color::Cyan)),
            Span::raw("Start/Pause training"),
        ]),
        Line::from(vec![
            Span::styled("  [x]    ", Style::default().fg(Color::Cyan)),
            Span::raw("Stop training"),
        ]),
        Line::from(vec![
            Span::styled("  [j/k]  ", Style::default().fg(Color::Cyan)),
            Span::raw("Navigate fields"),
        ]),
        Line::from(vec![
            Span::styled("  [Enter]", Style::default().fg(Color::Cyan)),
            Span::raw("Edit field"),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Chat:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("  Type   ", Style::default().fg(Color::Cyan)),
            Span::raw("Enter message"),
        ]),
        Line::from(vec![
            Span::styled("  [Enter]", Style::default().fg(Color::Cyan)),
            Span::raw("Send message"),
        ]),
        Line::from(""),
    ];

    let popup_area = centered_rect(50, 60, area);
    f.render_widget(Clear, popup_area);
    let help = Paragraph::new(help_text).block(block);
    f.render_widget(help, popup_area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn format_duration(seconds: f64) -> String {
    let mins = (seconds / 60.0) as u64;
    let secs = (seconds % 60.0) as u64;
    format!("{}:{:02}", mins, secs)
}
