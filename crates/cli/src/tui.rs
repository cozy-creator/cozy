//! Live full-screen views over Calcifer's event streams. A view is drawn again when its data
//! changes or a key is pressed, never on a timer: ↑/↓ or k/j select, Enter opens the selected
//! row's own view (a run's as `cozy run show` draws it, a machine's as `cozy machine show` does),
//! Esc or q goes back to the list as it was left, and q there quits. The terminal is restored on
//! every way out: q, Ctrl-C, a signal, an error, and (ratatui's hook) a panic.
use std::io;

use protocol::error::{Error, Kind};
use crossterm::event::{Event as Input, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::{Stream, StreamExt as _};
use ratatui::{
    Frame, Terminal,
    backend::Backend,
    layout::Constraint,
    style::{Modifier, Style},
    text::Line,
    widgets::{Block, Paragraph, Row, Table, TableState},
};

use crate::{Context, machine, package, run};

/// One live view.
pub(crate) enum Screen {
    Runs(run::live::Runs),
    Run(Box<run::live::Live>),
    Machines(machine::live::Machines),
    Machine(Box<machine::live::Shown>),
    Packages(package::live::Packages),
}

impl Screen {
    /// Waits until its data changed. Cancel-safe: a key pressed meanwhile loses nothing.
    async fn changed(&mut self, ctx: &Context) -> Result<(), Error> {
        match self {
            Screen::Runs(view) => view.changed(ctx).await,
            Screen::Run(view) => view.changed().await,
            Screen::Machines(view) => view.changed(ctx).await,
            Screen::Machine(view) => view.changed().await,
            Screen::Packages(view) => view.changed(ctx).await,
        }
    }

    /// The view opened by Enter on row `selected`, if its rows open one.
    async fn open(&self, ctx: &Context, selected: usize) -> Result<Option<Screen>, Error> {
        match self {
            Screen::Runs(view) => match view.id(selected) {
                Some(id) => Ok(Some(Screen::Run(Box::new(
                    run::live::Live::open(ctx, &id).await?,
                )))),
                None => Ok(None),
            },
            Screen::Machines(view) => match view.id(selected) {
                Some(id) => Ok(Some(Screen::Machine(Box::new(
                    machine::live::Shown::open(ctx, &id).await?,
                )))),
                None => Ok(None),
            },
            _ => Ok(None),
        }
    }

    fn draw(&self, ctx: &Context, frame: &mut Frame, selected: &mut TableState) {
        match self {
            Screen::Runs(view) => {
                table(frame, &view.title(), &view.headers(), view.rows(), selected)
            }
            Screen::Machines(view) => table(
                frame,
                &view.title(),
                &view.headers(),
                view.rows(ctx),
                selected,
            ),
            Screen::Packages(view) => table(
                frame,
                &view.title(),
                &view.headers(),
                view.rows(ctx),
                selected,
            ),
            Screen::Run(view) => detail(frame, &view.title(), view.lines(ctx)),
            Screen::Machine(view) => detail(frame, &view.title(), view.lines(ctx)),
        }
    }

    fn len(&self, ctx: &Context) -> usize {
        match self {
            Screen::Runs(view) => view.rows().len(),
            Screen::Machines(view) => view.rows(ctx).len(),
            Screen::Packages(view) => view.rows(ctx).len(),
            Screen::Run(_) | Screen::Machine(_) => 0,
        }
    }
}

/// Shows `first` until the person quits. Only for a terminal: a caller off one prints once.
pub(crate) async fn show(ctx: &Context, first: Screen) -> Result<(), Error> {
    let mut terminal = ratatui::try_init().map_err(drawing)?;
    let shown = live(ctx, &mut terminal, first, EventStream::new()).await;
    ratatui::restore();
    shown
}

/// The views on `terminal`, read from `keys`, until q, Ctrl-C or a signal.
pub(crate) async fn live<B: Backend>(
    ctx: &Context,
    terminal: &mut Terminal<B>,
    first: Screen,
    mut keys: impl Stream<Item = io::Result<Input>> + Unpin,
) -> Result<(), Error> {
    let mut stack = vec![(first, TableState::default().with_selected(Some(0)))];
    let mut signals = Signals::new()?;
    loop {
        let (screen, selected) = stack.last_mut().expect("a view");
        terminal
            .draw(|frame| screen.draw(ctx, frame, selected))
            .map_err(|e| drawing(io::Error::other(e.to_string())))?;
        let rows = screen.len(ctx);
        let next = tokio::select! {
            changed = screen.changed(ctx) => Next::Changed(changed),
            input = keys.next() => Next::Input(input),
            () = signals.next() => Next::Quit,
        };
        let key = match next {
            Next::Changed(changed) => {
                changed?;
                continue;
            }
            Next::Quit => return Ok(()),
            Next::Input(None) => return Ok(()),
            Next::Input(Some(input)) => match input.map_err(drawing)? {
                Input::Key(key) if key.kind == KeyEventKind::Press => key,
                _ => continue,
            },
        };
        let ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        let depth = stack.len();
        let (screen, selected) = stack.last_mut().expect("a view");
        let at = selected.selected().unwrap_or(0);
        match key.code {
            _ if ctrl_c => return Ok(()),
            KeyCode::Esc | KeyCode::Char('q') if depth > 1 => drop(stack.pop()),
            KeyCode::Esc | KeyCode::Char('q') => return Ok(()),
            KeyCode::Up | KeyCode::Char('k') => selected.select(Some(at.saturating_sub(1))),
            KeyCode::Down | KeyCode::Char('j') if rows > 0 => {
                selected.select(Some((at + 1).min(rows - 1)))
            }
            KeyCode::Enter => {
                if let Some(opened) = screen.open(ctx, at).await? {
                    stack.push((opened, TableState::default()));
                }
            }
            _ => {}
        }
    }
}

/// What woke the view.
enum Next {
    Changed(Result<(), Error>),
    Input(Option<io::Result<Input>>),
    Quit,
}

/// Rows under `headers`, the selected one highlighted.
fn table(
    frame: &mut Frame,
    title: &str,
    headers: &[&str],
    rows: Vec<Vec<String>>,
    selected: &mut TableState,
) {
    let widths: Vec<Constraint> = (0..headers.len())
        .map(|i| {
            let widest = rows
                .iter()
                .filter_map(|r| r.get(i))
                .map(|c| c.chars().count());
            let width = widest.chain([headers[i].len()]).max().unwrap_or(0);
            match i + 1 == headers.len() {
                true => Constraint::Fill(1),
                false => Constraint::Length(width as u16),
            }
        })
        .collect();
    let table = Table::new(rows.into_iter().map(Row::new), widths)
        .header(Row::new(headers.to_vec()).style(Style::new().add_modifier(Modifier::BOLD)))
        .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
        .column_spacing(2)
        .block(Block::new().title(format!("{title}  (↑/↓, Enter opens, q quits)")));
    frame.render_stateful_widget(table, frame.area(), selected);
}

/// One run's or machine's record as its `show` prints it, drawn again as it changes.
fn detail(frame: &mut Frame, title: &str, lines: Vec<String>) {
    let block = Block::new().title(format!("{title}  (Esc or q back)"));
    let lines: Vec<Line> = lines.into_iter().map(Line::from).collect();
    frame.render_widget(Paragraph::new(lines).block(block), frame.area());
}

/// SIGINT and SIGTERM sent from outside: raw mode turns the terminal's own Ctrl-C into a key.
struct Signals {
    #[cfg(unix)]
    term: tokio::signal::unix::Signal,
}

impl Signals {
    fn new() -> Result<Self, Error> {
        Ok(Self {
            #[cfg(unix)]
            term: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .map_err(drawing)?,
        })
    }

    async fn next(&mut self) {
        #[cfg(unix)]
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = self.term.recv() => {}
        }
        #[cfg(not(unix))]
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn drawing(e: io::Error) -> Error {
    Error::new(Kind::Internal, "terminal", format!("the terminal: {e}"))
}

/// For product tests: the live views drawn on ratatui's `TestBackend`, fed by a real Calcifer.
#[doc(hidden)]
pub mod testing {
    use std::path::Path;

    use platform::paths::Home;
    use protocol::{
        config::{Config, Loaded},
        error::Error,
    };
    use ratatui::{Terminal, backend::TestBackend};

    use super::Screen;
    use crate::{Context, output::Term, package, run};

    /// A live view the product tests draw.
    pub enum View {
        /// `cozy run list`
        Runs,
        /// `cozy package list`
        Packages,
    }

    /// Draws `view` of the Calcifer serving the cozy home `root` (a `~/.cozy`, whose Calcifer
    /// already runs), handing each frame's text to `frame` until it answers true. A frame is
    /// drawn only when Calcifer says something the view shows changed.
    pub async fn draw(
        root: &Path,
        view: View,
        mut frame: impl FnMut(&str) -> bool,
    ) -> Result<(), Error> {
        let ctx = context(root)?;
        let mut screen = match view {
            View::Runs => Screen::Runs(run::live::Runs::open(&ctx, Default::default(), 50).await?),
            View::Packages => Screen::Packages(package::live::Packages::open(&ctx, None).await?),
        };
        let mut terminal = Terminal::new(TestBackend::new(200, 20)).expect("a test terminal");
        let mut selected = ratatui::widgets::TableState::default().with_selected(Some(0));
        loop {
            let drawn = terminal.draw(|f| screen.draw(&ctx, f, &mut selected));
            let buffer = &drawn.expect("a test terminal draws").buffer;
            let text: Vec<String> = (0..buffer.area.height)
                .map(|y| {
                    (0..buffer.area.width)
                        .map(|x| buffer[(x, y)].symbol())
                        .collect()
                })
                .collect();
            if frame(&text.join("\n")) {
                return Ok(());
            }
            screen.changed(&ctx).await?;
        }
    }

    fn context(root: &Path) -> Result<Context, Error> {
        let path = root.join("config.toml");
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let (config, warnings) = Config::parse(&text, &path.display().to_string())?;
        let user_home = root.parent().unwrap_or(root).to_owned();
        let loaded = Loaded {
            config,
            home: Home::at(root),
            user_home,
            warnings,
        };
        Ok(Context::new(false, false, loaded, Term::detect()))
    }
}
