//! A live full-screen board of rows, redrawn about once a second until q, Esc or Ctrl-C. Shared
//! by `run list --watch` and `machine list --watch`.
use std::time::Duration;

use cozy_proto::error::{Error, Kind};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    layout::Constraint,
    style::{Modifier, Style},
    widgets::{Block, Row, Table},
};

/// How often the rows are read again.
const TICK: Duration = Duration::from_secs(1);

/// Shows `rows()` under `headers` until the person quits. Only for a terminal: a caller off one
/// prints one table instead.
pub async fn watch(
    title: &str,
    headers: &[&str],
    mut rows: impl AsyncFnMut() -> Result<Vec<Vec<String>>, Error>,
) -> Result<(), Error> {
    let mut terminal = ratatui::try_init().map_err(drawing)?;
    let shown = async {
        loop {
            let current = rows().await?;
            terminal
                .draw(|frame| frame.render_widget(table(title, headers, &current), frame.area()))
                .map_err(drawing)?;
            let key = tokio::task::spawn_blocking(|| match event::poll(TICK)? {
                true => event::read().map(Some),
                false => Ok(None),
            })
            .await
            .map_err(|e| drawing(std::io::Error::other(e)))?
            .map_err(drawing)?;
            if let Some(Event::Key(key)) = key
                && key.kind == KeyEventKind::Press
                && (matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                    || key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
            {
                return Ok(());
            }
        }
    }
    .await;
    ratatui::restore();
    shown
}

fn table<'a>(title: &'a str, headers: &'a [&str], rows: &'a [Vec<String>]) -> Table<'a> {
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
    Table::new(rows.iter().map(|r| Row::new(r.clone())), widths)
        .header(Row::new(headers.to_vec()).style(Style::new().add_modifier(Modifier::BOLD)))
        .column_spacing(2)
        .block(Block::new().title(format!("{title}  (q quits)")))
}

fn drawing(e: std::io::Error) -> Error {
    Error::new(Kind::Internal, "terminal", format!("the terminal: {e}"))
}
