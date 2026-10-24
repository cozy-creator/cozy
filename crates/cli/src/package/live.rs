//! `cozy package list` as a live view (`crate::tui`): an install is a run whose outcome reaches
//! Calcifer's machine events as the machine's environments.
use protocol::{
    calcifer::{Machine, MachineState, On},
    error::Error,
};

use super::machine::{HEADERS, named, rows};
use crate::{Context, machine::live::Machines};

/// The packages of every ready machine, or of the one `--on` names, kept as they change.
pub(crate) struct Packages {
    on: Option<String>,
    machines: Machines,
}

impl Packages {
    pub(crate) async fn open(ctx: &Context, on: Option<On>) -> Result<Self, Error> {
        ctx.narrowed()?;
        let on = match on.as_ref().map(named).transpose()? {
            Some(text) => Some(crate::machine::machine(ctx, &text.to_ascii_lowercase()).await?.name),
            None => None,
        };
        let machines = Machines::open(ctx).await?;
        Ok(Self { on, machines })
    }

    pub(crate) async fn changed(&mut self, ctx: &Context) -> Result<(), Error> {
        self.machines.changed(ctx).await
    }

    pub(crate) fn title(&self) -> String {
        match &self.on {
            Some(machine) => format!("packages on {machine}"),
            None => "packages".into(),
        }
    }

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        HEADERS.to_vec()
    }

    pub(crate) fn rows(&self, ctx: &Context) -> Vec<Vec<String>> {
        let shown = |m: &&Machine| match &self.on {
            Some(name) => m.name == *name,
            // A stopped `local` shows what it held when it last ran.
            None => matches!(m.state, MachineState::Ready | MachineState::Stopped),
        };
        let machines = self.machines.list().machines.iter().filter(shown);
        // `open` checked `--tensorhub`, the one way reading the rows fails.
        let rows = rows(ctx, machines).unwrap_or_default();
        rows.iter().map(|row| row.cells()).collect()
    }
}
