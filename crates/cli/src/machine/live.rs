//! `cozy machine list` as a live view (`crate::tui`), fed by Calcifer's machine events.
use protocol::{
    calcifer::{Machine, MachineList, MachineState, MachinesEvent},
    error::Error,
};

use super::{list, route};
use crate::{
    Context,
    calcifer_client::{Client, Watch},
};

/// One machine as Calcifer says it changes, known by its rental id where it has one: the Hub may
/// rename a rental while it boots.
pub(crate) struct Following {
    watch: Watch<MachinesEvent>,
    name: String,
    rental_id: Option<String>,
}

impl Following {
    /// Subscribes, then reads the machine (by name, alias or rental id): a change in between is
    /// in the stream.
    pub(crate) async fn open(calcifer: &Client, name: &str) -> Result<(Self, Machine), Error> {
        let watch = calcifer.watch("/v1/machines/events").await?;
        let machine: Machine = calcifer.get(&route(name)?).await?;
        let following = Self {
            watch,
            name: machine.name.clone(),
            rental_id: machine.rental_id.clone(),
        };
        Ok((following, machine))
    }

    /// The machine as it next changes; `None` once it left the list.
    pub(crate) async fn next(&mut self, calcifer: &Client) -> Result<Option<Machine>, Error> {
        loop {
            match self.watch.next().await {
                Some(MachinesEvent::Updated { machine }) if self.is(&machine) => {
                    self.name.clone_from(&machine.name);
                    return Ok(Some(*machine));
                }
                Some(MachinesEvent::Removed { name }) if name == self.name => return Ok(None),
                Some(_) => {}
                // A stream that ended may have missed a change: subscribe and read again.
                None => {
                    let id = self.rental_id.clone().unwrap_or_else(|| self.name.clone());
                    let (again, machine) = Self::open(calcifer, &id).await?;
                    *self = again;
                    return Ok(Some(machine));
                }
            }
        }
    }

    fn is(&self, machine: &Machine) -> bool {
        match &self.rental_id {
            Some(id) => machine.rental_id.as_ref() == Some(id),
            None => machine.name == self.name,
        }
    }
}

/// Every machine as `cozy machine list` shows it, kept as Calcifer says it changes.
pub(crate) struct Machines {
    list: MachineList,
    watch: Watch<MachinesEvent>,
}

impl Machines {
    /// Subscribes, then lists (the Hubs asked once): a change in between is in the stream.
    pub(crate) async fn open(ctx: &Context) -> Result<Self, Error> {
        let calcifer = ctx.calcifer().await?;
        let watch = calcifer.watch("/v1/machines/events").await?;
        let list = calcifer.get("/v1/machines?refresh=true").await?;
        Ok(Self { list, watch })
    }

    pub(crate) async fn changed(&mut self, ctx: &Context) -> Result<(), Error> {
        match self.watch.next().await {
            Some(MachinesEvent::Updated { machine }) => {
                let machines = &mut self.list.machines;
                match machines.iter_mut().find(|m| m.name == machine.name) {
                    Some(held) => *held = *machine,
                    None => machines.push(*machine),
                }
            }
            Some(MachinesEvent::Removed { name }) => self.list.machines.retain(|m| m.name != name),
            // A stream that ended may have missed a change: subscribe and list again.
            None => *self = Self::open(ctx).await?,
        }
        Ok(())
    }

    pub(crate) fn title(&self) -> String {
        "machines".into()
    }

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        list::HEADERS.to_vec()
    }

    pub(crate) fn rows(&self, ctx: &Context) -> Vec<Vec<String>> {
        list::rows(ctx, &self.list)
    }

    pub(crate) fn list(&self) -> &MachineList {
        &self.list
    }

    /// The machine on row `selected` (the rows are the live machines), by a name that keeps
    /// resolving: its rental id where it has one.
    pub(crate) fn id(&self, selected: usize) -> Option<String> {
        let mut live = self
            .list
            .machines
            .iter()
            .filter(|m| m.state != MachineState::Ended);
        let machine = live.nth(selected)?;
        Some(
            machine
                .rental_id
                .clone()
                .unwrap_or_else(|| machine.name.clone()),
        )
    }
}

/// One machine as `cozy machine show` draws it, kept as Calcifer says it changes.
pub(crate) struct Shown {
    calcifer: Client,
    following: Following,
    machine: Machine,
}

impl Shown {
    pub(crate) async fn open(ctx: &Context, name: &str) -> Result<Self, Error> {
        let calcifer = ctx.calcifer().await?;
        let (following, machine) = Following::open(&calcifer, name).await?;
        Ok(Self {
            calcifer,
            following,
            machine,
        })
    }

    /// Cancel-safe, as [`Following::next`] is.
    pub(crate) async fn changed(&mut self) -> Result<(), Error> {
        match self.following.next(&self.calcifer).await? {
            Some(machine) => self.machine = machine,
            // It left the list: nothing more will change.
            None => std::future::pending().await,
        }
        Ok(())
    }

    pub(crate) fn title(&self) -> String {
        format!("machine {}", self.machine.name)
    }

    pub(crate) fn lines(&self, ctx: &Context) -> Vec<String> {
        list::record(ctx, &self.machine).lines()
    }
}
