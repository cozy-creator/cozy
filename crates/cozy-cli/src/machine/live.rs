//! `cozy machine list` as a live view (`crate::tui`), fed by Calcifer's machine events.
use cozy_proto::{
    calcifer::{MachineList, MachinesEvent},
    error::Error,
};

use super::list;
use crate::{Context, calcifer_client::Watch};

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
}
