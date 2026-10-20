use std::collections::BTreeMap;

use crate::failure::{fail, Kind, Result};
use crate::format::{Body, ObjectRef};
use crate::store2::Checkpoint;

use super::Selection;

/// Metadata selection only: the established header span walker decides the chunks.
pub(super) fn objects(checkpoint: &Checkpoint, selection: &Selection) -> Result<Vec<ObjectRef>> {
    let mut found = BTreeMap::new();
    let mut add = |object: ObjectRef| {
        found.insert(object.sha256.clone(), object);
    };
    match selection {
        Selection::All => {
            for object in checkpoint.header().objects() {
                add(object.clone());
            }
        }
        Selection::Components(components) => {
            for object in crate::store2::wanted(checkpoint.header(), Some(components))? {
                add(object);
            }
        }
        Selection::Assets { component, names } => {
            let component = checkpoint.component(component)?;
            match names {
                None => {
                    for object in component.assets.values() {
                        add(object.clone());
                    }
                }
                Some(names) => {
                    for name in names {
                        let object =
                            component
                                .assets
                                .get(name)
                                .ok_or_else(|| crate::failure::Failure {
                                    kind: Kind::NotFound,
                                    detail: format!("no asset {name:?}"),
                                })?;
                        add(object.clone());
                    }
                }
            }
        }
        Selection::Tensors { component, keys } => {
            checkpoint.component(component)?;
            for key in keys {
                for (_, location) in checkpoint.tensor(component, key)?.locations() {
                    for span in checkpoint.header().spans(&location) {
                        add(checkpoint.header().chunks[span.chunk].clone());
                    }
                }
            }
        }
        Selection::Part {
            component,
            key,
            role,
            range,
        } => {
            let tensor = checkpoint.tensor(component, key)?;
            if role.is_none() && range.is_some() && matches!(&tensor.body, Body::Encoded { .. }) {
                return fail(
                    Kind::Invalid,
                    "an encoded raw byte range needs an explicit stored part role",
                );
            }
            let mut matched = false;
            for (name, location) in tensor.locations() {
                if role.as_deref().is_some_and(|wanted| wanted != name) {
                    continue;
                }
                matched = true;
                let range = range.clone().unwrap_or(0..location.length);
                if range.start > range.end || range.end > location.length {
                    return fail(
                        Kind::Invalid,
                        format!("{component}/{key}#{name}: range exceeds part length"),
                    );
                }
                let mut offset = 0;
                for span in checkpoint.header().spans(&location) {
                    if range.start < offset + span.length && range.end > offset {
                        add(checkpoint.header().chunks[span.chunk].clone());
                    }
                    offset += span.length;
                }
            }
            if !matched {
                return fail(
                    Kind::NotFound,
                    format!("no tensor part {component}/{key}#{role:?}"),
                );
            }
        }
    }
    Ok(found.into_values().collect())
}
