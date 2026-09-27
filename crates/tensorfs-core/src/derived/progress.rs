//! The Link chain stores immutable work context once and accepted-work deltas.
//! TransactionRow remains the sole declaration/part/config parser and validator.

use super::*;

pub(super) struct Delta {
    work: ObjectRef,
    parts: Vec<AddedPart>,
    configs: Vec<AddedConfig>,
}

pub(super) fn context(row: &TransactionRow) -> TransactionRow {
    let mut work = progress_row(row);
    work.added_parts.clear();
    work.added_configs.clear();
    work
}

fn part_key(part: &AddedPart) -> (&str, &str, &str) {
    (&part.component, &part.key, &part.role)
}

impl Delta {
    /// The parts a writer accepted since its last checkpoint, sorted as a delta stores them.
    pub(super) fn accepted(work: ObjectRef, mut parts: Vec<AddedPart>) -> Self {
        parts.sort_by(|left, right| part_key(left).cmp(&part_key(right)));
        Self {
            work,
            parts,
            configs: Vec::new(),
        }
    }

    pub(super) fn between(
        row: &TransactionRow,
        prior: Option<&TransactionRow>,
        work: ObjectRef,
    ) -> Result<Self> {
        let mut parts = row.added_parts.clone();
        let mut configs = row.added_configs.clone();
        if let Some(prior) = prior {
            for old in &prior.added_parts {
                let index = parts
                    .binary_search_by(|part| part_key(part).cmp(&part_key(old)))
                    .map_err(|_| Refusal {
                        code: Code::TRANSACTION_CONFLICT,
                        detail: "checkpoint lost accepted part".into(),
                    })?;
                if &parts[index] != old {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "checkpoint changed accepted part",
                    );
                }
                parts.remove(index);
            }
            for old in &prior.added_configs {
                let index = configs
                    .binary_search_by(|config| config.name.cmp(&old.name))
                    .map_err(|_| Refusal {
                        code: Code::TRANSACTION_CONFLICT,
                        detail: "checkpoint lost accepted config".into(),
                    })?;
                if &configs[index] != old {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "checkpoint changed accepted config",
                    );
                }
                configs.remove(index);
            }
        }
        Ok(Self {
            work,
            parts,
            configs,
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.parts.is_empty() && self.configs.is_empty()
    }

    pub(super) fn to_value(&self) -> Value {
        Value::obj(vec![
            ("work", self.work.to_value()),
            (
                "added_parts",
                Value::arr(self.parts.iter().map(AddedPart::to_value).collect()),
            ),
            (
                "added_configs",
                Value::arr(self.configs.iter().map(AddedConfig::to_value).collect()),
            ),
        ])
    }

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("DerivedProgress", value)?;
        let work = ObjectRef::from_value("derived progress work", fields.req("work")?)?;
        let parts = as_arr("DerivedProgress", "added_parts", fields.req("added_parts")?)?
            .iter()
            .map(AddedPart::from_value)
            .collect::<Result<Vec<_>>>()?;
        let configs = as_arr(
            "DerivedProgress",
            "added_configs",
            fields.req("added_configs")?,
        )?
        .iter()
        .map(AddedConfig::from_value)
        .collect::<Result<Vec<_>>>()?;
        fields.done()?;
        if parts
            .windows(2)
            .any(|rows| part_key(&rows[0]) >= part_key(&rows[1]))
            || configs.windows(2).any(|rows| rows[0].name >= rows[1].name)
        {
            return refuse(
                Code::SORT_ORDER,
                "derived progress additions must be sorted and unique",
            );
        }
        Ok(Self {
            work,
            parts,
            configs,
        })
    }

    fn apply(self, row: &mut TransactionRow) -> Result<()> {
        for part in self.parts {
            match row
                .added_parts
                .binary_search_by(|old| part_key(old).cmp(&part_key(&part)))
            {
                Ok(index) => {
                    return refuse(
                        if row.added_parts[index] == part {
                            Code::DUPLICATE_KEY
                        } else {
                            Code::TRANSACTION_CONFLICT
                        },
                        "derived progress repeats or changes accepted part",
                    )
                }
                Err(index) => row.added_parts.insert(index, part),
            }
        }
        for config in self.configs {
            match row
                .added_configs
                .binary_search_by(|old| old.name.cmp(&config.name))
            {
                Ok(index) => {
                    return refuse(
                        if row.added_configs[index] == config {
                            Code::DUPLICATE_KEY
                        } else {
                            Code::TRANSACTION_CONFLICT
                        },
                        "derived progress repeats or changes accepted config",
                    )
                }
                Err(index) => row.added_configs.insert(index, config),
            }
        }
        Ok(())
    }
}

pub(super) fn read(
    store: &Store,
    head: &ObjectRef,
    transaction: &str,
    declaration: &str,
    sources: &[SourceFact],
) -> Result<TransactionRow> {
    let link = crate::durability::local_link(store, head)?;
    let chain = crate::durability::local_chain(store, head, &link.plan)?;
    if chain.progress.is_empty() {
        return refuse(
            Code::MISSING_FIELD,
            "derived checkpoint has no progress document",
        );
    }
    let mut result: Option<(ObjectRef, TransactionRow)> = None;
    for object in &chain.progress {
        let bytes = crate::durability::read_document(store, object, limits::DOC_MAX_BYTES)?;
        let delta = Delta::from_value(&crate::canon::parse_canonical(
            &bytes,
            limits::DOC_MAX_BYTES,
        )?)?;
        if result.is_none() {
            if !chain.blobs.contains(&delta.work) {
                return refuse(
                    Code::JOURNAL_STALE,
                    "derived work context is outside its checkpoint closure",
                );
            }
            let bytes =
                crate::durability::read_document(store, &delta.work, limits::DOC_MAX_BYTES)?;
            let row = TransactionRow::from_value(&crate::canon::parse_canonical(
                &bytes,
                limits::DOC_MAX_BYTES,
            )?)?;
            if row.id != transaction
                || row.declaration.as_deref() != Some(declaration)
                || row.source_facts != sources
                || row.state != TransactionState::Open
                || row.current_session.is_some()
                || row.highest_session != 0
                || !row.added_parts.is_empty()
                || !row.added_configs.is_empty()
            {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "derived work context differs from current work or imports writer authority",
                );
            }
            if progress_plan(&row) != link.plan {
                return refuse(
                    Code::JOURNAL_STALE,
                    "derived checkpoint plan differs from current work",
                );
            }
            result = Some((delta.work.clone(), row));
        }
        let (work, row) = result.as_mut().expect("work context loaded");
        if work != &delta.work {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "derived checkpoint changes its immutable work context",
            );
        }
        delta.apply(row)?;
    }
    let row = result.expect("nonempty progress").1;
    validate_additions(&row)?;
    Ok(row)
}
