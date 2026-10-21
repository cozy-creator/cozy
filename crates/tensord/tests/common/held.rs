//! A rental proof's package as its machine holds it: an installed environment bound for an
//! actor, its models prepared by the machine's one resolver from what its store holds
//! (`local/<name>`, a checkpoint by hash). No Hub: these proofs leave the front door out.
use tensord::{
    api::domain::ModelChoice,
    gpu_service::Plan,
    journal::Installation,
    published::{Publisher, Request},
    service::Service,
};
use std::sync::Arc;

/// `package` at `release`, installed as `generation`, prepared for `actor`'s call of
/// `entrypoint` on one GPU with `choices`.
pub fn prepare(
    service: &Arc<Service>,
    publisher: &Publisher,
    actor: &str,
    (package, release, generation): (&str, &str, &str),
    entrypoint: &str,
    choices: &[ModelChoice],
) -> (Installation, Plan) {
    let held = service.catalog.resolve(generation).unwrap();
    let installed = service
        .engine
        .bind_installation(Installation {
            actor: actor.into(),
            alias: format!("held-{generation}"),
            generation: generation.into(),
            package: package.into(),
            release: release.into(),
            interface: serde_json::to_vec(&held.record.interface).unwrap(),
            hub: String::new(),
        })
        .unwrap();
    let request = Request {
        run: String::new(),
        source: None,
        package: package.into(),
        release: release.into(),
        installed: Some(installed.clone()),
        owner: actor.into(),
        catalog_revision: String::new(),
        providers: Default::default(),
        entrypoint: entrypoint.into(),
        choices: choices.to_vec(),
        gpus: 1,
        bits: 0,
        imports: true,
    };
    let prepared = publisher.prepare_now(service, actor, &request, None, None).unwrap();
    (installed, prepared.plan.expect("a model plan"))
}
