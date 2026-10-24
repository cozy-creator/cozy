use super::*;

#[test]
fn final_memory_figures_follow_physical_gpus_and_the_exact_request() {
    let devices = [Device { entry: "GPU-b".into(), memory: None }, Device { entry: "3".into(), memory: None }];
    let gpu = |device_index, device_uuid: &str| crate::machine::receipt::Gpu {
        device_index, device_uuid: device_uuid.into(), device_name: "same-model".into(),
        driver_version: String::new(), memory_bytes: 100, pci_bus_id: String::new(),
        compute_cap: String::new(), memory_class: String::new(),
    };
    let inventory = [gpu(3, "GPU-a"), gpu(7, "GPU-b")];
    let mut dispatched = Dispatched {
        gpus: &devices, id: "call".into(), running: Arc::default(), measured: vec![None; 2],
    };
    let reply = Frame {
        memory: Some(serde_json::json!({"varena": [
            {"gpu": 3, "stats": {"request": "call", "peak": 30, "compute": 10, "host": 50, "future": true}},
            {"gpu": 7, "stats": {"request": "call", "peak": 70, "compute": 20, "host": 90}},
            {"gpu": 99, "stats": {"request": "call", "peak": 99}}
        ]}).as_object().unwrap().clone()),
        ..Frame::default()
    };
    dispatched.record(&reply, &inventory);
    assert_eq!(dispatched.measured.iter().map(|s| s.as_ref().unwrap().peak).collect::<Vec<_>>(), [70, 30]);
    assert_eq!(dispatched.measured[0].as_ref().unwrap().host, Some(90));
    let mut wrong = reply;
    wrong.memory.as_mut().unwrap()["varena"][0]["stats"]["request"] = "another-call".into();
    dispatched.record(&wrong, &inventory);
    assert!(dispatched.measured[1].is_none(), "a stale report cannot train this request");
}
