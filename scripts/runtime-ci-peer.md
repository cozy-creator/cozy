# Runtime source peer for the LoRA cohort

CI builds the exact Runtime commit named by COZY_RUNTIME_SOURCE with the existing
read-only checkout and source-addressed CPU wheel recipe. It installs the explicit
model-execution and lora extras, which declare released PEFT/TorchAO dependencies.
No changed development bytes are presented as the public0.18.2 wheel and no PyPI
publication is required for this qualification.

Current peer includes the consolidated modern generic-LoRA/PEFT feature and exact
protocolb6e56391. It preserves current Runtime attention and preparation scope fixes.
The complete four-repository source cohort must qualify before coordinated delivery.
This source pin does not update the shared workstation CLI or rented workers.
