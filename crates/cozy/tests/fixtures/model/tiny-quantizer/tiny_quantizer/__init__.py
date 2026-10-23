"""A torch-free recipe for cozy's model tests: any checkpoint's attention and MLP projections to
fp8, every other tensor at its source bytes. Installs in seconds on a CPU machine."""

from cozy_runtime.author import App, Model, QuantizationPlan, Recipe

app = App()

FP8 = Recipe(
    lane="fp8",
    model=Model,
    plan=QuantizationPlan(
        encoding="fp8.rowwise",
        components=("model",),
        include=("*.c_attn.weight", "*.c_fc.weight"),
        output_precision="preserve",
    ),
)
