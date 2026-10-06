# Local package capture

A local run sends its machine one captured environment as a single local source.

- **One installation:** its captured source, which the machine replays with `uv sync`, or its wheels.
- **Several installations** (an unpublished script calling unpublished packages): one wheel-only
  source with the root's wheel, each callee's wheel and every locked dependency wheel, plus
  `callees`, a map from distribution to package identity. A callee runs in the caller's
  environment as a child run of its own package, with its own models and memo identity.
- **Conflicts** (two versions of a distribution, or two different wheels for one) are refused
  before submission.
- **Hub-index dependencies** are resolved once, at install, from the Hub their lock names. Their
  exact bytes travel with the run. No machine fetches them from a Hub, so the author's Hub does
  not have to be reachable from a rental, and a warm run reads no Hub.
