# Daemon shutdown

`cozy down` stops the client daemon and nothing else. It never refuses over work in
flight and never cancels it: it names what is in flight, and this computer's machine
and every rental keep running it. Rentals keep billing until `cozy rental end` (or
`cozy down --all`).

The daemon only submits, follows and collects. Everything it holds is durable, so the
next command's daemon picks each piece up where it was:

- a submission not yet accepted is prepared again (an interrupted upload resumes by
  content) and sent under the same frozen identity, which a machine answers once;
- an accepted run is observed from its durable cursor, and its outputs are collected
  into the outputs folder;
- a pending cancel or pause is sent again under the same command id;
- output exports and uploads, model transfers, rental acquisitions, installations and
  Runtime updates resume from their rows.

Startup does not submit a failed, paused, canceled, refused, or completed execution merely
because the client restarted. Work an older daemon left `blocked` fails at startup with the
reason it stopped for.

`cozy down --all` keeps its destructive meaning: cancel work and end rentals. The idle
exit waits until nothing is left to manage. Work at rest is not managed: a paused run and a
finished one keeping its results wait for the next command, whose daemon resumes or serves
them. An accepted run whose machine on this computer is gone (its process, not a slow
answer) fails as `machine_stopped`. A paid rental ask that was never answered fails once
the command making it is gone, and stays replayable under its `--idempotency-key`; one whose
rental the Hub answers 404 for is closed.
