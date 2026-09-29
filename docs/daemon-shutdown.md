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

Startup does not submit a failed, paused, blocked, canceled, refused, or completed
execution merely because the client restarted.

`cozy down --all` keeps its destructive meaning: cancel work and end rentals. The idle
exit still waits until nothing is left to manage.
