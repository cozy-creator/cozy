Remove `cozy rental installs`, its help/next-step suggestions, and the now-unused client/API listing surface. Accepted package installs and model downloads continue through the durable background queue, including boot wait and restart recovery.

Validation: focused queue/model/package regression files and isolated low-priority CLI build/help readback. No provider operations or new inference jobs.
