# Owned Comfy service client

`cozy dev comfy` is an explicit development client for a ComfyUI server already running on an owned development rental. It runs outside the Python package executor; package networking fences remain unchanged. It never rents a machine or starts/stops ComfyUI.

The command verifies the attached rental and boot, uses a supplied strict SSH known-hosts file, records a stable operation/prompt identity before submission, and resumes observation after uncertain acceptance instead of posting twice. It retains the original workflow, server history, trace summary, videos and trace files with hashes.

While the command observes or retrieves results it sends bounded, retrying keepalives using the rental's existing pinned machine connection. Closing the CLI detaches observation; it does not cancel the Comfy request. This is not worker-registered inference: losing all controller keepalives for more than the fixed15-minute idle window can release the rental. The command does not extend that policy or promise a worker-side hold.

Implementation and real ordinary-CLI qualification are in progress in this draft.
