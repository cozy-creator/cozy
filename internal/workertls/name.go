// Package workertls owns the provider-neutral TLS identity of a rented worker.
package workertls

// ServerName is the fixed internal name carried by a worker's self-signed certificate.
// Tensorhub pins the exact certificate bytes separately; this name lets Cozy validate
// that certificate while dialing the provider-read-back IP address.
const ServerName = "cozy-worker"
