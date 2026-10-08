package hub

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/rentalid"
)

// The RENTAL routes (cl-015, on th-042's product surface). The hub owns the pod: it
// provisions generic empty capacity, starts a worker hosting `WorkerControl` behind TLS,
// and answers with the address and certificate to pin. Creator sends desired package/model
// state directly to that worker after attachment.
//
// Creator mints one media bearer and one Ed25519 rental key before the paid ask. Only the
// bearer hash and public key cross this API. Tensorhub cannot open either pod door: the
// media bearer and rental private key stay here; the latter signs only ClaimProof.
//
// The contract is the hub's and is consumed verbatim, exactly as the catalog's is:
//
//	GET    /v1/rental-skus           -> [{name, accelerator_model, base_worker_profile,
//	                                 compute_capability, vram_gb,
//	                                 widths:[{accelerator_count, price_usd_micros_per_hour,
//	                                          storage_usd_micros_per_hour}]}]
//	POST   /v1/rentals               {name, sku, accelerator_count, media_token_sha256:<64 hex>,
//	                                  creator_public_key}
//	                                 -> 202 {rental_id, name, state, ...}
//	GET    /v1/rentals/{id}          -> {state, worker_address, cert_pem, media_address,
//	                                     detail, worker_id, worker_boot_id,
//	                                     media_token_sha256:[...]}
//	DELETE /v1/rentals/{id}          -> 204
//
// `media_address` is now ALWAYS the hub's own word. The client used to derive it from the
// control address by a stated host:port+1 convention, because the pinned demo contract had
// no field for it; the product surface names it, so a guess about where somebody else's
// byte plane listens has no reason to exist and is gone.

// Rental states, the hub's own words — only the ones a reader here branches on. The
// hub's in-flight states (pending_acquisition, acquiring, materializing) reach this
// client as opaque strings it renders verbatim; naming them as constants nobody read
// was law-13 dead code (cl-028).
const (
	RentalReady            = "ready"
	RentalDegraded         = "degraded"
	RentalFailed           = "failed"
	RentalReleaseRequested = "release_requested"
	RentalReleased         = "released"
)

// RentalAbsent reports states Tensorhub commits only after proving provider
// absence. Degraded, draining, and unknown states may still be billing.
func RentalAbsent(state string) bool {
	return state == RentalFailed || state == RentalReleased
}

// Rental is one rented pod as the hub reports it. No plaintext media bearer or private
// Creator key is part of this view.
type Rental struct {
	Provider, ProviderMachineID, ProviderResourceID string
	Development                                     bool
	SSHAddress                                      string
	ID                                              string
	Name                                            string
	State                                           string
	AcceleratorModel                                string
	// AcceleratorCount is the pod's WIDTH: how many accelerators this rental delivers,
	// and therefore the size of the device envelope its worker holds and the only degree
	// a group placement on it may be pinned to. One for a one-card pod, and one for a CPU
	// product, whose pod has no accelerator at all.
	AcceleratorCount int
	Address          string
	CertPEM          string
	Detail           string
	Failure          *RentalFailure
	// MediaAddress is the pod's BYTE PLANE (cl-014, ruled #506b): the co-resident media
	// server's own listener, which is where an owner uploads a payload and downloads an
	// output. The hub observed it and names it.
	MediaAddress     string
	WorkerID         string
	WorkerBootID     string
	CreatorPublicKey string
	// MediaTokenSHA256 is the pod media plane's LIVE credential set, as hashes. It is here so this host can
	// see that the hash of the token it minted is one the pod was provisioned with —
	// a comparison neither end can make by saying the token.
	MediaTokenSHA256    []string
	HourlyRateUSDMicros int64
	// Boot is the pod's boot while the rental is acquired; nil before an attempt exists,
	// once it is ready or ended, and from a Hub older than the field.
	Boot                  *RentalBoot
	BaseWorkerImageDigest string
	BaseWorkerImageTag    string
	BaseWorkerProfile     string
	// CreatedAt is when the hub opened the rental, which is when it began billing.
	// It is how long a pod this host holds no record of has been costing money —
	// there is no local `rented_at` for a rental the records never saw. RFC 3339 as
	// the hub says it, or blank from a hub older than th-199.
	CreatedAt string
	// ContainerDiskGB is the disk the Hub bought for the pod, or 0 when it does not say.
	ContainerDiskGB int
	// SpendUSDMicros is what the rental has cost so far. SpendBasis is "provider_billed" once
	// every provider charge has settled, else "estimate"; blank from a Hub older than the fact.
	SpendUSDMicros int64
	SpendBasis     string
	// WebRTC is a ready machine-image rental's browser-media listener: the address a
	// browser dials over ICE-TCP and the pinned leaf's "sha-256 AB:…". Nil otherwise.
	WebRTC *RentalWebRTC
	// ReleaseCause is who ended a released rental (owner_stop, released_by_pod,
	// idle_unreached, ...), and EndedAt when it ended, RFC 3339 on the Hub's clock. Blank
	// while it lives, and from a Hub older than the facts.
	ReleaseCause, EndedAt string
	// UnreachableSince is when the Hub last reached a ready rental's pod, once a probe has
	// not answered since; blank otherwise.
	UnreachableSince string
	// ComputeUSDMicrosPerHour and StorageUSDMicrosPerHour are what the rental costs an
	// hour, its machine and its disk, from its create on; VCPUCount and MemoryGB are the
	// machine's shape where the Hub states one. Zero from a Hub older than the facts.
	ComputeUSDMicrosPerHour, StorageUSDMicrosPerHour int64
	VCPUCount, MemoryGB                              int
}

type RentalWebRTC struct {
	Address     string `json:"address"`
	Fingerprint string `json:"fingerprint"`
}

// RentalBoot is one boot attempt as the Hub observed it: which attempt, where, how far the
// provider has brought the container, and when its boot log last moved. State is the
// attempt's (obligated, ambiguous, booting) or "replanning" between a failed attempt and
// the next; ReplannedFrom is the failed attempt this one replaces. Phase is the boot phase
// the Hub times (container: provider running to a started container; host: to the pod's
// supervisor answering; runtime: to the Runtime's receipt), begun at PhaseStartedAt.
// Activity is the Hub's reading of the provider boot log in the container phase
// (pulling_image, creating_container, starting_container, stopping_container). Times are on
// this host's clock: the decoder shifts them by the Hub's reading of its own.
type RentalBoot struct {
	Attempt         int            `json:"attempt"`
	State           string         `json:"state"`
	Datacenter      string         `json:"datacenter,omitempty"`
	Container       string         `json:"container,omitempty"`
	RuntimeObserved bool           `json:"runtime_observed,omitempty"`
	Phase           string         `json:"phase,omitempty"`
	PhaseStartedAt  time.Time      `json:"phase_started_at,omitzero"`
	Activity        string         `json:"activity,omitempty"`
	StartedAt       time.Time      `json:"started_at,omitzero"`
	LastProgressAt  time.Time      `json:"last_progress_at,omitzero"`
	BootLogAt       time.Time      `json:"boot_log_at,omitzero"`
	ReplannedFrom   *ReplannedBoot `json:"replanned_from,omitempty"`
}

// ReplannedBoot is a failed attempt the rental replaced with another host. ObservedMaxMS is
// the boot time the Hub had observed for that product and image when it ended this one.
type ReplannedBoot struct {
	Attempt       int       `json:"attempt"`
	Datacenter    string    `json:"datacenter"`
	StartedAt     time.Time `json:"started_at"`
	EndedAt       time.Time `json:"ended_at"`
	FailureCode   string    `json:"failure_code,omitempty"`
	ObservedMaxMS int64     `json:"observed_max_ms,omitempty"`
}

// Started is a container the provider has started: the pod's own boot is under way.
func (b *RentalBoot) Started() bool {
	return b.RuntimeObserved || b.Container == "running" || b.Phase == "host" || b.Phase == "runtime"
}

// wireBoot is the Hub's boot on the Hub's clock, read at observed_at.
type wireBoot struct {
	RentalBoot
	ObservedAt time.Time `json:"observed_at"`
}

// local moves the Hub's times onto this host's clock, as of now.
func (w *wireBoot) local(now time.Time) *RentalBoot {
	if w == nil {
		return nil
	}
	boot := w.RentalBoot
	if w.ObservedAt.IsZero() {
		return &boot
	}
	skew := now.Sub(w.ObservedAt)
	shift := func(at *time.Time) {
		if !at.IsZero() {
			*at = at.Add(skew)
		}
	}
	shift(&boot.StartedAt)
	shift(&boot.LastProgressAt)
	shift(&boot.BootLogAt)
	shift(&boot.PhaseStartedAt)
	if from := boot.ReplannedFrom; from != nil {
		moved := *from
		shift(&moved.StartedAt)
		shift(&moved.EndedAt)
		boot.ReplannedFrom = &moved
	}
	return &boot
}

// RentalFailure is Tensorhub's sanitized terminal boot diagnosis. It contains
// only frozen image/resource identity and direct provider lifecycle facts.
type RentalFailure struct {
	Code                  string `json:"code"`
	BaseWorkerImageDigest string `json:"base_worker_image_digest"`
	Provider              string `json:"provider,omitempty"`
	ProviderResourceID    string `json:"provider_resource_id,omitempty"`
	ProviderHostID        string `json:"provider_host_id,omitempty"`
	ProviderState         string `json:"provider_state,omitempty"`
	ContainerState        string `json:"container_state,omitempty"`
}

// ExactDocument is the package interface returned with exact wheel downloads.
type ExactDocument struct {
	CanonicalBytes []byte `json:"canonical_bytes"`
	Digest         string `json:"digest"`
	Length         int64  `json:"length"`
}

// Ready answers whether this rental carries its native dial identity and Creator key.
func (r Rental) Ready() bool {
	return r.State == RentalReady && (!r.Development || r.SSHAddress != "") && r.Address != "" &&
		r.CertPEM != "" && r.WorkerID != "" && r.WorkerBootID != "" && r.CreatorPublicKey != ""
}

func (r Rental) Attachable() bool { return r.Ready() }

// EndCause is why an ended rental ended: who released it, else its failure. A Hub older than
// release_cause names a released rental's cause as its detail.
func (r Rental) EndCause() string {
	switch {
	case !RentalAbsent(r.State):
		return ""
	case r.ReleaseCause != "":
		return r.ReleaseCause
	case r.Failure != nil && r.Failure.Code != "":
		return r.Failure.Code
	}
	return r.Detail
}

// HoldsMediaHash answers whether the pod media plane's live set carries this hash.
// Both bare and sha256-prefixed spellings describe the same value.

// wireRental is the answer's own shape.
type wireRental struct {
	Provider              string         `json:"provider,omitempty"`
	ProviderMachineID     string         `json:"provider_machine_id,omitempty"`
	ProviderResourceID    string         `json:"provider_resource_id,omitempty"`
	Development           bool           `json:"development,omitempty"`
	SSHAddress            string         `json:"ssh_address,omitempty"`
	ID                    string         `json:"rental_id"`
	Name                  string         `json:"name"`
	State                 string         `json:"state"`
	AcceleratorModel      string         `json:"requested_accelerator_model"`
	AcceleratorCount      int            `json:"accelerator_count"`
	WorkerAddress         string         `json:"worker_address"`
	CertPEM               string         `json:"cert_pem"`
	Detail                string         `json:"detail"`
	Failure               *RentalFailure `json:"failure"`
	MediaAddress          string         `json:"media_address"`
	WorkerID              string         `json:"worker_id"`
	WorkerBootID          string         `json:"worker_boot_id"`
	CreatorPublicKey      string         `json:"creator_public_key"`
	MediaTokenSHA256      []string       `json:"media_token_sha256"`
	HourlyRateUSDMicros   int64          `json:"hourly_rate_usd_micros"`
	Boot                  *wireBoot      `json:"boot,omitempty"`
	BaseWorkerImageDigest string         `json:"base_worker_image_digest,omitempty"`
	BaseWorkerImageTag    string         `json:"base_worker_image_tag,omitempty"`
	BaseWorkerProfile     string         `json:"base_worker_profile,omitempty"`
	CreatedAt             string         `json:"created_at,omitempty"`
	ContainerDiskGB       int            `json:"container_disk_gb,omitempty"`
	SpendUSDMicros        int64          `json:"spend_usd_micros"`
	SpendBasis            string         `json:"spend_basis"`
	WebRTC                *RentalWebRTC  `json:"webrtc,omitempty"`
	ReleaseCause          string         `json:"release_cause,omitempty"`
	EndedAt               string         `json:"ended_at,omitempty"`
	UnreachableSince      string         `json:"unreachable_since,omitempty"`
	// The rental's hourly cost, split, and its machine's shape.
	ComputeUSDMicrosPerHour int64 `json:"compute_usd_micros_per_hour"`
	StorageUSDMicrosPerHour int64 `json:"storage_usd_micros_per_hour"`
	VCPUCount               int   `json:"vcpu_count"`
	MemoryGB                int   `json:"memory_gb"`
}

var bareSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var computeCapabilityPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+$`)

const maxRentalResponseBytes = 1 << 20

// maxRentalListingBytes bounds the account listing. Each row carries the same view
// the by-id route serves — a pinned certificate included — so the bound is the
// per-rental cap over a fleet, not a new number about how many pods someone may own.
const maxRentalListingBytes = 64 * maxRentalResponseBytes

func validateRentalID(id string) *exit.Error {
	if !rentalid.Valid(id) {
		return exit.Named(exit.Validation, "hub.rental_id_invalid",
			"the hub returned an invalid opaque rental_id").
			WithRemedy("the hub must mint a portable opaque name, never path syntax or a reserved device name")
	}
	return nil
}

// machineName is the Hub's name folded to one a machine can carry, else the id's: a name
// is an alias, never the rental's identity, so an odd one never costs the pod.
func (w wireRental) machineName() string {
	for _, name := range []string{w.Name, w.ID} {
		name = strings.NewReplacer("_", "-", " ", "-").Replace(strings.ToLower(strings.TrimSpace(name)))
		if rentalid.ValidMachineName(name) {
			return name
		}
	}
	return ""
}

func (w wireRental) rental() Rental {
	return Rental{
		Provider: w.Provider, ProviderMachineID: w.ProviderMachineID, ProviderResourceID: w.ProviderResourceID,
		Development: w.Development, SSHAddress: w.SSHAddress, ID: w.ID, Name: w.machineName(), State: w.State,
		AcceleratorModel: w.AcceleratorModel, AcceleratorCount: w.AcceleratorCount,
		Address: w.WorkerAddress, CertPEM: w.CertPEM,
		Detail: w.Detail, Failure: w.Failure, MediaAddress: w.MediaAddress,
		WorkerID: w.WorkerID, WorkerBootID: w.WorkerBootID,
		CreatorPublicKey:      w.CreatorPublicKey,
		MediaTokenSHA256:      w.MediaTokenSHA256,
		HourlyRateUSDMicros:   w.HourlyRateUSDMicros,
		Boot:                  w.Boot.local(time.Now()),
		BaseWorkerImageDigest: w.BaseWorkerImageDigest,
		BaseWorkerImageTag:    w.BaseWorkerImageTag,
		BaseWorkerProfile:     w.BaseWorkerProfile,
		CreatedAt:             w.CreatedAt,
		ContainerDiskGB:       w.ContainerDiskGB,
		SpendUSDMicros:        w.SpendUSDMicros,
		SpendBasis:            w.SpendBasis,
		WebRTC:                w.WebRTC,
		ReleaseCause:          w.ReleaseCause,
		EndedAt:               w.EndedAt,
		UnreachableSince:      w.UnreachableSince,
		// The hourly cost's split and the machine's shape.
		ComputeUSDMicrosPerHour: w.ComputeUSDMicrosPerHour, StorageUSDMicrosPerHour: w.StorageUSDMicrosPerHour,
		VCPUCount: w.VCPUCount, MemoryGB: w.MemoryGB,
	}
}

// RentalRequest is the closed provider-neutral product intent. Provider,
// datacenter, offer, cache volume, and ports do not have fields here: Tensorhub
// resolves and selects them. The workload fields say what the pod is bought for;
// ContainerDiskGB is the renter's explicit disk request.
type RentalDevelopment struct {
	SSHPublicKey string `json:"ssh_public_key"`
}

type RentalRequest struct {
	Development *RentalDevelopment `json:"development,omitempty"`
	Name        string             `json:"name"`
	// Provider names the marketplace (`--provider`); omitted buys from the hub's default.
	Provider                 string   `json:"provider,omitempty"`
	ExcludedProviderMachines []string `json:"excluded_provider_machines,omitempty"`
	SKU                      string   `json:"sku"`
	AcceleratorCount         int      `json:"accelerator_count"`
	MediaTokenSHA256         string   `json:"media_token_sha256"`
	CreatorPublicKey         string   `json:"creator_public_key"`
	// Image names one worker image registered with the hub (digest, tag, or
	// kind) in place of the machine's default; empty boots the default.
	Image string `json:"image,omitempty"`
	// PlannedSourceBytes declares the workload this pod is being bought FOR, so
	// the hub can size its container disk to the job instead of to one constant
	// baked into the worker image (th-152). Omitted for a serving rental, whose
	// models are not resolved yet; SET for an ingest, whose whole selection is
	// resolved before the pod is asked for.
	//
	// It matters because an ingest holds its source objects and the canonical
	// CAS output built from them in ONE Store on the container disk. A 210 GB
	// source needs roughly twice that, and a pod bought at the serving default
	// would not fail cleanly — a full filesystem blocks writes, which looks
	// exactly like a stall, hours into a paid run.
	PlannedSourceBytes int64 `json:"planned_source_bytes,omitempty"`
	// ServingModels declares the models this pod is being bought to SERVE
	// (th-155), so the hub can size its container disk to that set. It carries
	// model IDENTITY and never a byte total: a closure is a set of
	// content-addressed objects, so two models sharing a component share those
	// bytes on disk exactly once, and only the hub — which holds the digests —
	// can take the union. A total computed here would over-count every shared
	// byte or under-count the sharing, and under-counting is a full filesystem
	// hours into a paid run.
	//
	// SET for a rental bought for a request whose models are already resolved,
	// which is every managed request: the models are committed to this store
	// before any pod is asked for, and the same rows become the pod's desired
	// download set. Omitted when nothing is declared.
	ServingModels []ServingModel `json:"serving_models,omitempty"`
	// ContainerDiskGB is the container disk the renter asks for (`--disk-gb`). The Hub
	// buys at least this much, on an offer whose disk allows it. Omitted when not given.
	ContainerDiskGB int `json:"container_disk_gb,omitempty"`
}

// ServingModel is one declared model, in the exact grammar the pod's desired
// download set already speaks (pb.DownloadModelRef's model/release/lane/manifest).
// One vocabulary, so what a rental said it would serve and what its pod asks for
// are comparable without a translation nobody maintains.
type ServingModel struct {
	Lane     string `json:"lane"`
	Manifest string `json:"manifest"`
	Model    string `json:"model"`
	Release  string `json:"release"`
}

// DeclaredWorkload is everything a rental states about the work it is bought
// for. The two halves are independent and additive: an ingest declares source
// bytes, a serving pod declares models, and a pod that does both declares both.
type DeclaredWorkload struct {
	SourceBytes     int64
	ServingModels   []ServingModel
	ContainerDiskGB int
}

// maxServingModels is the hub's cap and the pod's download-set cap, not a new
// number (DownloadSet enforces the same 32).
const maxServingModels = 32

// RentalRequestBytes authors the exact bytes persisted before POST and replayed
// unchanged after response loss. There is one encoder, not a digest struct plus
// a separately marshaled transport map that can drift.
func RentalRequestBytes(name, sku string, gpus int, mediaTokenSHA256, creatorPublicKey string,
	workload DeclaredWorkload, development *RentalDevelopment, image, provider string, excluded ...string,
) ([]byte, *exit.Error) {
	excluded, problem := RentalMachineExclusions(provider, excluded)
	if problem != nil {
		return nil, problem
	}
	req := RentalRequest{
		Development:              development,
		Name:                     strings.TrimSpace(name),
		Provider:                 provider,
		ExcludedProviderMachines: excluded,
		SKU:                      strings.TrimSpace(sku),
		AcceleratorCount:         gpus,
		MediaTokenSHA256:         strings.TrimPrefix(strings.TrimSpace(mediaTokenSHA256), "sha256:"),
		CreatorPublicKey:         strings.TrimSpace(creatorPublicKey),
		PlannedSourceBytes:       workload.SourceBytes,
		ServingModels:            canonicalServingModels(workload.ServingModels),
		ContainerDiskGB:          workload.ContainerDiskGB,
		Image:                    image,
	}
	if len(image) > 512 || strings.TrimSpace(image) != image || strings.ContainsAny(image, " \t\r\n\x00") {
		return nil, exit.Usagef("--image names one registered worker image by digest, tag, or kind")
	}
	if !providerPattern.MatchString(provider) {
		return nil, exit.Usagef("--provider names one marketplace, such as runpod or vast")
	}
	if development != nil && (len(development.SSHPublicKey) == 0 || len(development.SSHPublicKey) > 8192 || strings.TrimSpace(development.SSHPublicKey) != development.SSHPublicKey || strings.ContainsAny(development.SSHPublicKey, "\r\n\x00")) {
		return nil, exit.Usagef("development requires one bounded SSH public-key line")
	}
	if workload.ContainerDiskGB < 0 {
		return nil, exit.Usagef("--disk-gb is a positive number of GB")
	}
	if workload.SourceBytes < 0 {
		return nil, exit.Named(exit.Validation, "rental.planned_workload_invalid",
			"a declared workload is the summed length of the source objects, or absent")
	}
	if len(req.ServingModels) > maxServingModels {
		return nil, exit.Named(exit.Validation, "rental.serving_set_too_large",
			"a rental declares at most %d serving models", maxServingModels)
	}
	for _, model := range req.ServingModels {
		if model.Model == "" || model.Manifest == "" || model.Release == "" && model.Lane != "" {
			return nil, exit.Named(exit.Validation, "rental.serving_model_incomplete",
				"a declared serving model pins a model and an exact manifest; a lane requires a release")
		}
	}
	public, publicErr := base64.RawURLEncoding.DecodeString(req.CreatorPublicKey)
	if !rentalid.ValidMachineName(req.Name) || req.SKU == "" || req.AcceleratorCount < 1 || !bareSHA256Pattern.MatchString(req.MediaTokenSHA256) ||
		publicErr != nil || len(public) != 32 {
		return nil, exit.Named(exit.Validation, "rental.intent_incomplete",
			"a safe name, sku, positive GPU count, media token hash, and one Ed25519 Creator public key are required")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, exit.Internalf("cannot encode the closed rental request: %s", err)
	}
	return raw, nil
}

// RentalMachineExclusions validates an explicit provider-specific development selection.
func RentalMachineExclusions(provider string, values []string) ([]string, *exit.Error) {
	if len(values) == 0 {
		return nil, nil
	}
	if provider != "vast" || len(values) > 64 {
		return nil, exit.Usagef("--exclude-provider-machine requires --provider=vast and at most64 machine IDs")
	}
	out := append([]string(nil), values...)
	for _, id := range out {
		n, err := strconv.ParseInt(id, 10, 64)
		if err != nil || n <= 0 || strconv.FormatInt(n, 10) != id {
			return nil, exit.Usagef("--exclude-provider-machine names a positive Vast machine ID")
		}
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

// ParseRentalRequestBytes reopens the persisted paid intent. Acquisition replay
// resends the persisted bytes unchanged, so a record written by an older or newer
// Creator only needs the fields this build reads: name, SKU and width.
func ParseRentalRequestBytes(raw []byte) (RentalRequest, *exit.Error) {
	var req RentalRequest
	if err := json.Unmarshal(raw, &req); err != nil || !rentalid.ValidMachineName(req.Name) ||
		req.SKU == "" || req.AcceleratorCount < 1 {
		return RentalRequest{}, exit.Named(exit.Conflict, "rental.intent_invalid",
			"persisted rental intent names no machine, SKU and GPU count")
	}
	return req, nil
}

// RentalSKU is one Cozy-priced machine choice: a product at one GPU count. The hub lists
// each product once with its buyable widths; RentalSKUs flattens that into one RentalSKU
// per (Name, AcceleratorCount), the unit every placement decision and paid ask uses.
// Provider offer names and prices are deliberately absent: the caller rents from
// Tensorhub, not its adapter.
type RentalSKU struct {
	Name             string `json:"name"`
	Provider         string `json:"provider,omitempty"`
	AcceleratorModel string `json:"accelerator_model"`
	// AcceleratorCount is the machine's GPU count. VRAMGB stays the ONE-CARD figure at
	// every count, and that is the right fit test: under a sequence-parallel group every
	// GPU holds the FULL weights, so width buys latency, never capacity.
	AcceleratorCount int `json:"accelerator_count"`
	// BaseWorkerProfile is the hub's own label for the base image this product boots,
	// e.g. `torch2.13.0-cu130-cp312-linux-x86`. It is read so a published release whose
	// requirements the label already contradicts is refused before the paid ask. It is
	// deliberately NOT validated: a spelling this client cannot read means one fewer
	// pre-spend check, never an unrentable catalog.
	PythonProvisionableMinors []string `json:"python_provisionable_minors"`
	BaseWorkerProfile         string   `json:"base_worker_profile"`
	ComputeCapability         string   `json:"compute_capability"`
	VRAMGB                    int64    `json:"vram_gb"`
	// PriceUSDMicrosPerHour is the per-machine GPU list rate at this count — the unit the
	// hub's offer matching and replan cap run on, and the accepted quote this client locks.
	PriceUSDMicrosPerHour int64 `json:"price_usd_micros_per_hour"`
	// StorageUSDMicrosPerHour is the hub's estimated per-running-hour storage
	// adder for one pod of this product (th-126): deterministic from the SKU's
	// image disk spec. The renter pays price + storage, so every pre-spend
	// display and the fleet spend admission total the two; the locked quote
	// stays the GPU rate alone. Zero from a hub that itemizes none.
	StorageUSDMicrosPerHour int64 `json:"storage_usd_micros_per_hour"`
	// VCPUCount and MemoryGB are the shape of the machine this count is priced at, where the
	// hub states one (CPU): the same product can be priced at another tier later.
	VCPUCount int `json:"vcpu_count,omitempty"`
	MemoryGB  int `json:"memory_gb,omitempty"`
}

// RentalProduct is one catalog row as the hub publishes it: a product and its widths.
type RentalProduct struct {
	PythonProvisionableMinors []string      `json:"python_provisionable_minors"`
	Name                      string        `json:"name"`
	Provider                  string        `json:"provider,omitempty"`
	AcceleratorModel          string        `json:"accelerator_model"`
	BaseWorkerProfile         string        `json:"base_worker_profile"`
	ComputeCapability         string        `json:"compute_capability"`
	VRAMGB                    int64         `json:"vram_gb"`
	Widths                    []RentalWidth `json:"widths"`
}

// RentalWidth is one buyable GPU count and its per-machine prices.
type RentalWidth struct {
	AcceleratorCount        int   `json:"accelerator_count"`
	PriceUSDMicrosPerHour   int64 `json:"price_usd_micros_per_hour"`
	StorageUSDMicrosPerHour int64 `json:"storage_usd_micros_per_hour"`
	VCPUCount               int   `json:"vcpu_count,omitempty"`
	MemoryGB                int   `json:"memory_gb,omitempty"`
}

// Machines flattens the product into one RentalSKU per width.
func (p RentalProduct) Machines() []RentalSKU {
	out := make([]RentalSKU, 0, len(p.Widths))
	for _, width := range p.Widths {
		out = append(out, RentalSKU{Name: p.Name, Provider: p.Provider, AcceleratorModel: p.AcceleratorModel,
			AcceleratorCount: width.AcceleratorCount, PythonProvisionableMinors: p.PythonProvisionableMinors,
			BaseWorkerProfile: p.BaseWorkerProfile,
			ComputeCapability: p.ComputeCapability, VRAMGB: p.VRAMGB,
			PriceUSDMicrosPerHour: width.PriceUSDMicrosPerHour, StorageUSDMicrosPerHour: width.StorageUSDMicrosPerHour,
			VCPUCount: width.VCPUCount, MemoryGB: width.MemoryGB})
	}
	return out
}

// RentalProducts groups machines back into catalog rows, in first-seen product order
// with widths ascending. It is the listing's shape: a product once, its counts beside it.
func RentalProducts(skus []RentalSKU) []RentalProduct {
	index := map[string]int{}
	var out []RentalProduct
	for _, sku := range skus {
		i, seen := index[sku.Name]
		if !seen {
			i = len(out)
			index[sku.Name] = i
			out = append(out, RentalProduct{Name: sku.Name, AcceleratorModel: sku.AcceleratorModel,
				PythonProvisionableMinors: sku.PythonProvisionableMinors,
				BaseWorkerProfile:         sku.BaseWorkerProfile, ComputeCapability: sku.ComputeCapability,
				VRAMGB: sku.VRAMGB})
		}
		out[i].Widths = append(out[i].Widths, RentalWidth{AcceleratorCount: sku.AcceleratorCount,
			PriceUSDMicrosPerHour: sku.PriceUSDMicrosPerHour, StorageUSDMicrosPerHour: sku.StorageUSDMicrosPerHour,
			VCPUCount: sku.VCPUCount, MemoryGB: sku.MemoryGB})
	}
	for i := range out {
		sort.Slice(out[i].Widths, func(a, b int) bool {
			return out[i].Widths[a].AcceleratorCount < out[i].Widths[b].AcceleratorCount
		})
	}
	return out
}

// Counts lists the product's buyable GPU counts.
func (p RentalProduct) Counts() []int {
	out := make([]int, 0, len(p.Widths))
	for _, width := range p.Widths {
		out = append(out, width.AcceleratorCount)
	}
	return out
}

// FindRentalSKU is the machine for (name, gpus) in a flattened catalog.
func FindRentalSKU(skus []RentalSKU, name string, gpus int) (RentalSKU, bool) {
	for _, sku := range skus {
		if sku.Name == name && sku.AcceleratorCount == gpus {
			return sku, true
		}
	}
	return RentalSKU{}, false
}

// RentalSKUs reads Tensorhub's public product catalog, flattened to one machine per width.
// A product or width this build cannot price or size is left out; one bad row never
// hides the rest of the catalog.
// RentalSKUs lists what one provider sells now; "" is the hub's default. A hub that
// predates providers answers its default listing, which names no provider, so an
// ask for a named one finds nothing there rather than a different marketplace.
func (c *Client) RentalSKUs(ctx context.Context, provider string) ([]RentalSKU, *exit.Error) {
	var products []RentalProduct
	if e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rental-skus" + providerQuery("?", provider)}, &products); e != nil {
		return nil, e
	}
	if provider != "" {
		products = slices.DeleteFunc(products, func(p RentalProduct) bool { return p.Provider != provider })
	}
	var out []RentalSKU
	seen := map[string]bool{}
	for _, product := range products {
		cpu := product.AcceleratorModel == "CPU"
		// Host RAM is an informational fact: placement fits GPUs only, so neither a
		// RAM figure nor its absence decides whether a product can be rented.
		invalidCPU := cpu && (product.ComputeCapability != "" || product.VRAMGB != 0)
		invalidGPU := !cpu && (!computeCapabilityPattern.MatchString(product.ComputeCapability) || product.VRAMGB <= 0)
		if strings.TrimSpace(product.Name) == "" || strings.TrimSpace(product.AcceleratorModel) == "" ||
			invalidCPU || invalidGPU || seen[product.Name] {
			continue
		}
		widths := map[int]bool{}
		valid := product.Widths[:0:0]
		for _, width := range product.Widths {
			if width.AcceleratorCount < 1 || widths[width.AcceleratorCount] || cpu && width.AcceleratorCount != 1 ||
				width.PriceUSDMicrosPerHour <= 0 || width.StorageUSDMicrosPerHour < 0 {
				continue
			}
			widths[width.AcceleratorCount] = true
			valid = append(valid, width)
		}
		if len(valid) == 0 {
			continue
		}
		seen[product.Name] = true
		product.Widths = valid
		out = append(out, product.Machines()...)
	}
	return out, nil
}

// Rent asks for a pod, presenting the media bearer HASH and Creator public key minted before the ask. It
// returns as soon as the hub has ACCEPTED the ask — provisioning is the hub's work and
// this client watches it through `Rental`, because a POST that blocked until a pod booted
// would be a request whose failure mode is a lost id.
//
// `mediaTokenSHA256` is the BARE 64 lowercase hex — the spelling the hub's field takes; the
// `sha256:` prefix is the token-hash FILE's spelling and belongs to the pod, not the wire.
// The bearer itself is not an argument here or anywhere on this wire.
// Rent retransmits the exact canonical bytes the caller persisted before the
// paid mutation. The hub authority is bound separately by the local operation
// digest; a retry against another hub therefore conflicts before this method.
//
// THE SECOND RESULT IS THE ONE THAT COSTS MONEY. It says the hub ANSWERED the ask:
// a 2xx create answer means a pod may exist and bill from that moment, whatever this
// client then makes of the body. Only a refusal that created nothing answers false,
// and only that may be recorded as an ask that bought nothing. An answered ask whose
// body this client cannot use still comes back with whatever identity it carried —
// that identity is the only name the pod has here, and destroying it because a
// sibling field was missing is how six paid pods came to boot unattended (cl-192).
func (c *Client) Rent(ctx context.Context, requestBody []byte, reason, operationKey string) (Rental, bool, *exit.Error) {
	var out wireRental
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/rentals", auth: true, reason: reason,
		idempotency: operationKey, bodyBytes: requestBody, responseBytes: maxRentalResponseBytes,
	}, &out)
	if e != nil {
		return Rental{}, false, e
	}
	if e := out.named("accepted the rental"); e != nil {
		return out.rental(), true, e
	}
	return out.rental(), true, nil
}

func (w wireRental) named(what string) *exit.Error {
	if w.ID == "" {
		return exit.Named(exit.Internal, "hub.rental_unnamed",
			"the hub %s and named no rental_id", what).
			WithRemedy("this route may not exist on this hub build; `cozy package search` names it and its version")
	}
	if w.HourlyRateUSDMicros <= 0 {
		return exit.Named(exit.Conflict, "hub.rental_hourly_rate_missing",
			"the hub %s without a positive locked Cozy retail hourly rate", what).
			WithRemedy("upgrade Tensorhub before accepting a rental")
	}
	// THE WIDTH IS NOT OPTIONAL. It decides the device envelope this host grants the pod's
	// worker and the degree it pins a group placement to, so a rental that will not say how
	// many accelerators it delivers cannot be attached at all — and reading its silence as
	// one would attach a wide paid pod as a single card and idle the rest.
	if w.AcceleratorCount < 1 {
		return exit.Named(exit.Conflict, "hub.rental_accelerator_count_missing",
			"the hub %s with accelerator count %d", what, w.AcceleratorCount).
			WithRemedy("upgrade Tensorhub; every rental states the width it delivers")
	}
	return validateRentalID(w.ID)
}

// Rentals is every rental THE HUB says this account owns (th-199). It is the only
// answer to a question Creator cannot answer from its own records: a pod bought by
// this account that this host never recorded — because the create answer was
// refused, because the records were lost, or because another host bought it — is
// billing and is nameable nowhere else. Six H100 NVLs proved that on 2026-09-07.
//
// The second result says whether the hub PUBLISHES a listing. A hub older than
// th-199 has no such route and answers 404 for it, which is a statement about the
// hub's version and not about the account's pods; a caller must not read it as "you
// own nothing", so the two are separated here rather than collapsed into an empty
// slice. Nothing else is inferred from an absent route.
//
// Only a row whose id is unusable is DROPPED, rather than failing the listing: a hub that
// serves one malformed row must not thereby hide the ten good ones (cl-193). An odd name
// is folded or replaced by the id, since a billing pod must stay nameable.
func (c *Client) Rentals(ctx context.Context) ([]Rental, *exit.Error) {
	rentals, _, problem := c.RentalListing(ctx)
	return rentals, problem
}

// RentalListing is Rentals with the account's bindings revision the listing carries (0 from a
// Hub that states none): a rebind from another computer or the Hub moves it.
func (c *Client) RentalListing(ctx context.Context) ([]Rental, int64, *exit.Error) {
	var out struct {
		Rentals          []wireRental `json:"rentals"`
		BindingsRevision int64        `json:"bindings_revision"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals", auth: true,
		responseBytes: maxRentalListingBytes}, &out)
	if e != nil {
		return nil, 0, e
	}
	rentals := make([]Rental, 0, len(out.Rentals))
	for _, wire := range out.Rentals {
		if !rentalid.Valid(wire.ID) {
			continue
		}
		rentals = append(rentals, wire.rental())
	}
	return rentals, out.BindingsRevision, nil
}

// RentalHistory is every rental of this account that bore `name` (any name when blank), ended
// ones included, newest first. A Hub that does not filter by name answers its whole history;
// the filter here holds.
func (c *Client) RentalHistory(ctx context.Context, name string) ([]Rental, *exit.Error) {
	var out struct {
		Rentals []wireRental `json:"rentals"`
	}
	query := url.Values{"state": {"all"}}
	if name != "" {
		query.Set("name", strings.ToLower(name))
	}
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals?" + query.Encode(), auth: true,
		responseBytes: maxRentalListingBytes}, &out)
	if e != nil {
		return nil, e
	}
	var named []Rental
	for _, wire := range out.Rentals {
		if rental := wire.rental(); rentalid.Valid(wire.ID) && (name == "" || strings.EqualFold(rental.Name, name)) {
			named = append(named, rental)
		}
	}
	created := func(r Rental) time.Time { at, _ := time.Parse(time.RFC3339Nano, r.CreatedAt); return at }
	slices.SortStableFunc(named, func(a, b Rental) int { return created(b).Compare(created(a)) })
	return named, nil
}

// Rental reads one rental's current state using the renter's account authority.
func (c *Client) Rental(ctx context.Context, id string) (Rental, *exit.Error) {
	if e := validateRentalID(id); e != nil {
		return Rental{}, e
	}
	var out wireRental
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals/" + url.PathEscape(id),
		auth: true, responseBytes: maxRentalResponseBytes}, &out)
	if e != nil {
		return Rental{}, e
	}
	if e := out.named("answered rental " + id); e != nil {
		return Rental{}, e
	}
	if out.ID != id {
		return Rental{}, exit.Named(exit.Conflict, "hub.rental_id_changed",
			"the hub answered rental %s with identity %s", id, out.ID).
			WithRemedy("preserve the original rental identity; never attach the response under another id")
	}
	return out.rental(), nil
}

// RentalView reads a rental for a decision that does not depend on what the pod IS —
// whether it still exists, and tearing it down. It validates IDENTITY and nothing else.
//
// The strict read above is right for attachment: a pod whose width the hub will not state
// must not be attached. It is wrong for a release, and dangerously so — a fence meant to
// protect an attachment then stands between an operator and a machine that is billing,
// refusing the one command that stops the money over a field the teardown never reads.
func (c *Client) RentalView(ctx context.Context, id string) (Rental, *exit.Error) {
	if e := validateRentalID(id); e != nil {
		return Rental{}, e
	}
	var out wireRental
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals/" + url.PathEscape(id),
		auth: true, responseBytes: maxRentalResponseBytes}, &out)
	if e != nil {
		return Rental{}, e
	}
	if out.ID != id {
		return Rental{}, exit.Named(exit.Conflict, "hub.rental_id_changed",
			"the hub answered rental %s with identity %s", id, out.ID).
			WithRemedy("preserve the original rental identity; never attach the response under another id")
	}
	return out.rental(), nil
}

// Release asks the hub to tear the pod down. The hub answers 204 and nothing is decoded;
// a 404 reaches the caller as NotFound, and the caller decides what absence means.
func (c *Client) Release(ctx context.Context, id, reason string) *exit.Error {
	if e := validateRentalID(id); e != nil {
		return e
	}
	return c.do(ctx, call{
		method: http.MethodDelete, path: "/v1/rentals/" + url.PathEscape(id), auth: true, reason: reason,
	}, nil)
}

// canonicalServingModels sorts and de-duplicates the declared set so the authored
// bytes are a pure function of the selection: the same models in any order
// author the same request, and a replay re-authors byte-identically. An empty
// set is nil so the key stays OFF the wire and an undeclared rental is
// byte-identical to one authored before th-155.
func canonicalServingModels(models []ServingModel) []ServingModel {
	if len(models) == 0 {
		return nil
	}
	key := func(m ServingModel) string {
		return m.Model + "\x00" + m.Release + "\x00" + m.Manifest + "\x00" + m.Lane
	}
	sorted := append([]ServingModel(nil), models...)
	sort.Slice(sorted, func(i, j int) bool { return key(sorted[i]) < key(sorted[j]) })
	out := sorted[:0]
	prior := ""
	for _, model := range sorted {
		if current := key(model); current != prior {
			out = append(out, model)
			prior = current
		}
	}
	return out
}

// RentalSKUStatus is one product name's standing (th-150), and it is asked ONLY
// on the refusal path: the catalog is live provider inventory, so a name missing
// from RentalSKUs may be a product this hub never sells OR a real one whose
// inventory is momentarily empty or temporarily excluded after boot failures.
// The status distinguishes these reasons without initiating another rental.
//
// A hub too old to serve the route answers 404; that is not an error worth
// failing a refusal over, so the caller gets an empty status and says the plain
// thing instead.
type RentalSKUStatus struct {
	Name              string         `json:"name"`
	AcceleratorCount  int            `json:"accelerator_count"`
	Known             bool           `json:"known"`
	Offered           bool           `json:"offered"`
	SKU               *RentalProduct `json:"sku,omitempty"`
	LastSeenAt        *time.Time     `json:"last_seen_at,omitempty"`
	UnavailableReason string         `json:"unavailable_reason,omitempty"`
	RetryAfter        *time.Time     `json:"retry_after,omitempty"`
}

// RentalQuote is what one exact rental request will lock: its disk and declared
// workload choose the machine, so it can differ from the default-disk listing.
type RentalQuote struct {
	ExcludedProviderMachines []string `json:"excluded_provider_machines,omitempty"`
	PriceUSDMicrosPerHour    int64    `json:"price_usd_micros_per_hour"`
	StorageUSDMicrosPerHour  int64    `json:"storage_usd_micros_per_hour"`
	ContainerDiskGB          int      `json:"container_disk_gb"`
	VCPUCount                int      `json:"vcpu_count,omitempty"`
	MemoryGB                 int      `json:"memory_gb,omitempty"`
}

// QuoteRental prices the exact body a rental POST would send.
func (c *Client) QuoteRental(ctx context.Context, body []byte) (RentalQuote, *exit.Error) {
	var out RentalQuote
	// The quote sizes the owner's own unpublished checkpoints only for a signed-in owner.
	e := c.do(ctx, call{method: http.MethodPost, path: "/v1/rental-quotes", bodyBytes: body, optionalAuth: true}, &out)
	return out, e
}

func (c *Client) RentalSKUStatus(ctx context.Context, provider, name string, gpus int) (RentalSKUStatus, *exit.Error) {
	var out RentalSKUStatus
	if e := c.do(ctx, call{method: http.MethodGet,
		path: "/v1/rental-skus/" + url.PathEscape(name) + "?accelerator_count=" + strconv.Itoa(gpus) + providerQuery("&", provider)}, &out); e != nil {
		return RentalSKUStatus{}, e
	}
	return out, nil
}

var providerPattern = regexp.MustCompile(`^([a-z][a-z0-9-]{0,31})?$`)

func providerQuery(sep, provider string) string {
	if provider == "" {
		return ""
	}
	return sep + "provider=" + url.QueryEscape(provider)
}
