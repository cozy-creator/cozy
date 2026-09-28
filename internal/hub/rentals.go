package hub

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/rentalid"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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
	Development      bool
	SSHAddress       string
	ID               string
	Name             string
	State            string
	AcceleratorModel string
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
	// ProviderState and ContainerState are the provider's own lifecycle words for this
	// pod, as the hub last observed them. They are here for one reason: without them the
	// whole interval between "renting" and "attachable" is a single edge, and a person
	// watching it cannot tell a provider queue from a multi-gigabyte image pull. They are
	// the same class of fact the hub already publishes inside RentalFailure — a provider
	// lifecycle fact the hub observed — and carry no acquisition identity of their own.
	// Blank whenever the hub has not observed the pod yet, or is older than the field.
	ProviderState         string
	ContainerState        string
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

// Ready answers whether this rental carries the whole dial triple and an observed
// media credential set. The caller still checks that set contains its bearer hash; Ready only
// prevents a partial ready projection from being mistaken for a usable pod.
func (r Rental) Ready() bool {
	return r.State == RentalReady && (!r.Development || r.SSHAddress != "") && r.Address != "" && r.MediaAddress != "" &&
		r.CertPEM != "" && r.WorkerID != "" && r.WorkerBootID != "" && r.CreatorPublicKey != "" &&
		len(r.MediaTokenSHA256) > 0
}

func (r Rental) Attachable() bool { return r.Ready() }

// HoldsMediaHash answers whether the pod media plane's live set carries this hash.
// Both bare and sha256-prefixed spellings describe the same value.
func (r Rental) HoldsMediaHash(hash string) bool {
	bare := strings.TrimPrefix(hash, "sha256:")
	for _, h := range r.MediaTokenSHA256 {
		if strings.TrimPrefix(h, "sha256:") == bare {
			return true
		}
	}
	return false
}

// wireRental is the answer's own shape.
type wireRental struct {
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
	ProviderState         string         `json:"provider_state,omitempty"`
	ContainerState        string         `json:"container_state,omitempty"`
	BaseWorkerImageDigest string         `json:"base_worker_image_digest,omitempty"`
	BaseWorkerImageTag    string         `json:"base_worker_image_tag,omitempty"`
	BaseWorkerProfile     string         `json:"base_worker_profile,omitempty"`
	CreatedAt             string         `json:"created_at,omitempty"`
	ContainerDiskGB       int            `json:"container_disk_gb,omitempty"`
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

func (w wireRental) rental() Rental {
	return Rental{
		Development: w.Development, SSHAddress: w.SSHAddress, ID: w.ID, Name: w.Name, State: w.State,
		AcceleratorModel: w.AcceleratorModel, AcceleratorCount: w.AcceleratorCount,
		Address: w.WorkerAddress, CertPEM: w.CertPEM,
		Detail: w.Detail, Failure: w.Failure, MediaAddress: w.MediaAddress,
		WorkerID: w.WorkerID, WorkerBootID: w.WorkerBootID,
		CreatorPublicKey:      w.CreatorPublicKey,
		MediaTokenSHA256:      w.MediaTokenSHA256,
		HourlyRateUSDMicros:   w.HourlyRateUSDMicros,
		ProviderState:         w.ProviderState,
		ContainerState:        w.ContainerState,
		BaseWorkerImageDigest: w.BaseWorkerImageDigest,
		BaseWorkerImageTag:    w.BaseWorkerImageTag,
		BaseWorkerProfile:     w.BaseWorkerProfile,
		CreatedAt:             w.CreatedAt,
		ContainerDiskGB:       w.ContainerDiskGB,
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
	Development      *RentalDevelopment `json:"development,omitempty"`
	Name             string             `json:"name"`
	SKU              string             `json:"sku"`
	AcceleratorCount int                `json:"accelerator_count"`
	MediaTokenSHA256 string             `json:"media_token_sha256"`
	CreatorPublicKey string             `json:"creator_public_key"`
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
	workload DeclaredWorkload, development *RentalDevelopment, image string,
) ([]byte, *exit.Error) {
	req := RentalRequest{
		Development:        development,
		Name:               strings.TrimSpace(name),
		SKU:                strings.TrimSpace(sku),
		AcceleratorCount:   gpus,
		MediaTokenSHA256:   strings.TrimPrefix(strings.TrimSpace(mediaTokenSHA256), "sha256:"),
		CreatorPublicKey:   strings.TrimSpace(creatorPublicKey),
		PlannedSourceBytes: workload.SourceBytes,
		ServingModels:      canonicalServingModels(workload.ServingModels),
		ContainerDiskGB:    workload.ContainerDiskGB,
		Image:              image,
	}
	if len(image) > 512 || strings.TrimSpace(image) != image || strings.ContainsAny(image, " \t\r\n\x00") {
		return nil, exit.Usagef("--image names one registered worker image by digest, tag, or kind")
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
	AcceleratorModel string `json:"accelerator_model"`
	// AcceleratorCount is the machine's GPU count. VRAMGB stays the ONE-CARD figure at
	// every count, and that is the right fit test: under a sequence-parallel group every
	// rank holds the FULL weights, so width buys latency, never capacity.
	AcceleratorCount int `json:"accelerator_count"`
	// BaseWorkerProfile is the hub's own label for the base image this product boots,
	// e.g. `torch2.13.0-cu130-cp312-linux-x86`. It is read so a published release whose
	// requirements the label already contradicts is refused before the paid ask. It is
	// deliberately NOT validated: a spelling this client cannot read means one fewer
	// pre-spend check, never an unrentable catalog.
	PythonProvisionableMinors []string                `json:"python_provisionable_minors"`
	PythonInterpreters        []*pb.PythonInterpreter `json:"python_interpreters"`
	BaseWorkerProfile         string                  `json:"base_worker_profile"`
	ComputeCapability         string                  `json:"compute_capability"`
	VRAMGB                    int64                   `json:"vram_gb"`
	// PriceUSDMicrosPerHour is the per-machine GPU list rate at this count — the unit the
	// hub's offer matching and replan cap run on, and the accepted quote this client locks.
	PriceUSDMicrosPerHour int64 `json:"price_usd_micros_per_hour"`
	// StorageUSDMicrosPerHour is the hub's estimated per-running-hour storage
	// adder for one pod of this product (th-126): deterministic from the SKU's
	// image disk spec. The renter pays price + storage, so every pre-spend
	// display and the fleet spend admission total the two; the locked quote
	// stays the GPU rate alone. Zero from a hub that itemizes none.
	StorageUSDMicrosPerHour int64 `json:"storage_usd_micros_per_hour"`
}

// RentalProduct is one catalog row as the hub publishes it: a product and its widths.
type RentalProduct struct {
	PythonProvisionableMinors []string                `json:"python_provisionable_minors"`
	PythonInterpreters        []*pb.PythonInterpreter `json:"python_interpreters"`
	Name                      string                  `json:"name"`
	AcceleratorModel          string                  `json:"accelerator_model"`
	BaseWorkerProfile         string                  `json:"base_worker_profile"`
	ComputeCapability         string                  `json:"compute_capability"`
	VRAMGB                    int64                   `json:"vram_gb"`
	Widths                    []RentalWidth           `json:"widths"`
}

// RentalWidth is one buyable GPU count and its per-machine prices.
type RentalWidth struct {
	AcceleratorCount        int   `json:"accelerator_count"`
	PriceUSDMicrosPerHour   int64 `json:"price_usd_micros_per_hour"`
	StorageUSDMicrosPerHour int64 `json:"storage_usd_micros_per_hour"`
}

// Machines flattens the product into one RentalSKU per width.
func (p RentalProduct) Machines() []RentalSKU {
	out := make([]RentalSKU, 0, len(p.Widths))
	for _, width := range p.Widths {
		out = append(out, RentalSKU{Name: p.Name, AcceleratorModel: p.AcceleratorModel,
			AcceleratorCount: width.AcceleratorCount, PythonProvisionableMinors: p.PythonProvisionableMinors,
			PythonInterpreters: p.PythonInterpreters, BaseWorkerProfile: p.BaseWorkerProfile,
			ComputeCapability: p.ComputeCapability, VRAMGB: p.VRAMGB,
			PriceUSDMicrosPerHour: width.PriceUSDMicrosPerHour, StorageUSDMicrosPerHour: width.StorageUSDMicrosPerHour})
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
				PythonProvisionableMinors: sku.PythonProvisionableMinors, PythonInterpreters: sku.PythonInterpreters,
				BaseWorkerProfile: sku.BaseWorkerProfile, ComputeCapability: sku.ComputeCapability,
				VRAMGB: sku.VRAMGB})
		}
		out[i].Widths = append(out[i].Widths, RentalWidth{AcceleratorCount: sku.AcceleratorCount,
			PriceUSDMicrosPerHour: sku.PriceUSDMicrosPerHour, StorageUSDMicrosPerHour: sku.StorageUSDMicrosPerHour})
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
func (c *Client) RentalSKUs(ctx context.Context) ([]RentalSKU, *exit.Error) {
	var products []RentalProduct
	if e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rental-skus"}, &products); e != nil {
		return nil, e
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
	if !rentalid.ValidMachineName(w.Name) {
		return exit.Named(exit.Conflict, "hub.rental_name_invalid",
			"the hub %s with invalid private rental name %q", what, w.Name).
			WithRemedy("Tensorhub must return the exact safe name Creator sent at rental creation")
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
// Identity is validated per row and a row that cannot be named is DROPPED rather
// than failing the listing: a hub that serves one malformed row must not thereby
// hide the ten good ones, which is the same lesson as the release path's (cl-193).
func (c *Client) Rentals(ctx context.Context) ([]Rental, *exit.Error) {
	var out struct {
		Rentals []wireRental `json:"rentals"`
	}
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals", auth: true,
		responseBytes: maxRentalListingBytes}, &out)
	if e != nil {
		return nil, e
	}
	rentals := make([]Rental, 0, len(out.Rentals))
	for _, wire := range out.Rentals {
		if !rentalid.Valid(wire.ID) || !rentalid.ValidMachineName(wire.Name) {
			continue
		}
		rentals = append(rentals, wire.rental())
	}
	return rentals, nil
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
	PriceUSDMicrosPerHour   int64 `json:"price_usd_micros_per_hour"`
	StorageUSDMicrosPerHour int64 `json:"storage_usd_micros_per_hour"`
	ContainerDiskGB         int   `json:"container_disk_gb"`
}

// QuoteRental prices the exact body a rental POST would send.
func (c *Client) QuoteRental(ctx context.Context, body []byte) (RentalQuote, *exit.Error) {
	var out RentalQuote
	e := c.do(ctx, call{method: http.MethodPost, path: "/v1/rental-quotes", bodyBytes: body}, &out)
	return out, e
}

func (c *Client) RentalSKUStatus(ctx context.Context, name string, gpus int) (RentalSKUStatus, *exit.Error) {
	var out RentalSKUStatus
	if e := c.do(ctx, call{method: http.MethodGet,
		path: "/v1/rental-skus/" + url.PathEscape(name) + "?accelerator_count=" + strconv.Itoa(gpus)}, &out); e != nil {
		return RentalSKUStatus{}, e
	}
	return out, nil
}
