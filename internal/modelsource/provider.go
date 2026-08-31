package modelsource

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/secret"
)

const (
	metadataLimit = 8 << 20
	indexLimit    = 32 << 20
	maxFiles      = 4096
	maxSourceSize = int64(2) << 40
)

type File struct {
	Member   string
	URL      string
	SHA256   string
	Length   int64
	Inline   []byte
	Carrier  bool
	Requires []string
}

type Plan struct {
	Source          Source
	Canonical       string
	SelectionSHA256 string
	License         string
	Files           []File
	Bytes           int64
}

type Resolver struct {
	kind  Kind
	token secret.Value
	http  *http.Client
}

func NewResolver(kind Kind, token secret.Value) (*Resolver, *exit.Error) {
	if kind != HuggingFace && kind != Civitai {
		return nil, exit.Internalf("provider resolver cannot serve source kind %q", kind)
	}
	r := &Resolver{kind: kind, token: token}
	r.http = hardenedClient(kind)
	return r, nil
}

func (r *Resolver) Resolve(ctx context.Context, source Source) (Plan, *exit.Error) {
	switch source.Kind {
	case HuggingFace:
		return r.resolveHF(ctx, source)
	case Civitai:
		return r.resolveCivitai(ctx, source)
	default:
		return Plan{}, exit.Internalf("provider resolver received non-provider source")
	}
}

func hardenedClient(kind Kind) *http.Client {
	dialer := &net.Dialer{Timeout: 15 * time.Second, KeepAlive: 30 * time.Second}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	transport.TLSHandshakeTimeout = 15 * time.Second
	transport.ResponseHeaderTimeout = 30 * time.Second
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, err
		}
		addresses, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
		if err != nil || len(addresses) == 0 {
			return nil, fmt.Errorf("resolve %s: %w", host, err)
		}
		for _, address := range addresses {
			if !publicAddress(address) {
				return nil, fmt.Errorf("host %s resolved to refused address %s", host, address)
			}
		}
		var last error
		for _, resolved := range addresses {
			conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.String(), port))
			if err == nil {
				return conn, nil
			}
			last = err
		}
		return nil, last
	}
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 || req.URL.Scheme != "https" || req.URL.User != nil ||
				!allowedHost(kind, req.URL.Hostname()) {
				return fmt.Errorf("refused provider redirect")
			}
			if !authorizationHost(kind, req.URL.Hostname()) {
				req.Header.Del("Authorization")
			}
			return nil
		},
	}
}

func publicAddress(address netip.Addr) bool {
	return address.IsValid() && !address.IsPrivate() && !address.IsLoopback() &&
		!address.IsLinkLocalUnicast() && !address.IsLinkLocalMulticast() &&
		!address.IsMulticast() && !address.IsUnspecified()
}

func allowedHost(kind Kind, host string) bool {
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	switch kind {
	case HuggingFace:
		return host == "huggingface.co" || strings.HasSuffix(host, ".huggingface.co") ||
			host == "hf.co" || strings.HasSuffix(host, ".hf.co") ||
			host == "xethub.hf.co" || strings.HasSuffix(host, ".xethub.hf.co")
	case Civitai:
		return host == "civitai.com" || strings.HasSuffix(host, ".civitai.com") ||
			host == "civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com"
	}
	return false
}

func authorizationHost(kind Kind, host string) bool {
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	if kind == HuggingFace {
		return host == "huggingface.co" || host == "www.huggingface.co"
	}
	return host == "civitai.com" || host == "www.civitai.com"
}

func (r *Resolver) request(ctx context.Context, method, location string, headers http.Header) (*http.Response, *exit.Error) {
	u, err := url.Parse(location)
	if err != nil || u.Scheme != "https" || u.User != nil || !allowedHost(r.kind, u.Hostname()) {
		return nil, exit.Named(exit.Validation, "model_source_url_refused",
			"provider returned a non-allowlisted URL")
	}
	req, err := http.NewRequestWithContext(ctx, method, location, nil)
	if err != nil {
		return nil, exit.Internalf("cannot create provider request: %s", err)
	}
	req.Header.Set("Accept-Encoding", "identity")
	for key, values := range headers {
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	if r.token.Present() && authorizationHost(r.kind, u.Hostname()) {
		req.Header.Set("Authorization", "Bearer "+r.token.Reveal()) //cozy:allow-reveal provider token becomes an Authorization header only here
	}
	response, err := r.http.Do(req)
	if err != nil {
		return nil, exit.Unavailablef("provider request failed before a valid response")
	}
	if encoding := response.Header.Get("Content-Encoding"); encoding != "" && encoding != "identity" {
		response.Body.Close()
		return nil, exit.Named(exit.Validation, "model_source_content_encoding",
			"provider answered with unsupported content encoding %q", encoding)
	}
	return response, nil
}

func (r *Resolver) json(ctx context.Context, location string, value any) *exit.Error {
	response, problem := r.request(ctx, http.MethodGet, location, nil)
	if problem != nil {
		return problem
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden {
		return exit.Named(exit.Credential, "model_source_credential_required",
			"provider refused access to the model source").
			WithRemedy("configure the matching provider token or use --token-stdin for this local import")
	}
	if response.StatusCode != http.StatusOK {
		return exit.Named(exit.Validation, "model_source_metadata_refused",
			"provider metadata request answered HTTP %d", response.StatusCode)
	}
	if response.ContentLength > metadataLimit {
		return exit.Named(exit.Validation, "model_source_metadata_too_large",
			"provider metadata declares %d bytes", response.ContentLength)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, metadataLimit+1))
	if err != nil || len(raw) > metadataLimit {
		return exit.Named(exit.Validation, "model_source_metadata_too_large",
			"provider metadata exceeded its %d-byte bound", metadataLimit)
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	if err := decoder.Decode(value); err != nil {
		return exit.Named(exit.Validation, "model_source_metadata_invalid",
			"provider metadata is not one bounded JSON object: %s", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return exit.Named(exit.Validation, "model_source_metadata_invalid",
			"provider metadata contains trailing JSON")
	}
	return nil
}

type hfModel struct {
	SHA      string `json:"sha"`
	CardData struct {
		License string `json:"license"`
	} `json:"cardData"`
	Siblings []struct {
		Name string `json:"rfilename"`
		Size int64  `json:"size"`
		LFS  *struct {
			OID    string `json:"oid"`
			SHA256 string `json:"sha256"`
			Size   int64  `json:"size"`
		} `json:"lfs"`
	} `json:"siblings"`
}

func (r *Resolver) resolveHF(ctx context.Context, source Source) (Plan, *exit.Error) {
	reference := source.Reference
	if reference == "" {
		reference = "main"
	}
	api := "https://huggingface.co/api/models/" + url.PathEscape(source.Org) + "/" +
		url.PathEscape(source.Repo) + "/revision/" + url.PathEscape(reference) + "?blobs=true"
	var metadata hfModel
	if problem := r.json(ctx, api, &metadata); problem != nil {
		return Plan{}, problem
	}
	if !fullCommit(metadata.SHA) || len(metadata.Siblings) == 0 || len(metadata.Siblings) > maxFiles {
		return Plan{}, exit.Named(exit.Validation, "model_source_metadata_incomplete",
			"Hugging Face metadata lacks one full commit or has an invalid member count")
	}
	commit := strings.ToLower(metadata.SHA)
	base := "https://huggingface.co/" + url.PathEscape(source.Org) + "/" +
		url.PathEscape(source.Repo) + "/resolve/" + commit + "/"
	byMember := make(map[string]File)
	indexes := make([]string, 0)
	for _, sibling := range metadata.Siblings {
		if !safeMember(sibling.Name) {
			continue
		}
		lower := strings.ToLower(sibling.Name)
		if !strings.HasSuffix(lower, ".safetensors") && !strings.HasSuffix(lower, ".safetensors.index.json") {
			continue
		}
		file := File{Member: sibling.Name, URL: base + escapeMember(sibling.Name)}
		if sibling.LFS != nil {
			file.Length = sibling.LFS.Size
			file.SHA256 = strings.TrimPrefix(strings.ToLower(first(sibling.LFS.SHA256, sibling.LFS.OID)), "sha256:")
		} else {
			file.Length = sibling.Size
		}
		if strings.HasSuffix(lower, ".index.json") {
			indexes = append(indexes, sibling.Name)
		}
		byMember[sibling.Name] = file
	}
	if len(byMember) == 0 {
		return Plan{}, exit.Named(exit.Validation, "model_source_files_absent",
			"Hugging Face revision contains no supported tensor carriers")
	}
	referenced := make(map[string]bool)
	for _, member := range indexes {
		file := byMember[member]
		body, problem := r.small(ctx, file.URL, indexLimit)
		if problem != nil {
			return Plan{}, problem
		}
		file.Inline = body
		file.Length = int64(len(body))
		file.SHA256 = digest(body)
		file.Carrier = true
		byMember[member] = file
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if json.Unmarshal(body, &index) != nil || len(index.WeightMap) == 0 || len(index.WeightMap) > 200_000 {
			return Plan{}, exit.Named(exit.Validation, "model_source_index_invalid",
				"Hugging Face tensor index %s is malformed or unbounded", member)
		}
		for _, shard := range index.WeightMap {
			resolvedShard := path.Clean(path.Join(path.Dir(member), shard))
			if !safeMember(resolvedShard) || byMember[resolvedShard].Member == "" {
				return Plan{}, exit.Named(exit.Validation, "model_source_index_invalid",
					"tensor index %s names absent or unsafe shard %q", member, shard)
			}
			referenced[resolvedShard] = true
			file.Requires = append(file.Requires, resolvedShard)
		}
		sort.Strings(file.Requires)
		file.Requires = compact(file.Requires)
		byMember[member] = file
	}
	files := make([]File, 0, len(byMember))
	for member, file := range byMember {
		if strings.HasSuffix(strings.ToLower(member), ".safetensors") && !referenced[member] {
			file.Carrier = true
		}
		if !validDigest(file.SHA256) || file.Length <= 0 {
			return Plan{}, exit.Named(exit.Validation, "model_source_identity_missing",
				"provider did not supply an exact size and SHA-256 for %s", member)
		}
		files = append(files, file)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Member < files[j].Member })
	resolved := source
	resolved.Revision, resolved.Reference = commit, commit
	resolved.Canonical = "hf://" + source.Org + "/" + source.Repo + "@" + commit
	return finishPlan(resolved, metadata.CardData.License, files)
}

// Select narrows a metadata plan to the exact carrier members authorized by
// TensorFS source-plan plus their declared sharded dependencies. No body moves
// until this closure is known.
func (p Plan) Select(members []string) (Plan, *exit.Error) {
	byMember := make(map[string]File, len(p.Files))
	for _, file := range p.Files {
		byMember[file.Member] = file
	}
	wanted := make(map[string]bool)
	for _, member := range members {
		file, ok := byMember[member]
		if !ok || !file.Carrier {
			return Plan{}, exit.Named(exit.Validation, "model_source_plan_mismatch",
				"TensorFS selected unknown carrier member %q", member)
		}
		wanted[member] = true
		for _, dependency := range file.Requires {
			wanted[dependency] = true
		}
	}
	files := make([]File, 0, len(wanted))
	for member := range wanted {
		file, ok := byMember[member]
		if !ok {
			return Plan{}, exit.Internalf("resolved source plan lost dependency %s", member)
		}
		files = append(files, file)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Member < files[j].Member })
	return finishPlan(p.Source, p.License, files)
}

type civitaiVersion struct {
	ID      uint64 `json:"id"`
	ModelID uint64 `json:"modelId"`
	Files   []struct {
		ID          uint64  `json:"id"`
		Name        string  `json:"name"`
		SizeKB      float64 `json:"sizeKB"`
		DownloadURL string  `json:"downloadUrl"`
		Hashes      struct {
			SHA256 string `json:"SHA256"`
		} `json:"hashes"`
		Metadata struct {
			Format string `json:"format"`
		} `json:"metadata"`
	} `json:"files"`
}

type civitaiModel struct {
	AllowNoCredit         bool     `json:"allowNoCredit"`
	AllowCommercialUse    []string `json:"allowCommercialUse"`
	AllowDerivatives      bool     `json:"allowDerivatives"`
	AllowDifferentLicense bool     `json:"allowDifferentLicense"`
}

func (r *Resolver) resolveCivitai(ctx context.Context, source Source) (Plan, *exit.Error) {
	var version civitaiVersion
	api := "https://civitai.com/api/v1/model-versions/" + strconv.FormatUint(source.VersionID, 10)
	if problem := r.json(ctx, api, &version); problem != nil {
		return Plan{}, problem
	}
	if version.ID != source.VersionID || version.ModelID == 0 || len(version.Files) == 0 || len(version.Files) > maxFiles {
		return Plan{}, exit.Named(exit.Validation, "model_source_metadata_incomplete",
			"Civitai response does not describe the requested model version")
	}
	var model civitaiModel
	if problem := r.json(ctx, "https://civitai.com/api/v1/models/"+strconv.FormatUint(version.ModelID, 10), &model); problem != nil {
		return Plan{}, problem
	}
	files := make([]File, 0)
	for _, remote := range version.Files {
		if remote.ID == 0 || !strings.HasSuffix(strings.ToLower(remote.Name), ".safetensors") ||
			(remote.Metadata.Format != "" && !strings.EqualFold(remote.Metadata.Format, "SafeTensor")) {
			continue
		}
		sha := strings.ToLower(remote.Hashes.SHA256)
		if !validDigest(sha) {
			return Plan{}, exit.Named(exit.Validation, "model_source_identity_missing",
				"Civitai file %d lacks exact SHA-256/size facts", remote.ID)
		}
		if remote.DownloadURL == "" {
			remote.DownloadURL = "https://civitai.com/api/download/models/" + strconv.FormatUint(version.ID, 10)
		}
		length, problem := r.measure(ctx, remote.DownloadURL)
		if problem != nil {
			return Plan{}, problem
		}
		files = append(files, File{Member: "civitai/files/" + strconv.FormatUint(remote.ID, 10),
			URL: remote.DownloadURL, SHA256: sha, Length: length, Carrier: true})
	}
	if len(files) == 0 {
		return Plan{}, exit.Named(exit.Validation, "model_source_files_absent",
			"Civitai version contains no supported tensor carrier")
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Member < files[j].Member })
	license := fmt.Sprintf("civitai:no-credit=%t;commercial=%s;derivatives=%t;different-license=%t",
		model.AllowNoCredit, strings.Join(model.AllowCommercialUse, ","), model.AllowDerivatives, model.AllowDifferentLicense)
	return finishPlan(source, license, files)
}

func (r *Resolver) measure(ctx context.Context, location string) (int64, *exit.Error) {
	response, problem := r.request(ctx, http.MethodHead, location, nil)
	if problem != nil {
		return 0, problem
	}
	response.Body.Close()
	if response.StatusCode == http.StatusOK && response.ContentLength > 0 {
		return response.ContentLength, nil
	}
	headers := make(http.Header)
	headers.Set("Range", "bytes=0-0")
	response, problem = r.request(ctx, http.MethodGet, location, headers)
	if problem != nil {
		return 0, problem
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusPartialContent {
		return 0, exit.Named(exit.Validation, "model_source_length_unknown",
			"provider does not expose an exact file length")
	}
	raw := response.Header.Get("Content-Range")
	_, total, ok := strings.Cut(raw, "/")
	length, err := strconv.ParseInt(total, 10, 64)
	if !ok || err != nil || length <= 0 {
		return 0, exit.Named(exit.Validation, "model_source_length_unknown",
			"provider returned invalid Content-Range %q", raw)
	}
	return length, nil
}

func (r *Resolver) small(ctx context.Context, location string, limit int64) ([]byte, *exit.Error) {
	response, problem := r.request(ctx, http.MethodGet, location, nil)
	if problem != nil {
		return nil, problem
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, exit.Named(exit.Validation, "model_source_member_refused",
			"provider member request answered HTTP %d", response.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil || int64(len(body)) > limit {
		return nil, exit.Named(exit.Validation, "model_source_member_too_large",
			"provider member exceeded its %d-byte metadata bound", limit)
	}
	return body, nil
}

func finishPlan(source Source, license string, files []File) (Plan, *exit.Error) {
	if len(files) == 0 || len(files) > maxFiles {
		return Plan{}, exit.Named(exit.Validation, "model_source_files_invalid",
			"model source selected an invalid number of files")
	}
	var total int64
	hash := sha256.New()
	_, _ = io.WriteString(hash, source.Canonical+"\x00")
	for _, file := range files {
		if !safeMember(file.Member) || !safeProviderURL(source.Kind, file.URL) ||
			!validDigest(file.SHA256) || file.Length <= 0 ||
			file.Length > maxSourceSize-total {
			return Plan{}, exit.Named(exit.Validation, "model_source_identity_invalid",
				"model source contains an unsafe or unbounded file fact")
		}
		total += file.Length
		_, _ = io.WriteString(hash, file.Member+"\x00"+file.SHA256+"\x00"+strconv.FormatInt(file.Length, 10)+"\x00")
	}
	return Plan{Source: source, Canonical: source.Canonical,
		SelectionSHA256: hex.EncodeToString(hash.Sum(nil)), License: strings.TrimSpace(license),
		Files: files, Bytes: total}, nil
}

func safeMember(member string) bool {
	if member == "" || len(member) > 1024 || strings.HasPrefix(member, ".") ||
		path.IsAbs(member) || strings.Contains(member, "\\") {
		return false
	}
	for _, part := range strings.Split(member, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

func safeProviderURL(kind Kind, raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.User != nil || u.Fragment != "" ||
		!allowedHost(kind, u.Hostname()) {
		return false
	}
	for key := range u.Query() {
		key = strings.ToLower(key)
		if strings.Contains(key, "token") || strings.Contains(key, "key") || strings.Contains(key, "auth") {
			return false
		}
	}
	return true
}

func escapeMember(member string) string {
	parts := strings.Split(member, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func digest(value []byte) string {
	sum := sha256.Sum256(value)
	return hex.EncodeToString(sum[:])
}

func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func compact(values []string) []string {
	if len(values) < 2 {
		return values
	}
	out := values[:1]
	for _, value := range values[1:] {
		if value != out[len(out)-1] {
			out = append(out, value)
		}
	}
	return out
}
