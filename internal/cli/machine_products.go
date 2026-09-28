package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A run reports its work as it happens. The machine journals each product the function
// publishes (worker-protocol RunProduct); this client, the owner's one durable mirror, holds
// the product's bytes before the entry becomes a request.product event. A list product lands
// in the run's outputs folder when it arrives, a single output's current revision as
// `<n>-<output>.partial.<ext>`, and the terminal writes the fold: the run's result, whether it
// completed, failed or was canceled. Nothing here reaches the Hub.

type machineByteStream interface {
	Recv() (*pb.NativeByteReadChunk, error)
}

// productBlob is one byte string a product needs: the whole product, or one of its parts.
type productBlob struct {
	digest string
	length int64
	source *pb.NativeByteRetentionRequest
}

// holdProducts receives the bytes of every product a page names after `cursor`, and the
// Product each becomes.
func (m *machineRuns) holdProducts(ctx context.Context, request records.Request, connection *machineConnection, page *pb.MachineExecutionEventPage, cursor uint64) (map[uint64]records.Product, *exit.Error) {
	held := map[uint64]records.Product{}
	if request.Number == 0 {
		// Observers carry the durable row; its short number names the run's partial files.
		if numbered, problem := m.store.RequestByReference(request.ID); problem == nil && numbered != nil {
			request.Number = numbered.Number
		}
	}
	for _, event := range page.GetEvents() {
		if event.GetSequence() <= cursor || event.GetKind() != "product" {
			continue
		}
		product, blobs, problem := productOf(event)
		if problem != nil {
			return nil, problem
		}
		store := m.productStore(request)
		for _, blob := range blobs {
			if problem := receiveProductBlob(ctx, connection, store, blob); problem != nil {
				return nil, problem
			}
		}
		if product.MediaType == resultfiles.TreeMediaType {
			if problem := receiveProductTree(ctx, connection, store, blobs[0], product.ContentBytes); problem != nil {
				return nil, problem
			}
		}
		// The bytes are held here now; the outputs folder is only where people look. A folder
		// that refuses the file never stops the log: the run's end writes its result, and a
		// refusal there waits on the owner as a pending collection (never a silent hang).
		product.Path, _ = m.showProduct(request, product)
		held[event.Sequence] = product
	}
	return held, nil
}

// productOf decodes one RunProduct once, at this boundary.
func productOf(event *pb.MachineExecutionEvent) (records.Product, []productBlob, *exit.Error) {
	source := event.GetProduct()
	refuse := func(why string) (records.Product, []productBlob, *exit.Error) {
		return records.Product{}, nil, exit.New(exit.Conflict, "the machine journaled an unusable product: %s", why)
	}
	if source == nil {
		return refuse("the entry carries none")
	}
	op := map[pb.RunProductOp]string{pb.RunProductOp_RUN_PRODUCT_OP_SET: records.ProductSet, pb.RunProductOp_RUN_PRODUCT_OP_APPEND: records.ProductAppend}[source.Op]
	digest, err := canonical.Spell(source.GetContent().GetDigest())
	if op == "" || err != nil || source.Content.Length > math.MaxInt64 || !printable(source.Output, 1024) || source.Output == "" ||
		!printable(source.Label, 256) || !printable(source.MediaType, 256) {
		return refuse("its output, content or label is malformed")
	}
	product := records.Product{Sequence: event.Sequence, Output: source.Output, Op: op, Index: source.Index, Digest: digest,
		Length: int64(source.Content.Length), MediaType: source.MediaType, Label: source.Label}
	if (source.Source == nil) == (len(source.Parts) == 0) || len(source.Parts) > 128 {
		return refuse("it needs exactly one byte source or its parts")
	}
	if source.Source != nil {
		if records.ValidateByteRef(source.Source.GetSource()) != nil {
			return refuse("its byte source is malformed")
		}
		if product.MediaType == resultfiles.TreeMediaType {
			product.ContentBytes = int64(source.Source.Source.ContentBytes)
		}
		return product, []productBlob{{digest: digest, length: product.Length, source: source.Source}}, nil
	}
	var blobs []productBlob
	var total uint64
	for _, part := range source.Parts {
		spelled, err := canonical.Spell(part.GetContent().GetDigest())
		if err != nil || records.ValidateByteRef(part.GetSource().GetSource()) != nil || part.Content.Length > math.MaxInt64 {
			return refuse("a part is malformed")
		}
		total += part.Content.Length
		blobs = append(blobs, productBlob{digest: spelled, length: int64(part.Content.Length), source: part.Source})
		product.Parts = append(product.Parts, records.ProductPart{Digest: spelled, Length: int64(part.Content.Length), DurationUs: part.DurationUs})
	}
	if total != source.Content.Length {
		return refuse("its parts do not add up to its length")
	}
	return product, blobs, nil
}

func printable(value string, bound int) bool {
	return len(value) <= bound && strings.IndexFunc(value, func(r rune) bool { return r < 0x20 || r > 0x7e }) < 0
}

// productStore is the run's own content-addressed copy of its products.
func (m *machineRuns) productStore(request records.Request) string {
	return m.layout.Products(request.Org, request.ID)
}

func productBlobPath(store, digest string) string {
	return filepath.Join(store, strings.TrimPrefix(digest, "sha256:"))
}

// productSources are the files whose concatenation is the product's bytes.
func productSources(store string, product records.Product) []string {
	if len(product.Parts) == 0 {
		return []string{productBlobPath(store, product.Digest)}
	}
	sources := make([]string, len(product.Parts))
	for index, part := range product.Parts {
		sources[index] = productBlobPath(store, part.Digest)
	}
	return sources
}

// receiveProductBlob holds one byte string here: a verified file named by its digest. An
// interrupted receive resumes at the offset its staged file reached.
func receiveProductBlob(ctx context.Context, connection *machineConnection, store string, blob productBlob) *exit.Error {
	final := productBlobPath(store, blob.digest)
	if info, err := os.Lstat(final); err == nil && info.Mode().IsRegular() && info.Size() == blob.length {
		return nil
	}
	if err := os.MkdirAll(store, 0o700); err != nil {
		return exit.Internalf("cannot create the run's product store: %s", err)
	}
	staged := final + ".receiving"
	output, err := os.OpenFile(staged, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return exit.Internalf("cannot stage a product: %s", err)
	}
	defer output.Close()
	hash := sha256.New()
	offset, err := io.Copy(hash, output)
	if err != nil || offset > blob.length {
		offset = 0
		hash.Reset()
		if err := output.Truncate(0); err != nil {
			return exit.Internalf("cannot restage a product: %s", err)
		}
		if _, err := output.Seek(0, io.SeekStart); err != nil {
			return exit.Internalf("cannot restage a product: %s", err)
		}
	}
	raw, _ := canonical.Raw(blob.digest)
	stream, err := connection.readBytes(ctx, blob.source, &pb.Ref{Digest: raw, Length: uint64(blob.length)}, uint64(offset))
	if err != nil {
		return machineTransport(err)
	}
	for offset < blob.length {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return machineTransport(err)
		}
		if chunk == nil || chunk.Offset != uint64(offset) || len(chunk.Data) == 0 || len(chunk.Data) > pb.MaxNativeByteReadChunkBytes || int64(len(chunk.Data)) > blob.length-offset {
			return exit.New(exit.Conflict, "a product's byte stream left its exact bounds")
		}
		if _, err := io.MultiWriter(output, hash).Write(chunk.Data); err != nil {
			return exit.Internalf("cannot write a product: %s", err)
		}
		connection.progress.Advance(int64(len(chunk.Data)))
		offset += int64(len(chunk.Data))
	}
	if offset != blob.length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != blob.digest {
		_ = os.Remove(staged)
		return exit.New(exit.Conflict, "a received product differs from its digest")
	}
	if err := output.Sync(); err != nil {
		return exit.Internalf("cannot sync a product: %s", err)
	}
	if err := os.Rename(staged, final); err != nil {
		return exit.Internalf("cannot commit a product: %s", err)
	}
	return syncDirectory(store)
}

// receiveProductTree holds a tree product's members beside its manifest, where
// resultfiles.MaterializeTree reads them.
func receiveProductTree(ctx context.Context, connection *machineConnection, store string, manifest productBlob, contentBytes int64) *exit.Error {
	path := productBlobPath(store, manifest.digest)
	members, problem := resultfiles.ReadTreeManifest(path, manifest.digest, manifest.length, contentBytes)
	if problem != nil {
		return problem
	}
	for _, member := range members {
		if problem := receiveProductBlob(ctx, connection, path+".files", productBlob{digest: member.Digest, length: member.Length, source: manifest.source}); problem != nil {
			return problem
		}
	}
	return nil
}

func syncDirectory(path string) *exit.Error {
	directory, err := os.Open(path)
	if err != nil {
		return exit.Internalf("cannot open %s: %s", path, err)
	}
	defer directory.Close()
	if err := directory.Sync(); err != nil {
		return exit.Internalf("cannot sync %s: %s", path, err)
	}
	return nil
}

// showProduct writes a product where people look while the run goes on: a list product's
// own file, or its single output's `.partial` file, replaced atomically for each revision.
func (m *machineRuns) showProduct(request records.Request, product records.Product) (string, *exit.Error) {
	export, problem := m.store.OutputExportOf(request.ID)
	if problem != nil || export == nil || product.MediaType == resultfiles.TreeMediaType {
		return "", problem
	}
	sources := productSources(m.productStore(request), product)
	if product.Op == records.ProductAppend {
		name, problem := resultfiles.Filename(product.Digest, product.MediaType)
		if problem != nil {
			return "", problem
		}
		return resultfiles.MaterializeParts(sources, export.Directory, name, product.Digest, product.Length)
	}
	if !export.Partials {
		return "", nil
	}
	return resultfiles.MaterializeParts(sources, export.Directory, partialName(request, product), product.Digest, product.Length)
}

// partialName is a single output's revision file while its run goes on.
func partialName(request records.Request, product records.Product) string {
	output := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-' {
			return r
		}
		return '_'
	}, product.Output)
	return fmt.Sprintf("%d-%s.partial%s", request.Number, output, resultfiles.Extension(product.MediaType))
}

// exportProducts writes the run's result, the fold of its products, into its outputs folder
// under each file's digest name, removes the partial files, and settles the export.
func (m *machineRuns) exportProducts(request records.Request) *exit.Error {
	export, problem := m.store.OutputExportOf(request.ID)
	if problem != nil || export == nil || export.State == "published" {
		return problem
	}
	products, problem := m.store.Products(request.ID)
	if problem != nil {
		return problem
	}
	store := m.productStore(request)
	var paths []string
	for _, product := range records.Fold(products) {
		var path string
		if product.MediaType == resultfiles.TreeMediaType {
			path, problem = resultfiles.MaterializeTree(productBlobPath(store, product.Digest), export.Directory, product.Digest, product.Length, product.ContentBytes)
		} else {
			name, filenameProblem := resultfiles.Filename(product.Digest, product.MediaType)
			if problem = filenameProblem; problem == nil {
				path, problem = resultfiles.MaterializeParts(productSources(store, product), export.Directory, name, product.Digest, product.Length)
			}
		}
		if problem != nil {
			_ = m.store.FailOutputExport(request.ID, problem.ErrName(), problem.Message)
			return problem
		}
		paths = append(paths, path)
	}
	for _, product := range products {
		if product.Op == records.ProductSet && product.Path != "" && strings.HasSuffix(strings.TrimSuffix(product.Path, filepath.Ext(product.Path)), ".partial") {
			_ = os.Remove(product.Path)
		}
	}
	return m.store.CompleteOutputExport(request.ID, paths)
}
