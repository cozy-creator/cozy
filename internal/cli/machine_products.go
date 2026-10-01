package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/runoutputs"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// A run's outputs are items rewritten in place (progressive-outputs.md). The machine journals
// each revision as a `product` entry; the shared fold names its item, rev and how it changed;
// this follower writes it into the item's one stable file, `<n>-<output>[-<i>].<ext>` in the
// run's outputs folder: an append by writing only the new tail in place, a replace whole under
// a temporary name then renamed over. The file is the output from its first revision to its
// last. Nothing here reaches the Hub.

type machineByteStream interface {
	Recv() (*pb.NativeByteReadChunk, error)
}

// holdProducts writes every revision a page names after `cursor` into its item's file, and
// answers the Product each becomes.
func (m *machineRuns) holdProducts(ctx context.Context, request records.Request, connection *machineConnection, page *pb.MachineExecutionEventPage, cursor uint64) (map[uint64]records.Product, *exit.Error) {
	held := map[uint64]records.Product{}
	var fold *runoutputs.Fold
	var directory string
	for _, event := range page.GetEvents() {
		if event.GetSequence() <= cursor || event.GetKind() != "product" {
			continue
		}
		if fold == nil {
			var problem *exit.Error
			if fold, directory, request, problem = m.outputFold(request); problem != nil {
				return nil, problem
			}
		}
		// An entry this client cannot hold is skipped (the page records a note); its words are
		// made printable, never a reason to refuse the page.
		source := sanitizeProduct(event.GetProduct())
		if source == nil {
			continue
		}
		item, _, err := fold.Add(event.Sequence, source)
		if err != nil {
			continue
		}
		event = &pb.MachineExecutionEvent{Sequence: event.Sequence, Product: source}
		product := productOf(item, event.Product)
		if directory != "" {
			// The file is where people look while the run goes on. A folder that refuses it never
			// stops the log: the run's end brings every item's file to its final revision, and a
			// refusal there waits on the owner as a pending collection, never a silent hang.
			product.Path = filepath.Join(directory, itemFile(strconv.FormatInt(request.Number, 10), item, product.MediaType))
			_ = writeItem(ctx, connection, directory, product.Path, item, product)
		}
		held[event.Sequence] = product
	}
	return held, nil
}

// outputFold is the run's items as its recorded revisions left them, and the folder they are
// written in ("" for a run whose outputs are not written, a child's say).
func (m *machineRuns) outputFold(request records.Request) (*runoutputs.Fold, string, records.Request, *exit.Error) {
	if request.Number == 0 {
		// Observers carry the durable row; its short number names the run's files.
		if numbered, problem := m.store.RequestByReference(request.ID); problem == nil && numbered != nil {
			request.Number = numbered.Number
		}
	}
	products, problem := m.store.Products(request.ID)
	if problem != nil {
		return nil, "", request, problem
	}
	fold := runoutputs.New(strconv.FormatInt(request.Number, 10))
	for _, product := range products {
		if _, _, err := fold.Add(product.Sequence, recordedProduct(product)); err != nil {
			return nil, "", request, exit.Internalf("a recorded output revision is unreadable: %s", err)
		}
	}
	export, problem := m.store.OutputExportOf(request.ID)
	if problem != nil || export == nil {
		return fold, "", request, problem
	}
	return fold, export.Directory, request, nil
}

// sanitizeProduct is an entry as this client holds it: its output, label and media type
// made printable, or nil for one whose bytes cannot be read (no single source or parts).
func sanitizeProduct(source *pb.RunProduct) *pb.RunProduct {
	if source == nil || source.GetContent().GetLength() > math.MaxInt64 ||
		(source.Source == nil) == (len(source.Parts) == 0) || len(source.Parts) > 128 ||
		source.Source != nil && records.ValidateByteRef(source.Source.GetSource()) != nil {
		return nil
	}
	for _, part := range source.Parts {
		if records.ValidateByteRef(part.GetSource().GetSource()) != nil {
			return nil
		}
	}
	clean := proto.Clone(source).(*pb.RunProduct)
	clean.Output = printable(clean.Output, 1024, '_')
	clean.Label = printable(clean.Label, 256, '?')
	if clean.MediaType = printable(clean.MediaType, 256, '_'); clean.MediaType != source.MediaType || clean.MediaType == "" {
		clean.MediaType = "application/octet-stream"
	}
	if clean.Output == "" {
		return nil
	}
	return clean
}

// printable is a peer's word as this client keeps it: printable ASCII, the rest `substitute`.
func printable(value string, bound int, substitute rune) string {
	runes := []rune(value)
	if len(runes) > bound {
		runes = runes[:bound]
	}
	for i, r := range runes {
		if r < 0x20 || r > 0x7e {
			runes[i] = substitute
		}
	}
	return string(runes)
}

// productOf is the revision the fold made of one entry, as this client records it.
func productOf(item runoutputs.Item, source *pb.RunProduct) records.Product {
	current := item.Current
	product := records.Product{Sequence: current.Sequence, Item: item.ID, OutputIndex: item.OutputIndex, Type: item.Type, Output: item.Output, Op: records.ProductSet,
		Index: source.Index, Rev: current.Rev, Digest: current.Digest, Length: current.Length, AppendedFrom: current.AppendedFrom,
		DurationUs: current.DurationUs, MediaType: current.MediaType, Label: current.Label}
	if item.List {
		product.Op = records.ProductAppend
	}
	if source.Source != nil && current.MediaType == resultfiles.TreeMediaType {
		product.ContentBytes = int64(source.Source.Source.ContentBytes)
	}
	if len(source.Parts) > 0 {
		for _, part := range current.Parts {
			product.Parts = append(product.Parts, records.ProductPart{Digest: part.Digest, Length: part.Length, DurationUs: part.DurationUs})
		}
	}
	return product
}

// recordedProduct is a recorded revision as the fold reads it again; its bytes are here.
func recordedProduct(product records.Product) *pb.RunProduct {
	ref := func(digest string, length int64) *pb.Ref {
		raw, _ := canonical.Raw(digest)
		return &pb.Ref{Digest: raw, Length: uint64(length)}
	}
	op := pb.RunProductOp_RUN_PRODUCT_OP_SET
	if product.Op == records.ProductAppend {
		op = pb.RunProductOp_RUN_PRODUCT_OP_APPEND
	}
	source := &pb.RunProduct{Output: product.Output, Op: op, Index: product.Index, Content: ref(product.Digest, product.Length),
		MediaType: product.MediaType, Label: product.Label}
	for _, part := range product.Parts {
		source.Parts = append(source.Parts, &pb.RunProductPart{Content: ref(part.Digest, part.Length), DurationUs: part.DurationUs})
	}
	return source
}

// itemFile is the item's stable file name in its run's outputs folder.
func itemFile(run string, item runoutputs.Item, mediaType string) string {
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-' {
			return r
		}
		return '_'
	}, item.Name(run))
	if mediaType == resultfiles.TreeMediaType {
		return name
	}
	return name + resultfiles.Extension(mediaType)
}

// writeItem brings the item's file at `path` to its current revision: when the file holds an
// earlier revision this one extends, only the missing tail is appended in place; otherwise the
// revision is written whole and renamed over it.
func writeItem(ctx context.Context, connection *machineConnection, directory, path string, item runoutputs.Item, product records.Product) *exit.Error {
	if problem := resultfiles.Preflight(directory); problem != nil {
		return problem
	}
	current := item.Current
	if product.MediaType == resultfiles.TreeMediaType {
		return writeTree(ctx, connection, directory, path, current, product.ContentBytes)
	}
	if info, err := os.Lstat(path); err == nil && info.Mode().IsRegular() {
		for _, earlier := range item.History {
			if earlier.Rev < current.Rev && earlier.Length == info.Size() && earlier.PrefixOf(current) {
				return appendTail(ctx, connection, path, info.Size(), current)
			}
		}
	}
	return replaceWhole(ctx, connection, directory, path, current)
}

// appendTail writes bytes [have, length) of the revision onto the file in place: an open
// reader keeps reading, and sees the new bytes once written.
func appendTail(ctx context.Context, connection *machineConnection, path string, have int64, current runoutputs.Revision) *exit.Error {
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return exit.Internalf("cannot open %s: %s", path, err)
	}
	defer file.Close()
	if _, err := file.Seek(have, io.SeekStart); err != nil {
		return exit.Internalf("cannot append to %s: %s", path, err)
	}
	if problem := receiveParts(ctx, connection, file, nil, current.Parts, have); problem != nil {
		return problem
	}
	if err := file.Sync(); err != nil {
		return exit.Internalf("cannot sync %s: %s", path, err)
	}
	return nil
}

// replaceWhole writes the revision whole beside the file, verifies it, and renames it over.
func replaceWhole(ctx context.Context, connection *machineConnection, directory, path string, current runoutputs.Revision) *exit.Error {
	staged, err := os.CreateTemp(directory, ".cozy-output-")
	if err != nil {
		return exit.Internalf("cannot stage an output in %s: %s", directory, err)
	}
	defer func() { staged.Close(); _ = os.Remove(staged.Name()) }()
	if err := staged.Chmod(0o644); err != nil {
		return exit.Internalf("cannot stage an output: %s", err)
	}
	hash := sha256.New()
	if problem := receiveParts(ctx, connection, staged, hash, current.Parts, 0); problem != nil {
		return problem
	}
	if "sha256:"+hex.EncodeToString(hash.Sum(nil)) != current.Digest {
		return exit.New(exit.Conflict, "a received output differs from its digest")
	}
	if err := staged.Sync(); err != nil {
		return exit.Internalf("cannot sync an output: %s", err)
	}
	if err := os.Rename(staged.Name(), path); err != nil {
		return exit.Internalf("cannot write %s: %s", path, err)
	}
	return syncDirectory(directory)
}

// receiveParts writes the parts' bytes from offset `from` of their concatenation onward.
func receiveParts(ctx context.Context, connection *machineConnection, output io.Writer, hash io.Writer, parts []runoutputs.Part, from int64) *exit.Error {
	if hash != nil {
		output = io.MultiWriter(output, hash)
	}
	var start int64
	for _, part := range parts {
		end := start + part.Length
		if end <= from {
			start = end
			continue
		}
		offset := max(from-start, 0)
		raw, _ := canonical.Raw(part.Digest)
		stream, err := connection.readBytes(ctx, part.Source, &pb.Ref{Digest: raw, Length: uint64(part.Length)}, uint64(offset))
		if err != nil {
			return machineTransport(err)
		}
		for offset < part.Length {
			chunk, err := stream.Recv()
			if err != nil {
				return machineTransport(err)
			}
			if chunk == nil || chunk.Offset != uint64(offset) || len(chunk.Data) == 0 || len(chunk.Data) > pb.MaxNativeByteReadChunkBytes || int64(len(chunk.Data)) > part.Length-offset {
				return exit.New(exit.Conflict, "an output's byte stream left its exact bounds")
			}
			if _, err := output.Write(chunk.Data); err != nil {
				return exit.Internalf("cannot write an output: %s", err)
			}
			connection.progress.Advance(int64(len(chunk.Data)))
			offset += int64(len(chunk.Data))
		}
		start = end
	}
	return nil
}

// writeTree receives a tree item's manifest and members, then puts the tree at its stable
// name, replacing another revision whole.
func writeTree(ctx context.Context, connection *machineConnection, directory, path string, current runoutputs.Revision, contentBytes int64) *exit.Error {
	staging, err := os.MkdirTemp(directory, ".cozy-tree-receiving-")
	if err != nil {
		return exit.Internalf("cannot stage a tree in %s: %s", directory, err)
	}
	defer os.RemoveAll(staging)
	manifest := filepath.Join(staging, "manifest")
	if problem := replaceWhole(ctx, connection, staging, manifest, current); problem != nil {
		return problem
	}
	members, problem := resultfiles.ReadTreeManifest(manifest, current.Digest, current.Length, contentBytes)
	if problem != nil {
		return problem
	}
	if err := os.MkdirAll(manifest+".files", 0o700); err != nil {
		return exit.Internalf("cannot stage a tree: %s", err)
	}
	source := current.Parts[0].Source
	for _, member := range members {
		part := runoutputs.Revision{Digest: member.Digest, Length: member.Length,
			Parts: []runoutputs.Part{{Digest: member.Digest, Length: member.Length, Source: source}}}
		if problem := replaceWhole(ctx, connection, manifest+".files", filepath.Join(manifest+".files", strings.TrimPrefix(member.Digest, "sha256:")), part); problem != nil {
			return problem
		}
	}
	_, problem = resultfiles.MaterializeTree(manifest, directory, filepath.Base(path), current.Digest, current.Length, contentBytes)
	return problem
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

// exportProducts settles the run's result, whatever its terminal: each item's file at its
// final revision, checked against that revision's sha256. A file a refused write left behind is
// written now from the machine's log, which holds every item's current bytes until the ack.
func (m *machineRuns) exportProducts(ctx context.Context, connection *machineConnection, query *pb.MachineExecutionQuery, request records.Request) *exit.Error {
	export, problem := m.store.OutputExportOf(request.ID)
	if problem != nil || export == nil || export.State == "published" {
		return problem
	}
	products, problem := m.store.Products(request.ID)
	if problem != nil {
		return problem
	}
	var live *runoutputs.Fold
	var paths []string
	for _, product := range records.Fold(products) {
		if product.Path == "" {
			continue
		}
		if verifyFinal(product) != nil {
			if live == nil {
				if live, problem = m.logFold(ctx, connection, query, request); problem != nil {
					return problem
				}
			}
			index := uint32(0)
			if product.Op == records.ProductAppend {
				index = product.Index + 1
			}
			item, ok := live.Item(product.Output, index)
			if !ok {
				return exit.New(exit.Conflict, "the machine's log no longer names output %s", product.Item)
			}
			problem = writeItem(ctx, connection, export.Directory, product.Path, item, product)
			if problem == nil {
				problem = verifyFinal(product)
			}
			if problem != nil {
				named := *problem
				named.Message = "output " + product.Output + ": " + problem.Message
				_ = m.store.FailOutputExport(request.ID, named.ErrName(), named.Message)
				return &named
			}
		}
		paths = append(paths, product.Path)
	}
	return m.store.CompleteOutputExport(request.ID, paths)
}

// logFold reads the run's product entries from its machine again, with the byte sources the
// recorded revisions do not keep.
func (m *machineRuns) logFold(ctx context.Context, connection *machineConnection, query *pb.MachineExecutionQuery, request records.Request) (*runoutputs.Fold, *exit.Error) {
	fold := runoutputs.New(strconv.FormatInt(request.Number, 10))
	for after := uint64(0); ; {
		page, err := connection.Host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: after, Limit: 128})
		if err != nil {
			return nil, machineTransport(err)
		}
		for _, event := range page.GetEvents() {
			if event.GetKind() == "product" && event.GetProduct() != nil {
				if _, _, err := fold.Add(event.Sequence, event.Product); err != nil {
					return nil, exit.New(exit.Conflict, "the machine journaled an unusable output revision: %s", err)
				}
			}
		}
		if page.NextAfter >= page.HeadSequence || page.NextAfter <= after {
			return fold, nil
		}
		after = page.NextAfter
	}
}

// verifyFinal checks an item's finished file against its final revision's sha256. A tree is
// checked when it is written.
func verifyFinal(product records.Product) *exit.Error {
	if product.MediaType == resultfiles.TreeMediaType {
		if info, err := os.Lstat(product.Path); err != nil || !info.IsDir() {
			return exit.Named(exit.Conflict, "output_final_changed", "%s is not its tree", product.Path)
		}
		return nil
	}
	file, err := os.Open(product.Path)
	if err != nil {
		return exit.Named(exit.Conflict, "output_final_changed", "%s is gone: %s", product.Path, err)
	}
	defer file.Close()
	hash := sha256.New()
	written, err := io.Copy(hash, io.LimitReader(file, product.Length+1))
	if err != nil || written != product.Length || "sha256:"+hex.EncodeToString(hash.Sum(nil)) != product.Digest {
		return exit.Named(exit.Conflict, "output_final_changed", "%s differs from its final revision", product.Path)
	}
	return nil
}
