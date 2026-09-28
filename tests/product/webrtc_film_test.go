package producttest

import (
	"bytes"
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// fmp4Fragments cuts a fragmented MP4 at its top-level boxes: the init (ftyp, moov), then one
// piece per moof+mdat.
func fmp4Fragments(film []byte) (header []byte, pieces [][]byte) {
	for at := 0; at < len(film); {
		size, kind := int(binary.BigEndian.Uint32(film[at:])), string(film[at+4:at+8])
		switch kind {
		case "moof":
			pieces = append(pieces, nil)
		case "ftyp", "moov":
			header = append(header, film[at:at+size]...)
			at += size
			continue
		}
		pieces[len(pieces)-1] = append(pieces[len(pieces)-1], film[at:at+size]...)
		at += size
	}
	return header, pieces
}

func countFrames(t *testing.T, media []byte) string {
	path := filepath.Join(t.TempDir(), "piece.mp4")
	if err := os.WriteFile(path, media, 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
		"-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path).CombinedOutput()
	if err != nil {
		t.Fatalf("ffprobe: %v %s", err, out)
	}
	return strings.TrimSpace(string(out))
}

// A 3-segment film published by appends plays as it lands, and a seek fetches the init and
// one middle segment by the byte ranges its entries map.
func TestWebRTCAFilmStreamsLiveAndItsMiddleSegmentDecodes(t *testing.T) {
	h := newMediaHarness(t)
	path := filepath.Join(t.TempDir(), "film.mp4")
	if out, err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=24",
		"-c:v", "libx264", "-preset", "ultrafast", "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
		"-movflags", "frag_keyframe+empty_moov+default_base_moof", path).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg: %v %s", err, out)
	}
	film, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	header, pieces := fmp4Fragments(film)
	if len(pieces) != 3 {
		t.Fatalf("%d fragments", len(pieces))
	}

	c := h.open(64 << 20)
	h.follow(c, 0, 0)
	var f mediaFollower
	var published []byte
	for k, piece := range pieces {
		if k == 0 {
			piece = append(append([]byte(nil), header...), piece...)
		}
		published = append(published, piece...)
		e := h.m.Append(7, "video", -1, piece, 1_000_000)
		h.until(c, &f, func() bool { return len(f.got) == len(published) && f.seq == e.Seq })
		if got := countFrames(t, f.got); got != []string{"24", "48", "72"}[k] {
			t.Fatalf("after mediaSegment %d the follower's copy decodes %s frames", k+1, got)
		}
	}
	h.m.End(7, "completed")
	h.until(c, &f, func() bool { return f.end != nil })
	if !bytes.Equal(f.got, film) || f.end.SHA256 != digestOf(film) {
		t.Fatal("the followed film is not the film")
	}

	seek := h.open(64 << 20)
	h.send(seek, map[string]any{"t": "get", "id": "init", "run": "7", "output": "video", "length": len(header)})
	head, _ := h.body(seek)
	from, to := f.entries[0].Length, f.entries[1].Length // segment 2, by the map
	h.send(seek, map[string]any{"t": "get", "id": "s2", "run": "7", "output": "video", "offset": from, "length": to - from})
	middle, _ := h.body(seek)
	if got := countFrames(t, append(head, middle...)); got != "24" {
		t.Fatalf("init + mediaSegment 2 decodes %s frames", got)
	}
}
