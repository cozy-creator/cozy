package producttest

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
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

type indexedFilmPacket struct {
	PTS      string `json:"pts_time"`
	DTS      string `json:"dts_time"`
	Duration string `json:"duration_time"`
	Hash     string `json:"data_hash"`
}

// Finalization changes the container, not the video. Exercise the bytes delivered
// to each consumer, including fresh seeks into predictive frames between keyframes.
func assertIndexedFilm(t *testing.T, final []byte, previews [][]byte, frames, fps int) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "final.mp4")
	must(t, os.WriteFile(path, final, 0600))
	raw, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height,nb_frames,duration:format=duration", "-of", "json", path).Output()
	must(t, err)
	var facts struct {
		Streams []struct {
			Width, Height int
			Frames        string `json:"nb_frames"`
			Duration      string `json:"duration"`
		}
		Format struct{ Duration string }
	}
	must(t, json.Unmarshal(raw, &facts))
	if len(facts.Streams) != 1 || facts.Streams[0].Frames != strconv.Itoa(frames) {
		t.Fatalf("completed MP4 has no complete frame index: %s", raw)
	}
	videoDuration, err := strconv.ParseFloat(facts.Streams[0].Duration, 64)
	must(t, err)
	fileDuration, err := strconv.ParseFloat(facts.Format.Duration, 64)
	must(t, err)
	wantDuration := float64(frames) / float64(fps)
	if math.Abs(videoDuration-wantDuration) > 0.000001 || math.Abs(fileDuration-wantDuration) > 1/float64(fps) {
		t.Fatalf("completed MP4 duration differs from its frame clock: %s", raw)
	}
	packets := func(path string) []indexedFilmPacket {
		t.Helper()
		raw, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
			"-show_data_hash", "sha256", "-show_entries", "packet=pts_time,dts_time,duration_time,data_hash", "-of", "json", path).Output()
		must(t, err)
		var data struct{ Packets []indexedFilmPacket }
		must(t, json.Unmarshal(raw, &data))
		return data.Packets
	}
	decode := func(path string, before ...string) []byte {
		t.Helper()
		args := append([]string{"-v", "error"}, before...)
		args = append(args, "-i", path, "-map", "0:v:0")
		if len(before) != 0 {
			args = append(args, "-frames:v", "1")
		}
		out, err := exec.Command("ffmpeg", append(args, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1")...).Output()
		must(t, err)
		return out
	}
	encoded, decoded := packets(path), decode(path)
	frameBytes := facts.Streams[0].Width * facts.Streams[0].Height * 3
	if len(encoded) != frames || len(decoded) != frames*frameBytes {
		t.Fatalf("completed MP4 packet/frame counts differ: packets=%d RGB bytes=%d", len(encoded), len(decoded))
	}
	for _, packet := range encoded {
		if packet.PTS == "" || packet.DTS == "" || !strings.HasPrefix(packet.Hash, "SHA256:") {
			t.Fatalf("packet comparison lacks timestamps or byte identity: %+v", packet)
		}
	}
	for i, preview := range previews {
		live := filepath.Join(dir, fmt.Sprintf("preview-%d.mp4", i))
		must(t, os.WriteFile(live, preview, 0600))
		prior := packets(live)
		if len(prior) == 0 || len(prior) >= len(encoded) || !slices.Equal(prior, encoded[:len(prior)]) || !bytes.HasPrefix(decoded, decode(live)) {
			t.Fatalf("finalization changed preview %d's packets, timestamps, or decoded frames", i+1)
		}
	}
	for _, frame := range []int{1, frames/3 + 1, 2*frames/3 + 1, frames - 2} {
		// Seek midway between the preceding frame and this one, avoiding decimal
		// rounding across a frame boundary while forcing a cold predictive decode.
		position := fmt.Sprintf("%.9f", (float64(frame)-0.5)/float64(fps))
		if got := decode(path, "-ss", position); !bytes.Equal(got, decoded[frame*frameBytes:(frame+1)*frameBytes]) {
			t.Fatalf("cold seek near frame %d differs from sequential decoding", frame)
		}
	}
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
