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

type indexedFilmPacket struct {
	PTS      string `json:"pts_time"`
	DTS      string `json:"dts_time"`
	Duration string `json:"duration_time"`
	Hash     string `json:"data_hash"`
}

func indexedMP4(data []byte) bool {
	moov := false
	for at := 0; at+8 <= len(data); {
		size := uint64(binary.BigEndian.Uint32(data[at:]))
		kind := string(data[at+4 : at+8])
		if size == 1 && at+16 <= len(data) {
			size = binary.BigEndian.Uint64(data[at+8:])
		}
		if size < 8 || size > uint64(len(data)-at) || kind == "moof" {
			return false
		}
		moov = moov || kind == "moov"
		at += int(size)
	}
	return moov
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
