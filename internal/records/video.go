package records

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

var videoSchema = []string{`
CREATE TABLE IF NOT EXISTS video_compositions (
  source_digest        TEXT NOT NULL,
  creative_plan_digest TEXT NOT NULL,
  creative_plan        BLOB NOT NULL,
  assets               TEXT NOT NULL,
  created_at           TEXT NOT NULL,
  PRIMARY KEY (source_digest,creative_plan_digest)
)`, `
CREATE INDEX IF NOT EXISTS video_compositions_by_plan
ON video_compositions(creative_plan_digest)`}

type CompositionAsset struct {
	Step      int    `json:"step"`
	FieldPath string `json:"field_path"`
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type"`
	Kind      string `json:"kind"`
	Order     uint32 `json:"order"`
}

type VideoComposition struct {
	SourceDigest       string
	CreativePlanDigest string
	CreativePlan       []byte
	Assets             []CompositionAsset
	CreatedAt          string
}

func (s *Store) RecordVideoComposition(row VideoComposition) (VideoComposition, bool, *exit.Error) {
	assets, err := json.Marshal(row.Assets)
	if err != nil {
		return VideoComposition{}, false,
			exit.Internalf("cannot encode video composition assets: %s", err)
	}
	row.CreatedAt = now()
	result, err := s.db.Exec(`INSERT INTO video_compositions(
		source_digest,creative_plan_digest,creative_plan,assets,created_at)
		VALUES(?,?,?,?,?) ON CONFLICT(source_digest,creative_plan_digest) DO NOTHING`,
		row.SourceDigest, row.CreativePlanDigest, row.CreativePlan, string(assets), row.CreatedAt)
	if err != nil {
		return VideoComposition{}, false,
			exit.Internalf("cannot record video composition %s: %s", row.CreativePlanDigest, err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return row, true, nil
	}
	existing, problem := s.VideoComposition(row.SourceDigest, row.CreativePlanDigest)
	if problem != nil {
		return VideoComposition{}, false, problem
	}
	if existing == nil {
		return VideoComposition{}, false,
			exit.Internalf("video composition conflict committed no readable row")
	}
	if !bytes.Equal(existing.CreativePlan, row.CreativePlan) {
		return VideoComposition{}, false, exit.New(exit.Conflict,
			"video composition %s has changed canonical creative bytes", row.CreativePlanDigest)
	}
	return *existing, false, nil
}

func (s *Store) VideoComposition(sourceDigest, creativeDigest string) (*VideoComposition, *exit.Error) {
	var row VideoComposition
	var assets string
	err := s.db.QueryRow(`SELECT source_digest,creative_plan_digest,creative_plan,assets,created_at
		FROM video_compositions WHERE source_digest=? AND creative_plan_digest=?`,
		sourceDigest, creativeDigest).Scan(&row.SourceDigest,
		&row.CreativePlanDigest, &row.CreativePlan, &assets, &row.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read video composition %s: %s", creativeDigest, err)
	}
	if err := json.Unmarshal([]byte(assets), &row.Assets); err != nil {
		return nil, exit.Internalf("cannot decode video composition %s assets: %s", creativeDigest, err)
	}
	return &row, nil
}

func (s *Store) VideoCompositionByCreativePlan(digest string) (*VideoComposition, *exit.Error) {
	var row VideoComposition
	var assets string
	err := s.db.QueryRow(`SELECT source_digest,creative_plan_digest,creative_plan,assets,created_at
		FROM video_compositions WHERE creative_plan_digest=? ORDER BY created_at,source_digest LIMIT 1`,
		digest).Scan(&row.SourceDigest, &row.CreativePlanDigest, &row.CreativePlan, &assets, &row.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read video composition %s: %s", digest, err)
	}
	if err := json.Unmarshal([]byte(assets), &row.Assets); err != nil {
		return nil, exit.Internalf("cannot decode video composition %s assets: %s", digest, err)
	}
	return &row, nil
}

func (s *Store) compositionAssetInUse(digest string) (bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT assets FROM video_compositions WHERE assets <> '[]'`)
	if err != nil {
		return false, exit.Internalf("cannot read video composition asset ownership: %s", err)
	}
	defer rows.Close()
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return false, exit.Internalf("cannot read video composition assets: %s", err)
		}
		var assets []CompositionAsset
		if err := json.Unmarshal([]byte(raw), &assets); err != nil {
			return false, exit.Internalf("cannot decode video composition assets: %s", err)
		}
		for _, asset := range assets {
			if asset.Digest == digest {
				return true, nil
			}
		}
	}
	return false, nil
}
