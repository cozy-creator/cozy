package modelsource

// MetadataFile is one small reviewed provider file a model ingestion recipe names.
type MetadataFile struct {
	Length int64  `json:"length"`
	SHA256 string `json:"sha256"`
}
