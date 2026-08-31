package workerprotov1

// Artifact, model-source, and private-package host-exchange bounds are generated beside the schema.
const MaxArtifactReceipts = 16
const MaxArtifactReceiptBytes = 1 << 20
const MaxArtifactReceiptAggregateBytes = 4 << 20
const MaxArtifactDeclarationBytes = 1 << 20
const MaxArtifactInventoryBytes = 4 << 20
const MaxArtifactObjects = 65536
const MaxArtifactReadBytes = 4 << 20
const MaxArtifactGrantURLBytes = 16 << 10
const MaxModelSourceFiles = 4096
const MaxModelSourceProfiles = 16
const MaxModelSourceMemberBytes = 1024
const MaxModelSourceProfileBytes = 1024
const MaxModelSourceURIBytes = 4096
const MaxModelSourceLicenseBytes = 4096
const MaxModelSourceURLBytes = 16 << 10
const MaxCheckpointEvidenceBytes = 64 << 10
const MaxPrivatePackageFiles = 33
const MaxPrivatePackageChunkBytes = 1 << 20
const MaxPrivatePackageFilenameBytes = 255
const MaxPrivatePackageFileBytes = 512 << 20
const MaxPrivatePackageAggregateBytes = 1 << 30
