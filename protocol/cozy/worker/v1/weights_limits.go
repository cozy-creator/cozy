package workerprotov1

// Weights, model-source, and private-package host-exchange bounds are generated beside the schema.
const MaxWeightsReceipts = 16
const MaxWeightsReceiptBytes = 1 << 20
const MaxWeightsReceiptAggregateBytes = 4 << 20
const MaxWeightsDeclarationBytes = 1 << 20
const MaxWeightsInventoryBytes = 4 << 20
const MaxWeightsObjects = 65536
const MaxWeightsReadBytes = 4 << 20
const MaxWeightsGrantURLBytes = 16 << 10

// MaxInlineControlBytes is the ceiling every content-bearing bound above must respect: the
// control stream carries control, not content. See the schema header; th-094 lands the fence.
const MaxInlineControlBytes = 4 << 20
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
