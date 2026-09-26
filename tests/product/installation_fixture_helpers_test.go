package producttest

// Native peers carry installed callable metadata alongside their opaque lifetime
// handle. Object hashes remain transport integrity, never installation identity.
var fixturePackageInterface = []byte(`{"application":"fixture:app","entrypoints":[{"name":"tile","request":{"fields":[]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
