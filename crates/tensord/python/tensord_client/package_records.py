"""Consumed package metadata, the helper's captured environment and the in-environment description records."""
import msgspec


class Callee(msgspec.Struct, frozen=True):
    """Another package's App this environment holds as a dependency: a job of the root calls
    its invocables as child runs here."""
    distribution: str
    version: str
    application: str
    interface: msgspec.Raw
    source_digest: str = ""
    package: str = ""


class ApplicationEntry(msgspec.Struct, frozen=True):
    default: str


class EntryPoints(msgspec.Struct, frozen=True):
    application: ApplicationEntry = msgspec.field(name="cozy.application")


class Project(msgspec.Struct, frozen=True):
    name: str
    version: str
    entry_points: EntryPoints = msgspec.field(name="entry-points")


class Pyproject(msgspec.Struct, frozen=True):
    project: Project


class PackageMetadata(msgspec.Struct, frozen=True):
    name: str
    version: str
    application: str


class Describe(msgspec.Struct, tag="describe", tag_field="kind"):
    project: str
    environment_python: str | None = None


class DescribeInstalled(msgspec.Struct, tag="describe_installed", tag_field="kind"):
    distribution: str
    environment_python: str


class DescribeEnvironment(msgspec.Struct, tag="describe_environment", tag_field="kind"):
    root: str
    packages: dict[str, str] = {}
    # Also the root's own installed interface, in the same process.
    interface: bool = False


class DescribedEnvironment(msgspec.Struct, tag="described_environment", tag_field="kind"):
    source_digest: str
    callees: list[Callee]
    # The root's interface document as JSON text, when asked for.
    interface: str = ""


class Described(msgspec.Struct, tag="described", tag_field="kind"):
    interface: msgspec.Raw


class DescribeFailed(msgspec.Struct, tag="describe_failed", tag_field="kind"):
    code: str
    detail: str


class InstallFailed(msgspec.Struct, tag="install_failed", tag_field="kind"):
    code: str
    detail: str


class Captured(msgspec.Struct, tag="captured", tag_field="kind"):
    """One environment of a local package's own closure (`captured_packages`); the machine
    installs its pair, its client and its record next (`environments.rs`)."""
    package: str
    version: str
    application: str
    python: str
    #: An editable environment already ready: what the machine kept of it (`.ready`).
    ready: str | None = None
