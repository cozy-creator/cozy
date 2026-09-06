"""Native worker half of Creator's opt-in publication regression.

The Go peer supplies claimed transport. Runtime owns transaction identity, the
native receipt/inventory, object upload and finalization. No host ledger is copied.
"""

import base64
import importlib.metadata
import json
import queue
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace

import tensorfs
from cozy_runtime.author import WeightsConfig, WeightsPart, WeightsTarget, WeightsTensor
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.weights_sink import WeightsHostBinding, WeightsTransactionHost, protocol_receipt
from cozy_runtime.internal.worker.weights import WeightsExchange
from cozy_runtime.internal.worker.weights_finalize import finalize
from cozy_runtime.protocol import documents, worker_pb2 as pb

assert importlib.metadata.version("cozy-runtime") == "0.2.24"
assert importlib.metadata.version("tensorfs") == "0.3.10"
root = Path(sys.argv[1])
store = tensorfs.Store.ensure(str(root / "store"))
owner = None
bridge = "--host-bridge" in sys.argv[2:]
held = None
output_lock = threading.Lock()
incoming = queue.Queue()
stop = threading.Event()


def emit(frame):
    with output_lock:
        print(json.dumps({"frame": base64.b64encode(frame.SerializeToString()).decode()}), flush=True)


def envelope(request):
    return dict(record_owner_epoch=request.record_owner_epoch,
        control_stream_epoch=request.control_stream_epoch, worker_boot_id=request.worker_boot_id)


def send_host(frame):
    inner = getattr(frame, frame.WhichOneof("msg"))
    for key, value in envelope(offer).items():
        setattr(inner, key, value)
    emit(frame)


exchange = WeightsExchange(store_root=root / "store", send=send_host,
    stop=stop, owner_scope=lambda: owner, allow_private_egress=True)


def read_input():
    try:
        for line in sys.stdin:
            if line.startswith("{"):
                message = json.loads(line)
                if "ack" in message:
                    ack = pb.WeightsHostAck.FromString(base64.b64decode(message["ack"], validate=True))
                    exchange.absorb_ack(ack, lane=("control", ack.control_stream_epoch))
                elif "upload" in message:
                    request = pb.WeightsUploadRequest.FromString(base64.b64decode(message["upload"], validate=True))
                    emit(exchange.upload(request))
                else:
                    raise RuntimeError("unexpected Host bridge message")
            else:
                incoming.put(pb.RecordOwnerFrame.FromString(base64.b64decode(line.strip(), validate=True)))
    finally:
        stop.set()
        incoming.put(None)


threading.Thread(target=read_input, daemon=True).start()
try:
    while (frame := incoming.get()) is not None:
        kind = frame.WhichOneof("msg")
        if kind == "claim":
            owner = frame.claim.record_owner_id
        elif kind == "attempt_offer":
            assert owner
            offer = frame.attempt_offer
            digest = documents.spell(offer.invocation_spec_digest)
            declaration_digest = None

            attempt = SimpleNamespace(request_id=offer.request_id, attempt=offer.attempt_ordinal,
                digest=offer.invocation_spec_digest, canceling=False, weights_receipts={})

            def bind(slot, transaction_id, native_digest, declaration):
                global declaration_digest
                declaration_digest = native_digest
                if bridge:
                    result = exchange.intent(attempt, output_slot=slot, transaction_id=transaction_id,
                        declaration_digest=native_digest, declaration=declaration, requested_writer_epoch=0)
                    return WeightsHostBinding(result["weights_transaction_id"], result["writer_epoch"])
                return WeightsHostBinding(transaction_id, 1)

            def record(receipt):
                reference, canonical, digest = protocol_receipt(receipt, owner_scope=owner,
                    request_id=offer.request_id, invocation_spec_digest=documents.spell(offer.invocation_spec_digest))
                exchange.receipt(attempt, output_slot="model", transaction_id=receipt.weights_transaction_id,
                    receipt_digest=digest, canonical_receipt=canonical)

            host = WeightsTransactionHost(store=store, owner_scope=owner, request_id=offer.request_id,
                invocation_spec_digest=digest, writer_session_id=1, allowed_sources={},
                output_bounds={"model": 1 << 20}, bind_intent=bind, record_receipt=record if bridge else None)
            config = json.dumps({"proof": offer.request_id}).encode()
            transaction = host.open(WeightsCommit(output_slot="model", sources={},
                targets={"transformer": WeightsTarget(add={"weight": WeightsTensor(
                    logical_dtype="f32", shape=(1024,), encoding=SPEC_PLAIN,
                    parts={"value": WeightsPart(dtype="f32", shape=(1024,))})})},
                configs={"model": WeightsConfig(data=config)}, order=(("transformer", "weight"),),
                max_new_bytes=1 << 20))
            transaction.add_part("transformer", "weight", "value", struct.pack("<1024f", *range(1024)))
            transaction.add_config("model", config, len(config))
            receipt = transaction.commit()
            reference, _, _ = protocol_receipt(receipt, owner_scope=owner,
                request_id=offer.request_id, invocation_spec_digest=digest)
            held = exchange._hold(receipt.weights_transaction_id, 1, reference)
            if not bridge:
                emit(pb.WorkerFrame(weights_receipt=pb.WeightsReceiptFrame(**envelope(offer),
                    request_id=offer.request_id, attempt_ordinal=offer.attempt_ordinal,
                    invocation_spec_digest=offer.invocation_spec_digest, output_slot="model",
                    weights_transaction_id=receipt.weights_transaction_id, writer_epoch=1,
                    tensorfs_declaration_digest=declaration_digest,
                    weights_receipt=reference, manifest=held.manifest,
                    objects=[held.objects[key] for key in sorted(held.objects)])))
            body, outcome_digest = documents.identity(pb.AttemptOutcomeBody(
                request_id=offer.request_id, attempt_ordinal=offer.attempt_ordinal,
                invocation_spec_digest=digest, status=pb.OUTCOME_STATUS_SUCCEEDED,
                execution_started=True, cause=pb.OutcomeCause(origin=pb.CAUSE_ORIGIN_RUNTIME), weights_receipts=[reference]))
            emit(pb.WorkerFrame(attempt_outcome=pb.AttemptOutcome(**envelope(offer),
                request_id=offer.request_id, attempt_ordinal=offer.attempt_ordinal,
                invocation_spec_digest=offer.invocation_spec_digest, outcome_id="native-output",
                outcome_digest=outcome_digest, outcome_canonical_bytes=body)))
        elif kind == "weights_finalize_request":
            request = frame.weights_finalize_request
            result = finalize(request, owner_scope=owner, attempts=[], tensorfs_root=root / "store",
                journal_root=root / "worker", on_abandon=exchange.abandon)
            result.record_owner_epoch = request.record_owner_epoch
            result.control_stream_epoch = request.control_stream_epoch
            result.worker_boot_id = request.worker_boot_id
            emit(pb.WorkerFrame(weights_finalize_result=result))
        elif kind == "weights_transfer_request":
            if bridge:
                raise RuntimeError("Host bridge did not consume the transfer request")
            request = frame.weights_transfer_request
            assert held is not None
            grant = request.upload_grant if request.WhichOneof("decision") == "upload_grant" else request.held
            obj = held.objects[grant.object_id]
            assert obj.length == grant.length
            status = pb.WeightsTransferStatus(**envelope(request), request_id=request.request_id,
                attempt_ordinal=request.attempt_ordinal, invocation_spec_digest=request.invocation_spec_digest,
                output_slot=request.output_slot, weights_transaction_id=request.weights_transaction_id,
                operation_id=request.operation_id, object_id=obj.object_id, length=obj.length,
                grant_revision=request.grant_revision, update_sequence=1,
                state=pb.WEIGHTS_TRANSFER_STATE_HELD)
            if request.WhichOneof("decision") == "upload_grant":
                result = exchange.upload(pb.WeightsUploadRequest(**envelope(request),
                    weights_transaction_id=request.weights_transaction_id, writer_epoch=1,
                    operation_id=request.operation_id, object_id=obj.object_id,
                    source_ref=obj.source_ref, length=obj.length, grant=grant,
                    grant_revision=request.grant_revision)).weights_upload_result
                assert result.outcome in (pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED,
                    pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT), result.safe_detail
                status.state = (pb.WEIGHTS_TRANSFER_STATE_UPLOADED if result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED
                    else pb.WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT)
                status.transferred_bytes = result.transferred_bytes
                status.http_status = result.http_status
                status.checksum_sha256 = result.checksum_sha256
            emit(pb.WorkerFrame(weights_transfer_status=status))
        elif kind == "outcome_ack":
            break
        else:
            raise RuntimeError("unexpected publication frame " + str(kind))
finally:
    exchange.close()
