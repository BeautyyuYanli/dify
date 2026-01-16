"""
Logfire tracing provider (minimal OTLP/HTTP).

This provider is intentionally implemented in the simplest possible way, similar
to other providers like Langfuse/LangSmith:
- It exposes `trace()` as the single entrypoint (called by the ops trace worker).
- It validates config via `api_check()` when the user saves credentials.
- It builds ONE OpenTelemetry span per Dify trace entity.

We intentionally do NOT:
- fetch workflow node executions from DB,
- create nested spans for nodes/tools/LLM,
- depend on OpenInference semantic convention packages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from urllib.parse import urlparse

import httpx
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpOTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)
from opentelemetry.util.types import AttributeValue

from configs import dify_config
from core.ops.base_trace_instance import BaseTraceInstance
from core.ops.entities.config_entity import LogfireConfig
from core.ops.entities.trace_entity import (
    BaseTraceInfo,
    DatasetRetrievalTraceInfo,
    GenerateNameTraceInfo,
    MessageTraceInfo,
    ModerationTraceInfo,
    SuggestedQuestionTraceInfo,
    ToolTraceInfo,
    TraceTaskName,
    WorkflowTraceInfo,
)

logger = logging.getLogger(__name__)


def _build_minimal_otlp_trace_payload(*, service_name: str) -> bytes:
    """
    Build a minimal OTLP/HTTP protobuf payload for Logfire connectivity checks.

    Logfire validates that request bodies decode as an `ExportTraceServiceRequest` protobuf.
    An empty request body will be rejected (commonly as 4xx like 422).
    """
    # Import OTEL protobuf models lazily to keep import overhead minimal.
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, InstrumentationScope, KeyValue
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource as OtelResource
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans
    from opentelemetry.proto.trace.v1.trace_pb2 import Span as OtelSpan

    now_ns = time.time_ns()

    # Deterministic-enough IDs; must be correct lengths (trace_id=16 bytes, span_id=8 bytes).
    trace_id = hashlib.sha256(f"{service_name}:{now_ns}".encode()).digest()[:16]
    span_id = hashlib.sha256(f"{service_name}:{now_ns}:span".encode()).digest()[:8]

    # Include `service.name` since it’s a widely expected OTEL resource attribute.
    resource = OtelResource(
        attributes=[
            KeyValue(key="service.name", value=AnyValue(string_value=service_name)),
        ]
    )

    span = OtelSpan(
        trace_id=trace_id,
        span_id=span_id,
        name="dify.logfire.api_check",
        kind=OtelSpan.SpanKind.SPAN_KIND_INTERNAL,
        start_time_unix_nano=now_ns,
        end_time_unix_nano=now_ns + 1,
    )

    request = ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=resource,
                scope_spans=[
                    ScopeSpans(
                        scope=InstrumentationScope(name="dify.logfire"),
                        spans=[span],
                    )
                ],
            )
        ]
    )

    return request.SerializeToString()


def _normalize_logfire_traces_endpoint(endpoint: str) -> str:
    """
    Normalize a Logfire endpoint into an OTLP traces endpoint URL.

    Logfire accepts OTLP/HTTP at `/v1/traces`. Users may provide:
    - a full traces endpoint (e.g. `https://logfire-us.pydantic.dev/v1/traces`)
    - a base endpoint (e.g. `https://logfire-us.pydantic.dev`)
    - a v1 endpoint (e.g. `https://logfire-us.pydantic.dev/v1`)
    """
    endpoint = (endpoint or "").strip().rstrip("/")
    if not endpoint:
        return "https://logfire-api.pydantic.dev/v1/traces"
    if endpoint.endswith("/v1/traces"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/traces"
    return f"{endpoint}/v1/traces"


def _datetime_to_nanos(dt: datetime | None) -> int | None:
    """Convert datetime to nanoseconds (OTEL expects ns)."""
    if dt is None:
        return None
    return int(dt.timestamp() * 1_000_000_000)


def _safe_json_dumps(obj: object) -> str:
    """Safely encode any object to JSON for span attributes."""
    return json.dumps(obj, default=str, ensure_ascii=False)


def _stable_trace_and_parent_span_id(seed: str) -> tuple[int, int]:
    """
    Build stable OTEL ids from an input seed.

    We create a synthetic parent `SpanContext` from the seed and then create a child span.
    This forces a stable trace_id for correlation in Logfire.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    trace_id = int.from_bytes(digest[:16], byteorder="big", signed=False)
    span_id = int.from_bytes(digest[16:24], byteorder="big", signed=False)

    # Avoid invalid IDs (all zeros).
    if trace_id == 0:
        trace_id = 1
    if span_id == 0:
        span_id = 1
    return trace_id, span_id


def _get_trace_seed(trace_info: BaseTraceInfo) -> str:
    """
    Pick a stable seed string for trace correlation.

    Priority:
    - explicit trace_id if present
    - message_id (common across multiple trace types)
    - fallback to a deterministic hash of metadata
    """
    if trace_info.trace_id:
        return str(trace_info.trace_id)
    if trace_info.message_id:
        return str(trace_info.message_id)
    return hashlib.sha256(_safe_json_dumps(trace_info.metadata).encode("utf-8")).hexdigest()


def _set_span_status(span: Span, error: object | None) -> None:
    """Set span status and add a small error message when available."""
    if error:
        span.set_status(Status(StatusCode.ERROR, str(error)))
    else:
        span.set_status(Status(StatusCode.OK))


def _build_attributes(trace_info: BaseTraceInfo, trace_type: str) -> dict[str, AttributeValue]:
    """
    Convert a Dify trace entity into span attributes.

    Keep attributes small and predictable. Payloads are stored as JSON strings.
    """
    # Core identity fields.
    attributes: dict[str, AttributeValue] = {
        "dify.trace.type": trace_type,
        "dify.deploy_env": str(dify_config.DEPLOY_ENV),
        "dify.edition": str(dify_config.EDITION),
    }

    # Include high-signal metadata keys if present.
    for key in ("tenant_id", "app_id", "workflow_id", "workflow_run_id", "conversation_id", "user_id"):
        value = trace_info.metadata.get(key)
        if value is not None and value != "":
            attributes[f"dify.meta.{key}"] = str(value)

    # Include trace inputs/outputs as JSON strings.
    if trace_info.inputs is not None:
        attributes["dify.inputs"] = _safe_json_dumps(trace_info.inputs)
    if trace_info.outputs is not None:
        attributes["dify.outputs"] = _safe_json_dumps(trace_info.outputs)

    # Include the raw metadata for debugging (as a single JSON string).
    attributes["dify.metadata"] = _safe_json_dumps(trace_info.metadata)

    return attributes


class LogfireDataTrace(BaseTraceInstance):
    """
    Minimal Logfire tracer implementation.

    It sends OTEL spans to Logfire via OTLP/HTTP with a write token.
    """

    # --- typed members (declare in class, assign in __init__) ---
    logfire_config: LogfireConfig
    tracer: trace_sdk.Tracer
    span_processor: SimpleSpanProcessor

    def __init__(self, logfire_config: LogfireConfig):
        """Initialize OTLP exporter + tracer for Logfire."""
        super().__init__(logfire_config)

        # Keep the provider config for URL building / api checks.
        self.logfire_config = logfire_config

        # Normalize endpoint into a full OTLP traces endpoint.
        traces_endpoint = _normalize_logfire_traces_endpoint(logfire_config.endpoint)

        # Build an OTEL resource so spans are searchable by service/version/environment.
        resource = Resource(
            attributes={
                ResourceAttributes.SERVICE_NAME: dify_config.APPLICATION_NAME,
                ResourceAttributes.SERVICE_VERSION: f"dify-{dify_config.project.version}-{dify_config.COMMIT_SHA}",
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: f"{dify_config.DEPLOY_ENV}-{dify_config.EDITION}",
                "logfire.organization": (logfire_config.organization or ""),
                "logfire.project": (logfire_config.project or ""),
            }
        )

        # Configure OTLP/HTTP exporter with Logfire write token.
        exporter = HttpOTLPSpanExporter(
            endpoint=traces_endpoint,
            headers={"Authorization": logfire_config.write_token},
            timeout=30,
        )

        # Use a per-instance tracer provider so config changes do not require process restart.
        provider = trace_sdk.TracerProvider(resource=resource)

        # Export spans immediately (no background worker thread).
        self.span_processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(self.span_processor)

        # Create a named tracer (do not overwrite global provider).
        tracer_name = f"logfire_tracer_{logfire_config.project or 'default'}"
        self.tracer = provider.get_tracer(tracer_name)

    def trace(self, trace_info: BaseTraceInfo) -> None:
        """
        Entry point: export a single Dify trace entity as one OTEL span.

        This is called by `tasks/ops_trace_task.py`.
        """
        # Decide the span name based on the trace entity type.
        if isinstance(trace_info, WorkflowTraceInfo):
            span_name = TraceTaskName.WORKFLOW_TRACE.value
        elif isinstance(trace_info, MessageTraceInfo):
            span_name = TraceTaskName.MESSAGE_TRACE.value
        elif isinstance(trace_info, ModerationTraceInfo):
            span_name = TraceTaskName.MODERATION_TRACE.value
        elif isinstance(trace_info, SuggestedQuestionTraceInfo):
            span_name = TraceTaskName.SUGGESTED_QUESTION_TRACE.value
        elif isinstance(trace_info, DatasetRetrievalTraceInfo):
            span_name = TraceTaskName.DATASET_RETRIEVAL_TRACE.value
        elif isinstance(trace_info, ToolTraceInfo):
            span_name = TraceTaskName.TOOL_TRACE.value
        elif isinstance(trace_info, GenerateNameTraceInfo):
            span_name = TraceTaskName.GENERATE_NAME_TRACE.value
        else:
            # Unknown types still get exported but are labeled clearly.
            span_name = "dify.unknown_trace"

        # Create a stable trace id for correlation across spans.
        seed = _get_trace_seed(trace_info)
        trace_id, parent_span_id = _stable_trace_and_parent_span_id(seed)

        # Create a synthetic parent context to force the trace_id.
        parent_ctx = SpanContext(
            trace_id=trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=None,
        )
        ctx = trace_api.set_span_in_context(NonRecordingSpan(parent_ctx))

        # Build attributes (small + predictable).
        attributes = _build_attributes(trace_info=trace_info, trace_type=span_name)

        # Convert timestamps when available.
        start_time = _datetime_to_nanos(trace_info.start_time)
        end_time = _datetime_to_nanos(trace_info.end_time)

        # Start a span (child of the synthetic parent).
        span = self.tracer.start_span(
            name=span_name,
            context=ctx,
            attributes=attributes,
            start_time=start_time,
        )
        try:
            # Mark error status if the entity carries an `error` field.
            error_value = getattr(trace_info, "error", None)
            _set_span_status(span, error_value)
        finally:
            # End span using the entity's end_time if available.
            span.end(end_time=end_time)

    def api_check(self) -> bool:
        """
        Validate that the endpoint is reachable and the token is accepted.

        Logfire behaves as an OTLP/HTTP server and requires requests to contain a valid
        protobuf-encoded `ExportTraceServiceRequest`. Sending an empty payload will fail.
        """
        traces_endpoint = _normalize_logfire_traces_endpoint(self.logfire_config.endpoint)
        try:
            parsed = urlparse(traces_endpoint)
            if parsed.scheme not in {"https", "http"}:
                raise ValueError("Endpoint URL must start with https:// or http://")

            service_name = str(dify_config.APPLICATION_NAME or "dify")
            payload = _build_minimal_otlp_trace_payload(service_name=service_name)

            response = httpx.post(
                traces_endpoint,
                headers={
                    "Authorization": self.logfire_config.write_token,
                    # OTLP/HTTP uses protobuf content type.
                    "Content-Type": "application/x-protobuf",
                },
                content=payload,
                timeout=5.0,
            )
            if response.status_code == 200:
                return True
            if response.status_code in (401, 403):
                return False
            raise ValueError(f"Logfire API check failed: {response.status_code} {response.text}")
        except httpx.RequestError as exc:
            raise ValueError(f"Logfire API check failed: {exc}") from exc

    def get_project_url(self) -> str:
        """Build a Logfire project URL for the UI 'View' button."""
        org = (self.logfire_config.organization or "").strip().strip("/")
        project = (self.logfire_config.project or "").strip().strip("/")
        if org and project:
            return f"https://logfire.pydantic.dev/{org}/{project}"
        return "https://logfire.pydantic.dev/"
