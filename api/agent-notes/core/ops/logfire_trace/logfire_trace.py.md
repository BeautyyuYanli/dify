# Notes: `core/ops/logfire_trace/logfire_trace.py`

## Purpose

This module implements the **Logfire tracing provider** for Dify’s Ops tracing pipeline.
It exports **one OpenTelemetry span per Dify trace entity** via **OTLP/HTTP** to Logfire.

## Where it’s used

- Trace tasks are produced by `core/ops/ops_trace_manager.py` and processed in `tasks/ops_trace_task.py`.
- The active provider is instantiated via `OpsTraceManager.get_ops_trace_instance(app_id)`.
- `LogfireDataTrace.trace()` is called with a `BaseTraceInfo` subclass.

## Minimal requirements for Logfire OTLP ingest

Logfire receives traces via **OTLP/HTTP**:

- **Endpoint**: must be a traces endpoint ending in `/v1/traces` (region-specific host is allowed).
- **Headers**:
  - `Authorization`: Logfire **write token**.
  - `Content-Type`: `application/x-protobuf` (protobuf OTLP payload).
- **Body**: must be a protobuf-encoded `ExportTraceServiceRequest`.
  - Empty bodies are rejected by Logfire (observed as 4xx like 422).

## Key decisions / invariants

- **Endpoint normalization**: `_normalize_logfire_traces_endpoint()` accepts:
  - base URL (`https://logfire-<region>.pydantic.dev`)
  - v1 base (`.../v1`)
  - full traces URL (`.../v1/traces`)
  and normalizes to a full traces endpoint.
- **Span export strategy**: uses `SimpleSpanProcessor` to export immediately, avoiding background
  batching/thread timing issues when running in workers.
- **Service identity**: `service.name` is set via `dify_config.APPLICATION_NAME` on the OTEL `Resource`
  and in the minimal payload used for connectivity checks.

## Config validation (`api_check`)

`LogfireDataTrace.api_check()` validates that the configured endpoint is a working OTLP/HTTP server
and that the token is accepted:

- It POSTs a **minimal valid OTLP protobuf** payload to `/v1/traces`.
- Success is only `HTTP 200`.
- `401/403` is treated as invalid credentials.
- Other statuses raise a `ValueError` to surface misconfiguration (wrong endpoint/path, proxy issues, etc.).

## Verification plan

- Save Logfire tracing config in console; `api_check()` should succeed for valid token/endpoint.
- Trigger a Dify trace (chat/workflow) and confirm a new trace appears in Logfire.

