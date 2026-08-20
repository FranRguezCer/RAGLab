# Orbit Telemetry Handbook

Orbit is a fictional observability library for small web services. Its default
dashboard follows the RED signals: request rate, error rate, and duration.

## Metrics

Record request duration in a histogram, not a gauge. The standard bucket bounds
are 50, 100, 250, 500, 1000, and 2500 milliseconds. These buckets support a stable
p95 calculation while preserving the distribution between releases.

Metric labels must have bounded cardinality. Allowed request labels are route
template, region, HTTP method, and status class. Never use a user ID, session ID,
or raw request path as a metric label. Dynamic identifiers belong in traces and
structured logs.

The release dashboard shows request volume, HTTP 5xx rate, p95 duration, and the
deployment marker on the same time axis. Compare the canary with the previous
release over identical windows; unrelated daily traffic changes otherwise hide
small regressions.

## Traces and logs

Sample 5 percent of successful traces and retain every trace that contains an
error. Propagate the trace ID across HTTP and queue boundaries. Logs include the
trace ID, service name, region, and deployment ID, but they exclude access tokens
and request bodies.

## Alerts

An alert must describe a user-visible symptom and link to its dashboard. Page the
operator when p95 duration exceeds 750 milliseconds for ten minutes in two
regions. Ticket-only alerts cover a single-region rise that lasts less than ten
minutes. Do not create a page from one slow request.
