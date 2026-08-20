# Meridian Incident Response Runbook

Meridian is a fictional service used for deterministic incident-response
exercises. This runbook covers production service incidents, not employee account
or password administration.

## Declare and stabilize

Declare severity S1 when all regions are unavailable or confirmed data loss is in
progress. Declare severity S2 when one region is unavailable and traffic can fail
over. The incident commander owns the timeline; the operations lead owns changes
to production.

Before changing the system, capture the alert, deployment ID, and UTC timestamp in
the incident log. Use the monitoring clock as the shared time source. Screenshots
are supporting material, not a substitute for the text timeline.

If an active deployment caused the failure, stop the rollout and restore the last
known-good artifact. Keep the database schema unchanged until compatibility with
that artifact is confirmed. Every production command must include the incident ID
in its audit note.

## Recover and verify

Recovery requires two checks: the error budget burn has returned to its normal
range, and a synthetic request succeeds from two regions. Observe both checks for
fifteen minutes before resolving the incident. A quiet alert alone is not proof
of recovery.

When stale cached pages remain after the service is healthy, ask the cache
operator to invalidate only the affected release. The incident commander records
the released artifact, the affected region, and the final customer impact.

## Preserve evidence

Export application logs and change records to the read-only incident archive.
Retain the original alert payload even when its title was misleading. Finish the
timeline before writing causal conclusions; the post-incident review may correct
those conclusions without rewriting the observed events.
