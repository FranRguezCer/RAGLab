# Harbor Edge Cache Deployment Guide

Harbor is a fictional HTTP edge cache used for repeatable operations training. It
stores only public documentation responses. Private API responses and personalized
pages always bypass the cache.

## Cache identity and freshness

Every cache key contains the tenant ID, normalized path, and content language. It
must not include the Authorization header or a raw query-string order. Query
parameters are sorted before the key is calculated so equivalent URLs share an
entry.

Public documentation has a default freshness lifetime of 300 seconds. A release
may shorten that lifetime, but it cannot extend it beyond 900 seconds. Emergency
invalidation uses surrogate keys rather than a full-cache flush. Each published
page carries one product key and one release key.

## Canary deployment

A new cache policy first receives 5 percent of eligible traffic for ten minutes.
The operator watches origin error rate, cache-hit ratio, and response latency. If
the canary produces more than 2 percent HTTP 5xx responses for three consecutive
one-minute windows, roll back immediately. A healthy canary may advance to 25
percent and then 100 percent.

After rollback, purge the affected release with its surrogate keys. Do not purge
unrelated tenants. Record the policy version and canary window in the deployment
log before traffic is restored.

## Origin retries

Harbor retries an origin request only for `GET` or `HEAD`, and only once. It never
retries `POST`, `PATCH`, or `DELETE`. A retry adds the `Harbor-Retry: 1` header so
the origin log can distinguish the second attempt.
