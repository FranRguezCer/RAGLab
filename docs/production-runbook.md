# Operate the public RAGLab portfolio

Production runs only Traefik and the CPU-only portfolio image. The public app serves the verified evidence embedded in that image; PostgreSQL, Ollama, models, and arbitrary queries stay off the server.

## Bootstrap the server

Use a Hetzner host with at least 2 vCPU and 4 GB RAM. Prefer CAX11 when available, then CX23, then the smallest shared plan meeting those limits. The published portfolio image supports both `linux/amd64` and `linux/arm64`.

1. Install current Docker Engine with the Compose plugin on an Ubuntu or Debian host.
2. Create `/srv/raglab`, owned by the unprivileged SSH deployment user.
3. Allow inbound TCP 22, 80, and 443 only. Restrict SSH by source IP when practical.
4. Add the GitHub Actions public key to the deployment user's `authorized_keys`. Do not permit password or root SSH login.
5. Copy `scripts/deploy_portfolio.sh` to `/srv/raglab/deploy_portfolio.sh` and make it executable.
6. Resolve and pin Traefik before configuring the host:

   ```bash
   docker pull traefik:v3.5
   docker image inspect traefik:v3.5 --format '{{index .RepoDigests 0}}'
   ```

7. Create `/srv/raglab/.env` with mode `0600`:

   ```dotenv
   RAGLAB_DOMAIN=raglab.example.com
   ACME_EMAIL=admin@example.com
   TRAEFIK_IMAGE=traefik@sha256:<64-hex-digest>
   ```

The deploy script rejects images without an immutable digest and rejects any production Compose file containing services other than `proxy` and `portfolio`.

## Configure DNS and GitHub

Point the domain's `A` record to the server's public IPv4 address. Add an `AAAA` record only when IPv6 is configured and reachable. Confirm both records resolve before releasing so Let's Encrypt can complete its TLS challenge.

Create a protected GitHub environment named `production`, require a reviewer, and add:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | Hostname or IP only, for example `203.0.113.10` |
| `DEPLOY_USER` | Dedicated unprivileged SSH user, for example `deploy` |
| `DEPLOY_SSH_KEY` | Dedicated private deployment key |
| `DEPLOY_HOST_KEY` | Exact `known_hosts` line collected out of band with `ssh-keyscan -H <host>` and verified against the server fingerprint |

GHCR remains public and read-only on the host; no registry credential is needed.

Create a repository-scoped self-hosted runner with the `raglab-gpu` label. Do not add it to an
organization-wide runner group and do not give pull-request workflows that label. The only job
using it is the manually dispatched release evaluation after the candidate SHA and successful
`main` CI run have been verified on a GitHub-hosted runner.

Protect `main` with a ruleset that requires pull requests and the **Quality and build** status
check, blocks force pushes and branch deletion, and dismisses stale approvals. Configure the
`production` environment with required reviewers so deployment cannot start merely because the
GPU gate passed. Finally, make both GHCR packages public after their first publication.

## Release

Start the manual release workflow with a full commit SHA from `main`. After CI, GPU evaluation, the evidence gate, image publication, and production approval, the workflow invokes:

```bash
bash /srv/raglab/deploy_portfolio.sh \
  /tmp/compose.production.yaml \
  ghcr.io/OWNER/raglab-portfolio@sha256:<64-hex-digest> \
  <40-character-build-sha>
```

The script validates Compose, pulls the digest, waits for readiness, then exercises the status and every curated case without a token. Traffic reaches only an evidence file whose build SHA matches the release input.

Verify independently:

```bash
curl --fail --silent --show-error https://raglab.example.com/health/ready
curl --fail --silent --show-error https://raglab.example.com/v1/demo/status
curl --fail --silent --show-error https://raglab.example.com/v1/demo/cases
```

## Roll back

Failed readiness or smoke checks automatically restore both the prior Compose file and its prior digest-pinned image. To roll back the last successful release manually:

```bash
bash /srv/raglab/deploy_portfolio.sh --rollback
```

The rollback is serialized with deployments and must pass the same readiness and smoke checks. Inspect state with:

```bash
cat /srv/raglab/.last-good-image
cat /srv/raglab/.last-good-build-sha
docker compose --env-file /srv/raglab/.env \
  --env-file /srv/raglab/.release.env \
  -f /srv/raglab/compose.production.yaml ps
```

If both deployment and rollback fail, stop and inspect `docker compose logs`; do not edit `.release.env` by hand. Restore the files under `/srv/raglab/.rollback/`, then rerun `--rollback` after correcting the host failure.
