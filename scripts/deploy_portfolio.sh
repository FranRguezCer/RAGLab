#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="${RAGLAB_DEPLOY_ROOT:-/srv/raglab}"
LIVE_COMPOSE="$DEPLOY_ROOT/compose.production.yaml"
BASE_ENV="$DEPLOY_ROOT/.env"
RELEASE_ENV="$DEPLOY_ROOT/.release.env"
ROLLBACK_DIR="$DEPLOY_ROOT/.rollback"
LOCK_FILE="$DEPLOY_ROOT/.deploy.lock"

usage() {
  cat >&2 <<'EOF'
Usage:
  deploy_portfolio.sh <candidate-compose> <image@sha256:digest> <40-character-build-sha>
  deploy_portfolio.sh --rollback
EOF
  exit 2
}

die() {
  printf 'deploy: %s\n' "$*" >&2
  exit 1
}

compose() {
  docker compose --env-file "$BASE_ENV" --env-file "$RELEASE_ENV" -f "$LIVE_COMPOSE" "$@"
}

release_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$RELEASE_ENV" | tail -n 1
}

smoke() {
  local expected_build_sha="$1"
  compose exec -T portfolio python scripts/smoke_portfolio.py \
    --base-url http://127.0.0.1:8000 \
    --expected-build-sha "$expected_build_sha"
}

activate_current_files() {
  compose pull
  compose up -d --wait --remove-orphans
  smoke "$(release_value RAGLAB_BUILD_SHA)"
}

restore_previous() {
  if [[ -f "$ROLLBACK_DIR/compose.production.yaml" && -f "$ROLLBACK_DIR/release.env" ]]; then
    install -m 0644 "$ROLLBACK_DIR/compose.production.yaml" "$LIVE_COMPOSE"
    install -m 0600 "$ROLLBACK_DIR/release.env" "$RELEASE_ENV"
    activate_current_files
    release_value RAGLAB_PORTFOLIO_IMAGE >"$DEPLOY_ROOT/.last-good-image"
    release_value RAGLAB_BUILD_SHA >"$DEPLOY_ROOT/.last-good-build-sha"
    printf 'deploy: restored %s\n' "$(release_value RAGLAB_PORTFOLIO_IMAGE)"
    return
  fi

  if [[ -f "$LIVE_COMPOSE" && -f "$RELEASE_ENV" ]]; then
    compose down --remove-orphans || true
  fi
  rm -f "$LIVE_COMPOSE" "$RELEASE_ENV"
  printf 'deploy: failed first release removed; no previous release existed\n' >&2
}

mkdir -p "$DEPLOY_ROOT"
command -v docker >/dev/null || die "docker is required"
command -v flock >/dev/null || die "flock is required"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another deployment is running"

[[ -f "$BASE_ENV" ]] || die "$BASE_ENV is missing; complete the server bootstrap first"

if [[ "${1:-}" == "--rollback" ]]; then
  [[ $# -eq 1 ]] || usage
  restore_previous
  exit 0
fi

[[ $# -eq 3 ]] || usage
candidate_compose="$1"
image="$2"
build_sha="$3"

[[ -f "$candidate_compose" ]] || die "candidate Compose file does not exist: $candidate_compose"
grep -Eq '^.+@sha256:[0-9a-f]{64}$' <<<"$image" || die "portfolio image must be pinned by sha256 digest"
grep -Eq '^[0-9a-f]{40}$' <<<"$build_sha" || die "build SHA must be the full 40-character Git SHA"

staged_env="$(mktemp "$DEPLOY_ROOT/.release.env.XXXXXX")"
trap 'rm -f "$staged_env"' EXIT
{
  printf 'RAGLAB_PORTFOLIO_IMAGE=%s\n' "$image"
  printf 'RAGLAB_BUILD_SHA=%s\n' "$build_sha"
} >"$staged_env"
chmod 0600 "$staged_env"

docker compose --env-file "$BASE_ENV" --env-file "$staged_env" \
  -f "$candidate_compose" config --quiet
while IFS= read -r configured_image; do
  grep -Eq '^.+@sha256:[0-9a-f]{64}$' <<<"$configured_image" || \
    die "every production image must be pinned by sha256 digest: $configured_image"
done < <(docker compose --env-file "$BASE_ENV" --env-file "$staged_env" \
  -f "$candidate_compose" config --images)
services="$(docker compose --env-file "$BASE_ENV" --env-file "$staged_env" \
  -f "$candidate_compose" config --services | LC_ALL=C sort)"
[[ "$services" == $'portfolio\nproxy' ]] || die "production Compose must contain only portfolio and proxy services"

mkdir -p "$ROLLBACK_DIR"
if [[ -e "$LIVE_COMPOSE" || -e "$RELEASE_ENV" ]]; then
  [[ -f "$LIVE_COMPOSE" && -f "$RELEASE_ENV" ]] || \
    die "current deployment state is incomplete; refusing to overwrite it"
  install -m 0644 "$LIVE_COMPOSE" "$ROLLBACK_DIR/compose.production.yaml"
  install -m 0600 "$RELEASE_ENV" "$ROLLBACK_DIR/release.env"
else
  rm -f "$ROLLBACK_DIR/compose.production.yaml" "$ROLLBACK_DIR/release.env"
fi

install -m 0644 "$candidate_compose" "$LIVE_COMPOSE"
install -m 0600 "$staged_env" "$RELEASE_ENV"

if activate_current_files; then
  printf '%s\n' "$image" >"$DEPLOY_ROOT/.last-good-image"
  printf '%s\n' "$build_sha" >"$DEPLOY_ROOT/.last-good-build-sha"
  printf 'deploy: activated %s (%s)\n' "$image" "$build_sha"
else
  printf 'deploy: readiness or smoke check failed; restoring previous release\n' >&2
  restore_previous || die "deployment and automatic rollback both failed"
  exit 1
fi
