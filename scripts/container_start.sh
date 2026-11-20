#!/bin/sh
set -eu
# Fixed single-instance Render disk contract. Fail before creating ephemeral uploads.
if [ "${RUNTIME_PROFILE:-local}" = "cloud" ]; then
    mountpoint -q /var/data || { echo 'persistent upload mount required' >&2; exit 1; }
    [ "${RUNTIME_UPLOAD_DIR:-}" = "/var/data/uploads" ] || { echo 'invalid upload configuration' >&2; exit 1; }
    [ ! -L /var/data/uploads ] || { echo 'invalid upload directory' >&2; exit 1; }
    mkdir -p /var/data/uploads
fi
exec docsqa-api
