#!/bin/sh
set -e

# reports/, logs/, and config/ are bind-mounted from the host (see docker-compose.yml),
# so whatever ownership those host directories happen to have at start time overrides
# the chown baked into the image at build time. If files in them were ever created by
# a different UID (e.g. the pipeline run directly on the host as the host user, instead
# of inside this container), the container's non-root appuser loses write access to them
# — this caused "Permission denied" failures saving reports/evaluations_store.json.
#
# docker-compose.debug.yml runs this same image with `user: "${LOCAL_UID}:${LOCAL_GID}"`,
# starting the container directly as that (non-root) UID instead of root+appuser — in
# that mode there's no privilege to drop and no gosu (setuid) capability, so only run
# the ownership-reconcile-and-drop dance when we actually started as root.
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser /app/reports /app/logs /app/config 2>/dev/null || true
    exec gosu appuser "$@"
else
    exec "$@"
fi
