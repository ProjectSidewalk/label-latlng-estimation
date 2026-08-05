#!/usr/bin/env bash
# Reconstruct the 2021 label lat/lng estimation dataset from production.
# See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1
#
# Run ON the database host (e.g. makelab1) or any UW CSE host that can reach the DB, from the
# directory containing the companion .sql files. Read-only by construction: every statement is a
# SELECT (COPY ... TO STDOUT).
#
#   ./extract-depth-labels.sh                 # extract the six modern cities
#   ./extract-depth-labels.sh --discover      # no extraction; report DBs + DC schema fingerprint
#   ./extract-depth-labels.sh --dc-db sidewalk_dc --dc-port 5434   # also extract DC
#
# Auth: uses your personal postgres role (-U); psql will prompt for a password, or use ~/.pgpass /
# PGPASSWORD. A SELECT-only role is all this needs.

set -euo pipefail

PSQL="${PSQL:-$(command -v psql || echo /usr/pgsql-16/bin/psql)}"
PSQL_USER="${USER}"
PORT=5434
DB=sidewalk_prod
OUTDIR="latlng-extraction-$(date +%Y%m%d)"
CITIES=(seattle newberg columbus spgg cdmx pittsburgh)
DISCOVER=0
DC_DB=""
DC_PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -U) PSQL_USER="$2"; shift 2 ;;
    -p) PORT="$2"; shift 2 ;;
    -d) DB="$2"; shift 2 ;;
    -o) OUTDIR="$2"; shift 2 ;;
    --discover) DISCOVER=1; shift ;;
    --dc-db) DC_DB="$2"; shift 2 ;;
    --dc-port) DC_PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
DC_PORT="${DC_PORT:-$PORT}"
SQL_DIR="$(cd "$(dirname "$0")" && pwd)"

psql_city() {  # psql_city <schema> <db> <port> <args...>
  local schema="$1" db="$2" port="$3"; shift 3
  "$PSQL" "dbname=$db options=--search_path=$schema,sidewalk_login,public" \
       -U "$PSQL_USER" -p "$port" -v ON_ERROR_STOP=1 -q "$@"
}

if [[ $DISCOVER -eq 1 ]]; then
  echo "== Databases on port $PORT:"
  "$PSQL" -U "$PSQL_USER" -p "$PORT" -d "$DB" -Atc \
    "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY 1;"
  echo
  echo "== Schemas in $DB:"
  "$PSQL" -U "$PSQL_USER" -p "$PORT" -d "$DB" -Atc \
    "SELECT schema_name FROM information_schema.schemata
     WHERE schema_name NOT IN ('information_schema','pg_catalog','pg_toast') ORDER BY 1;"
  if [[ -n "$DC_DB" ]]; then
    echo
    echo "== DC candidate $DC_DB (port $DC_PORT) label_point columns per schema:"
    "$PSQL" -U "$PSQL_USER" -p "$DC_PORT" -d "$DC_DB" -Atc \
      "SELECT table_schema || '.' || column_name FROM information_schema.columns
       WHERE table_name = 'label_point' ORDER BY 1;"
    echo
    echo "== DC candidate: does sidewalk.label have time_created; does gsv_data have image dims?"
    "$PSQL" -U "$PSQL_USER" -p "$DC_PORT" -d "$DC_DB" -Atc \
      "SELECT table_schema || '.' || table_name || '.' || column_name
       FROM information_schema.columns
       WHERE (table_name = 'label' AND column_name = 'time_created')
          OR (table_name IN ('gsv_data','pano_data') AND column_name IN
              ('image_width','image_height','width','height')) ORDER BY 1;"
  fi
  exit 0
fi

mkdir -p "$OUTDIR"
META="$OUTDIR/extraction-metadata.txt"
{
  echo "Extraction date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Host: $(hostname)"
  echo "Database: $DB port $PORT, role $PSQL_USER"
  echo "Server: $("$PSQL" -U "$PSQL_USER" -p "$PORT" -d "$DB" -Atc 'SELECT version();')"
  echo "Query: extract-depth-labels-modern.sql (six cities)$( [[ -n "$DC_DB" ]] && echo \
       " + DC from $DC_DB:$DC_PORT" )"
  echo
} > "$META"

extract_one() {  # extract_one <city> <schema> <db> <port> <sqlfile>
  local city="$1" schema="$2" db="$3" port="$4" sqlfile="$5"
  local csv="$OUTDIR/labels-$city-latlng.csv"
  echo "-- $city ($db/$schema) --" | tee -a "$META"
  psql_city "$schema" "$db" "$port" -f "$sqlfile" > "$csv"
  local rows=$(( $(wc -l < "$csv") - 1 ))
  echo "csv rows: $rows" | tee -a "$META"

  if [[ "$sqlfile" == *modern* ]]; then
    psql_city "$schema" "$db" "$port" -c "
      SELECT COUNT(*)                                     AS depth_rows,
             COUNT(*) FILTER (WHERE olm.label_id IS NULL) AS missing_old_label_metadata,
             COUNT(*) FILTER (WHERE pd.width IS NULL)     AS null_pano_width,
             MIN(l.time_created)                          AS first_label,
             MAX(l.time_created)                          AS last_label
      FROM label_point lp
      JOIN label l ON lp.label_id = l.label_id
      LEFT JOIN old_label_metadata olm ON l.label_id = olm.label_id
      LEFT JOIN pano_data pd           ON l.pano_id = pd.pano_id
      WHERE lp.computation_method = 'depth';" | tee -a "$META"
    psql_city "$schema" "$db" "$port" -c "
      SELECT lp.zoom::int AS zoom, COUNT(*)
      FROM label_point lp WHERE lp.computation_method = 'depth'
      GROUP BY 1 ORDER BY 1;" | tee -a "$META"
  fi
  echo | tee -a "$META"
}

for city in "${CITIES[@]}"; do
  extract_one "$city" "sidewalk_$city" "$DB" "$PORT" "$SQL_DIR/extract-depth-labels-modern.sql"
done

if [[ -n "$DC_DB" ]]; then
  legacy=$("$PSQL" -U "$PSQL_USER" -p "$DC_PORT" -d "$DC_DB" -Atc \
    "SELECT COUNT(*) FROM information_schema.columns
     WHERE table_name = 'label_point' AND column_name = 'sv_image_x';")
  if [[ "$legacy" -ge 1 ]]; then
    echo "DC: legacy (pre-179) schema detected; using legacy query" | tee -a "$META"
    extract_one dc sidewalk "$DC_DB" "$DC_PORT" "$SQL_DIR/extract-depth-labels-legacy-dc.sql"
  else
    echo "DC: modern schema detected; using modern query" | tee -a "$META"
    extract_one dc sidewalk_dc "$DC_DB" "$DC_PORT" "$SQL_DIR/extract-depth-labels-modern.sql"
  fi
else
  echo "DC skipped (no --dc-db given; run --discover to locate it)" | tee -a "$META"
fi

gzip -9 -f "$OUTDIR"/labels-*-latlng.csv
echo "Done. Output in $OUTDIR:"
ls -l "$OUTDIR"
