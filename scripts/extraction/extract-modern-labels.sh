#!/usr/bin/env bash
# Extract the modern-truth label population (post-2021, all current city schemas) from
# production. See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3
#
# Run ON the database host (e.g. makelab1) or any UW CSE host that can reach the DB, from the
# directory containing the companion extract-modern-labels.sql. Read-only by construction:
# every statement is a SELECT (COPY ... TO STDOUT).
#
#   ./extract-modern-labels.sh                # extract every current city schema
#   ./extract-modern-labels.sh -o my-outdir   # choose the output directory
#
# Unlike extract-depth-labels.sh (whose six-city list reconstructs a fixed 2021 dataset), the
# schema list here is discovered at run time: production adds cities, and the modern-truth set
# wants all of them. A few schemas are excluded with reasons recorded in the metadata file.
#
# Auth: uses your personal postgres role (-U); psql will prompt for a password, or use
# ~/.pgpass / PGPASSWORD. A SELECT-only role is all this needs.

set -euo pipefail

PSQL="${PSQL:-$(command -v psql || echo /usr/pgsql-16/bin/psql)}"
PSQL_USER="${USER}"
PORT=5434
DB=sidewalk_prod
OUTDIR="modern-labels-extraction-$(date +%Y%m%d)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -U) PSQL_USER="$2"; shift 2 ;;
    -p) PORT="$2"; shift 2 ;;
    -d) DB="$2"; shift 2 ;;
    -o) OUTDIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
SQL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Excluded schemas, with reasons (recorded in the metadata file below):
#   sidewalk_la_piedad_old         superseded deployment (sidewalk_la_piedad is current)
#   sidewalk_la_piedad_old_backup  pre-evolution-179 backup; has no pano_data table at all
#   sidewalk_richmond              Mapillary imagery only - no GSV depth exists
#   sidewalk_winterthur_infra3d    infra3d imagery; zero post-2021 rows besides
#   sidewalk_zurich_infra3d        infra3d imagery - no GSV depth exists
#   sidewalk_validation            validation-study schema, not a city deployment
EXCLUDE=(sidewalk_la_piedad_old sidewalk_la_piedad_old_backup sidewalk_richmond
         sidewalk_winterthur_infra3d sidewalk_zurich_infra3d sidewalk_validation)

psql_city() {  # psql_city <schema> <args...>
  local schema="$1"; shift
  "$PSQL" "dbname=$DB options=--search_path=$schema,sidewalk_login,public" \
       -U "$PSQL_USER" -p "$PORT" -v ON_ERROR_STOP=1 -q "$@"
}

SCHEMAS=$("$PSQL" -U "$PSQL_USER" -p "$PORT" -d "$DB" -Atc \
  "SELECT schema_name FROM information_schema.schemata
   WHERE schema_name LIKE 'sidewalk\\_%' AND schema_name <> 'sidewalk_login' ORDER BY 1;")

mkdir -p "$OUTDIR"
META="$OUTDIR/extraction-metadata.txt"
{
  echo "Extraction date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Host: $(hostname)"
  echo "Database: $DB port $PORT, role $PSQL_USER"
  echo "Server: $("$PSQL" -U "$PSQL_USER" -p "$PORT" -d "$DB" -Atc 'SELECT version();')"
  echo "Query: extract-modern-labels.sql, schema list discovered at run time"
  echo "Excluded schemas: ${EXCLUDE[*]}"
  echo
} > "$META"

for schema in $SCHEMAS; do
  skip=0
  for ex in "${EXCLUDE[@]}"; do [[ "$schema" == "$ex" ]] && skip=1; done
  if [[ $skip -eq 1 ]]; then
    echo "-- $schema: excluded (see list above) --" | tee -a "$META"
    continue
  fi
  city="${schema#sidewalk_}"
  csv="$OUTDIR/modern-labels-$city.csv"
  echo "-- $city ($DB/$schema) --" | tee -a "$META"
  psql_city "$schema" -f "$SQL_DIR/extract-modern-labels.sql" > "$csv"
  rows=$(( $(wc -l < "$csv") - 1 ))
  echo "csv rows: $rows" | tee -a "$META"
  if [[ $rows -eq 0 ]]; then
    rm "$csv"
    echo "(empty; file removed)" | tee -a "$META"
    continue
  fi
  # Population census for the sampling frame: extracted vs total post-2021 rows, and the
  # AI/human split. Everything the extraction filters away is visible here.
  psql_city "$schema" -c "
    SELECT COUNT(*)                                                  AS post2021_rows,
           COUNT(*) FILTER (WHERE pd.pano_id IS NOT NULL
                              AND pd.source = 'gsv'
                              AND pd.width IS NOT NULL
                              AND LENGTH(l.pano_id) = 22)            AS extracted,
           COUNT(*) FILTER (WHERE u.username = 'SidewalkAI')         AS ai_rows,
           MIN(l.time_created)::date                                 AS first_label,
           MAX(l.time_created)::date                                 AS last_label
    FROM label_point lp
    JOIN label l                        ON lp.label_id = l.label_id
    JOIN sidewalk_login.sidewalk_user u ON l.user_id = u.user_id
    LEFT JOIN pano_data pd              ON l.pano_id = pd.pano_id
    WHERE l.time_created >= '2021-01-01' AND NOT l.deleted AND NOT l.tutorial
      AND lp.pano_x IS NOT NULL AND lp.pano_y IS NOT NULL;" | tee -a "$META"
  echo | tee -a "$META"
done

gzip -9 -f "$OUTDIR"/modern-labels-*.csv
echo "Done. Output in $OUTDIR:"
ls -l "$OUTDIR"
