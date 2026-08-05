-- Reconstruction of the 2021 label lat/lng estimation dataset from today's production schema.
-- See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1
--
-- Run with psql against sidewalk_prod, scoped to one city via search_path, e.g.:
--   psql "dbname=sidewalk_prod options=--search_path=sidewalk_seattle,sidewalk_login,public" \
--        -U <user> -p 5434 -f extract-depth-labels-modern.sql > labels-seattle-latlng.csv
--
-- Column mapping, 2021 query -> today (schema drift since the original extraction):
--   sv_image_x, sv_image_y      -> old_label_metadata.old_pano_x/old_pano_y. Evolution 179
--                                  overwrote label_point.sv_image_x/y in place under a new
--                                  convention and renamed them pano_x/pano_y, but first copied
--                                  the originals into old_label_metadata for every label with
--                                  time_created < v7.12.2 (2023-03-29). All depth labels
--                                  (pre-Nov-2020) are inside that window.
--   panorama_lat, panorama_lng  -> old_label_metadata.old_pano_lat/old_pano_lng (dropped from
--                                  label by evolution 179).
--   photographer_heading/pitch  -> old_label_metadata.old_camera_heading/old_camera_pitch
--                                  (dropped from label by evolution 179; the pano_data analogues
--                                  are a one-row-per-pano backfill, NOT per-label values).
--   canvas_width, canvas_height -> literals 720/480. Dropped from label_point by evolution 177,
--                                  whose !Downs confirms those were the uniform defaults.
--   gsv_panorama_id             -> label.pano_id (renamed by evolution 298).
--   zoom                        -> cast ::int (column became DOUBLE PRECISION; depth-era values
--                                  are integral 1/2/3 and the 2021 CSVs had integers).
--   lat, lng                    -> label_point.lat/lng, unchanged: evolution 98's recompute
--                                  explicitly excluded computation_method = 'depth'.
--
-- WHERE computation_method = 'depth': the 2021 extraction was unfiltered, but every non-depth
-- row was discarded by the Rmd's cleaning filters, and the depth population is closed (the depth
-- API died Nov 2020), so this restriction loses nothing and keeps the files small.
--
-- Columns after computation_method are extras absent from the 2021 CSVs (readr and pandas both
-- tolerate appended columns): pano dimensions for the height-normalization question
-- (SidewalkWebpage#4765; pano resolution never changes for a given pano id), time_created for
-- provenance, and the post-evolution-179 recomputed pano_x/pano_y for cross-checking.

COPY (
  SELECT l.label_id,
         lt.label_type,
         lp.lat,
         lp.lng,
         olm.old_pano_lat       AS panorama_lat,
         olm.old_pano_lng       AS panorama_lng,
         lp.canvas_x,
         lp.canvas_y,
         720                    AS canvas_width,
         480                    AS canvas_height,
         lp.heading,
         lp.pitch,
         lp.zoom::int           AS zoom,
         olm.old_camera_heading AS photographer_heading,
         olm.old_camera_pitch   AS photographer_pitch,
         olm.old_pano_x         AS sv_image_x,
         olm.old_pano_y         AS sv_image_y,
         l.pano_id              AS gsv_panorama_id,
         l.street_edge_id,
         l.deleted,
         l.tutorial,
         lp.computation_method,
         pd.width               AS pano_width,
         pd.height              AS pano_height,
         l.time_created,
         lp.pano_x              AS current_pano_x,
         lp.pano_y              AS current_pano_y
  FROM label_point lp
  JOIN label l                     ON lp.label_id = l.label_id
  JOIN label_type lt               ON l.label_type_id = lt.label_type_id
  LEFT JOIN old_label_metadata olm ON l.label_id = olm.label_id
  LEFT JOIN pano_data pd           ON l.pano_id = pd.pano_id
  WHERE lp.computation_method = 'depth'
  ORDER BY l.label_id
) TO STDOUT WITH (FORMAT csv, HEADER);
