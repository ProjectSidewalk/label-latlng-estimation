-- Modern-truth extraction: post-2021 labels whose stored pano_x/pano_y replay the front-end
-- projection exactly, for scoring the issue-#3 refit against fresh GSV depth.
-- See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3
--
-- Run with psql against sidewalk_prod, scoped to one city via search_path (the companion
-- extract-modern-labels.sh drives it over every current city schema):
--   psql "dbname=sidewalk_prod options=--search_path=sidewalk_seattle,sidewalk_login,public" \
--        -U <user> -p 5434 -f extract-modern-labels.sql > modern-labels-seattle.csv
--
-- Selection notes:
--   time_created >= '2021-01-01': past this cutoff the stored pano_x/pano_y replay the
--     front-end projection at 100.0000% in every checked city (data/pov-inversion-summary.json,
--     pano_x_exact_match_rate_post_cutoff), so the stored pixels are trustworthy anchors for
--     sampling a depth map. Both pano_x and the depth raster are heading-centred, so no frame
--     rotation applies (reports/2026-08-06-depth-coordinate-conventions.md).
--   NOT deleted / NOT tutorial matches the 2021 cleaning. Tutorial labels sit on a fixed
--     legacy pano and are the only post-2021 rows still stamped computation_method = 'depth';
--     every non-tutorial post-2021 row is 'approximation2' (verified across all 54 schemas,
--     2026-08-07), i.e. the deployed estimator's own output.
--   pd.source = 'gsv' and 22-char pano ids: depth exists only for GSV panoramas. The handful
--     of rows with other id shapes (8/32/36/44/64 chars) or infra3d/mapillary sources are
--     unfetchable via the photometa endpoint and are dropped here rather than at fetch time.
--   lp.lat/lng are the DEPLOYED estimator's own output — exported as the ready-made deployed
--     prediction and as the circularity guard's check value, never usable as truth.
--   is_ai: labels submitted by the 'SidewalkAI' user (the submitAiLabelData path); everything
--     else is a human click. 99.9% of vancouver is SidewalkAI, so provenance must travel.
--   zoom stays DOUBLE PRECISION (no ::int): the modern front-end is not constrained to
--     integral zooms, and no downstream computation consumes it as an integer.

COPY (
  SELECT l.label_id,
         lt.label_type,
         lp.lat,
         lp.lng,
         lp.canvas_x,
         lp.canvas_y,
         lp.heading,
         lp.pitch,
         lp.zoom,
         lp.pano_x,
         lp.pano_y,
         lp.computation_method,
         l.pano_id,
         l.time_created,
         (u.username = 'SidewalkAI') AS is_ai,
         pd.width          AS pano_width,
         pd.height         AS pano_height,
         pd.lat            AS pano_lat,
         pd.lng            AS pano_lng,
         pd.camera_heading,
         pd.camera_pitch,
         pd.camera_roll,
         pd.capture_date,
         pd.source::text   AS pano_source
  FROM label_point lp
  JOIN label l                        ON lp.label_id = l.label_id
  JOIN label_type lt                  ON l.label_type_id = lt.label_type_id
  JOIN sidewalk_login.sidewalk_user u ON l.user_id = u.user_id
  JOIN pano_data pd                   ON l.pano_id = pd.pano_id
  WHERE l.time_created >= '2021-01-01'
    AND NOT l.deleted
    AND NOT l.tutorial
    AND lp.pano_x IS NOT NULL
    AND lp.pano_y IS NOT NULL
    AND pd.source = 'gsv'
    AND pd.width IS NOT NULL
    AND LENGTH(l.pano_id) = 22
  ORDER BY l.label_id
) TO STDOUT WITH (FORMAT csv, HEADER);
