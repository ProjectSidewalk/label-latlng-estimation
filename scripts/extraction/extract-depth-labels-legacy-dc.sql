-- DC variant: the original DC deployment lived in a separate legacy database (schema `sidewalk`,
-- pre-evolution-179 layout), which is why the 2021 Rmd documents a second query for it. This file
-- is that query, near-verbatim, with the shared column order of the modern extraction so all
-- seven CSVs concatenate cleanly.
-- See https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1
--
-- Use this variant only if run-time discovery shows the DC database still has the LEGACY schema
-- (label_point.sv_image_x exists). If DC was migrated to the modern schema, use
-- extract-depth-labels-modern.sql with the appropriate search_path instead.
--
-- Differences from the modern variant, faithful to the 2021 DC query:
--   - computation_method did not exist -> constant 'depth' (the DB predates the depth API's
--     removal; WHERE lat IS NOT NULL AND lng IS NOT NULL selects rows with depth estimates).
--   - tutorial did not exist as a label column -> membership in sidewalk.gsv_onboarding_pano.
--   - INNER JOIN audit_task, as in the original (drops labels with no audit task).
--   - Extras: discovery (2026-08-05, sidewalk_dc on makelab1) confirmed the legacy schema has
--     label.time_created and gsv_data.image_width/image_height, so those extras are real values;
--     current_pano_x/y stay NULL (evolution 179 never ran here — there is no recomputed value).

COPY (
  SELECT label.label_id,
         label_type.label_type,
         lat,
         lng,
         panorama_lat,
         panorama_lng,
         canvas_x,
         canvas_y,
         canvas_width,
         canvas_height,
         heading,
         pitch,
         zoom,
         photographer_heading,
         photographer_pitch,
         sv_image_x,
         sv_image_y,
         label.gsv_panorama_id,
         street_edge_id,
         deleted,
         label.gsv_panorama_id IN (SELECT gsv_panorama_id FROM sidewalk.gsv_onboarding_pano) AS tutorial,
         'depth'               AS computation_method,
         gsv_data.image_width  AS pano_width,
         gsv_data.image_height AS pano_height,
         label.time_created,
         NULL::int             AS current_pano_x,
         NULL::int             AS current_pano_y
  FROM sidewalk.label_point
  INNER JOIN sidewalk.label      ON label_point.label_id = label.label_id
  INNER JOIN sidewalk.label_type ON label.label_type_id = label_type.label_type_id
  INNER JOIN sidewalk.audit_task ON label.audit_task_id = audit_task.audit_task_id
  LEFT JOIN sidewalk.gsv_data    ON label.gsv_panorama_id = gsv_data.gsv_panorama_id
  WHERE lat IS NOT NULL AND lng IS NOT NULL
  ORDER BY label.label_id
) TO STDOUT WITH (FORMAT csv, HEADER);
