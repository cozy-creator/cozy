-- Hub release 2 (xs-045 U11, hard cut): every release lane names a checkpoint with a repo file
-- (U12 repointed them), so the old tensorfs_* rows go. An unreleased old checkpoint has no
-- reader left and is dropped; GC then reclaims its objects with repos/ and manifests/. A release
-- lane still on an old checkpoint refuses the migration: it would vanish from the catalog.
DO $$
DECLARE
    lanes text;
BEGIN
    SELECT string_agg(r.org || '/' || r.name || '@' || l.release || ' ' || l.lane, ', ' ORDER BY r.org, r.name, l.release, l.lane)
      INTO lanes
      FROM hub_v2.model_lanes l JOIN hub_v2.repos r ON r.id = l.repo_id
     WHERE l.v1 AND l.release <> '';
    IF lanes IS NOT NULL THEN
        RAISE EXCEPTION 'release lanes still name old checkpoints; migrate them first (U12): %', lanes;
    END IF;
END $$;

DELETE FROM hub_v2.model_lanes WHERE v1;
DELETE FROM hub_v2.model_repos m
 WHERE m.etag = '' AND NOT EXISTS (SELECT 1 FROM hub_v2.model_lanes l WHERE l.repo_id = m.repo_id);
ALTER TABLE hub_v2.model_lanes DROP COLUMN v1;

DROP TABLE hub_v2.tensorfs_releases;
DROP TABLE hub_v2.tensorfs_checkpoint_objects;
DROP TABLE hub_v2.tensorfs_checkpoint_encodings;
DROP TABLE hub_v2.tensorfs_checkpoint_components;
DROP TABLE hub_v2.tensorfs_checkpoints;
DROP TABLE hub_v2.tensorfs_repos;
