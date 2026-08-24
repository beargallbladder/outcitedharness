-- hypothetical new consumer copied from an analyst notebook
SELECT COUNT(*) AS n_mentions, COUNT(DISTINCT model) AS n_models
FROM category_mentions_v2
WHERE brand_id = (SELECT id FROM brands WHERE domain = 'examplebrand.com');
