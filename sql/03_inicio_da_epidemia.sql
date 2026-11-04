-- Em que semana cada região "entrou em epidemia" a cada ano: a mediana, entre os municípios,
-- da primeira semana com incidência semanal de 50 ou mais casos por 100 mil habitantes.
-- Mostra: CTEs encadeadas, ROW_NUMBER e agregação estatística (MEDIAN).
WITH semanais AS (
    SELECT c.geocode, m.regiao,
           EXTRACT(year FROM c.data_inise) AS ano,
           EXTRACT(week FROM c.data_inise) AS semana,
           c.casos / c.pop * 1e5           AS incidencia
    FROM casos_semanais AS c
    JOIN municipios AS m USING (geocode)
    WHERE m.pop_max >= 20000
),
primeira AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY geocode, ano ORDER BY semana) AS ordem
    FROM semanais
    WHERE incidencia >= 50
)
SELECT ano, regiao,
       COUNT(*)              AS municipios_em_epidemia,
       MEDIAN(semana)        AS semana_mediana_de_inicio
FROM primeira
WHERE ordem = 1 AND ano BETWEEN 2019 AND 2025
GROUP BY ano, regiao
ORDER BY ano, regiao;
