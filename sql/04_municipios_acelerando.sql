-- Municípios em aceleração na semana mais recente: casos das últimas 4 semanas comparados
-- com as 4 anteriores, usando a estimativa corrigida pelo atraso de notificação.
-- Mostra: médias móveis com janelas deslizantes (ROWS BETWEEN) e filtragem do último período.
WITH base AS (
    SELECT c.geocode, m.municipio, c.data_inise,
           GREATEST(c.casos, c.casos_est) AS casos,
           SUM(GREATEST(c.casos, c.casos_est)) OVER w4 AS ult_4,
           SUM(GREATEST(c.casos, c.casos_est)) OVER w8 AS ult_8
    FROM casos_semanais AS c
    JOIN municipios AS m USING (geocode)
    WINDOW w4 AS (PARTITION BY c.geocode ORDER BY c.data_inise ROWS BETWEEN 3 PRECEDING AND CURRENT ROW),
           w8 AS (PARTITION BY c.geocode ORDER BY c.data_inise ROWS BETWEEN 7 PRECEDING AND CURRENT ROW)
)
SELECT municipio,
       ult_4                                     AS casos_ultimas_4_semanas,
       ult_8 - ult_4                             AS casos_4_semanas_anteriores,
       ROUND(100.0 * (ult_4 - (ult_8 - ult_4)) / NULLIF(ult_8 - ult_4, 0), 1) AS variacao_pct
FROM base
WHERE data_inise = (SELECT MAX(data_inise) FROM casos_semanais)
  AND ult_8 - ult_4 >= 20                        -- base mínima para a variação fazer sentido
ORDER BY variacao_pct DESC
LIMIT 15;
