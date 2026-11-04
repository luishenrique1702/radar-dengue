-- Casos e incidência por ano no estado, com variação em relação ao ano anterior.
-- Mostra: agregação, junção, e a função de janela LAG.
WITH pop_estado AS (
    SELECT SUM(pop_max) AS populacao FROM municipios
),
anual AS (
    SELECT EXTRACT(year FROM data_inise) AS ano, SUM(casos) AS casos
    FROM casos_semanais
    WHERE EXTRACT(year FROM data_inise) < 2026      -- 2026 ainda incompleto
    GROUP BY ano
)
SELECT
    ano,
    casos,
    ROUND(casos / p.populacao * 1e5)                                    AS incidencia_100k,
    ROUND(100.0 * (casos - LAG(casos) OVER (ORDER BY ano))
          / LAG(casos) OVER (ORDER BY ano), 1)                          AS variacao_pct
FROM anual, pop_estado AS p
ORDER BY ano;
