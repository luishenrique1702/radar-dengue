-- Boletim da semana: municípios com risco alto ou muito alto previsto, com o intervalo de 80%.
-- É a consulta que um painel de vigilância rodaria toda segunda-feira.
SELECT
    p.horizonte                       AS semanas_a_frente,
    p.data_alvo,
    m.municipio,
    m.regiao,
    ROUND(p.casos_previstos)          AS casos_previstos,
    ROUND(p.casos_min)                AS minimo_provavel,
    ROUND(p.casos_max)                AS maximo_provavel,
    ROUND(p.incidencia_100k, 1)       AS incidencia_100k,
    p.nivel_risco
FROM previsoes AS p
JOIN municipios AS m USING (geocode)
WHERE p.nivel_risco IN ('Alto', 'Muito alto')
  AND p.casos_previstos >= 5
ORDER BY p.horizonte, p.incidencia_100k DESC;
