-- Erro do modelo e do baseline no período de teste, por porte do município e horizonte.
-- Mostra: junção entre a tabela de previsões e o cadastro, agregação condicional e razões.
SELECT
    m.porte,
    p.horizonte,
    COUNT(DISTINCT p.geocode)                                    AS municipios,
    ROUND(AVG(ABS(p.casos_reais - p.casos_previstos)), 2)        AS erro_modelo,
    ROUND(AVG(ABS(p.casos_reais - p.baseline)), 2)               AS erro_baseline,
    ROUND(100 * (1 - AVG(ABS(p.casos_reais - p.casos_previstos))
                   / AVG(ABS(p.casos_reais - p.baseline))), 1)   AS reducao_pct
FROM previsoes_teste AS p
JOIN municipios AS m USING (geocode)
GROUP BY m.porte, p.horizonte
ORDER BY p.horizonte, MIN(m.pop_max);
