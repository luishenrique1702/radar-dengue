-- Os 5 municípios com maior incidência em 2024 em cada região.
-- Mostra: junção entre tabelas, função de janela RANK com PARTITION BY e QUALIFY.
SELECT
    m.regiao,
    m.municipio,
    SUM(c.casos)                                         AS casos_2024,
    ROUND(SUM(c.casos) / MAX(m.pop_max) * 1e5)           AS incidencia_100k,
    RANK() OVER (PARTITION BY m.regiao
                 ORDER BY SUM(c.casos) / MAX(m.pop_max) DESC) AS posicao
FROM casos_semanais AS c
JOIN municipios AS m USING (geocode)
WHERE EXTRACT(year FROM c.data_inise) = 2024
  AND m.pop_max >= 20000                                 -- evita taxas instáveis de cidades muito pequenas
GROUP BY m.regiao, m.municipio
QUALIFY posicao <= 5
ORDER BY m.regiao, posicao;
