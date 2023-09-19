### SQL queries to explore covid death dataset ###

SELECT *
  FROM CovidDeath

-- Converting datatype:
ALTER TABLE CovidDeath
ALTER COLUMN total_deaths float;
ALTER TABLE CovidDeath
ALTER COLUMN total_cases float;


-- Highest Rate of Infection:
SELECT location, MAX(total_cases) AS MaxInfected, population, MAX((total_cases/population))*100 AS max_percent_infected
FROM CovidDeath
GROUP BY location, population
ORDER BY 4 DESC

-- Total Death by Continents:
SELECT location, MAX(total_deaths) as death_count
FROM CovidDeath
WHERE continent is null and location NOT LIKE '%income%'
Group by location
ORDER BY 2 DESC

-- A Look at Vaccination Counts(using cte):
With VacvsPop (Continent, Location, Date, Population, New_vaccinations, Total_vaccinated) AS
(
SELECT cd.continent, cd.location, cd.date, cd.population, cv.new_vaccinations,
   SUM(CONVERT(float, cv.new_vaccinations)) OVER (Partition By cd.location Order By cd.location, cd.date) AS total_vaccinated
FROM CovidDeath AS cd
JOIN CovidVaccination AS cv
  On cd.location=cv.location AND cd.date=cv.date
WHERE cd.continent is not null
)
SELECT *, (Total_vaccinated/Population)*100
FROM VacvsPop

-- Creating view for later use:
CREATE VIEW PercentVaccinated AS
SELECT cd.continent, cd.location, cd.date, cd.population, cv.new_vaccinations,
   SUM(CONVERT(float, cv.new_vaccinations)) OVER (Partition By cd.location Order By cd.location, cd.date) AS total_vaccinated
FROM CovidDeath AS cd
JOIN CovidVaccination AS cv
  On cd.location=cv.location AND cd.date=cv.date
WHERE cd.continent is not null
