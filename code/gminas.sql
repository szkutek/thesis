DROP TABLE IF EXISTS Gminas;
DROP TABLE IF EXISTS Voivodships;
DROP TABLE IF EXISTS Powiats;

DROP TABLE IF EXISTS VoivodshipGminas;

DROP TABLE IF EXISTS NeighbouringGminas;
DROP TABLE IF EXISTS WorkMigration;


CREATE TABLE IF NOT EXISTS Gminas (
  gmina_id           INTEGER PRIMARY KEY,
  name               TEXT NOT NULL,
  coord_lat          TEXT,
  coord_long         TEXT,
  population         INTEGER,
  area               REAL,
  population_density REAL
);

CREATE TABLE IF NOT EXISTS Voivodships (
  voivodship_id      INTEGER PRIMARY KEY,
  name               TEXT NOT NULL,
  coord_lat          TEXT,
  coord_long         TEXT,
  population         INTEGER,
  area               REAL,
  population_density REAL
);

CREATE TABLE IF NOT EXISTS Powiats (
  powiat_id          INTEGER PRIMARY KEY,
  name               TEXT NOT NULL,
  coord_lat          TEXT,
  coord_long         TEXT,
  population         INTEGER,
  area               REAL,
  population_density REAL
);

CREATE TABLE IF NOT EXISTS VoivodshipGminas (
  voivodship_id INTEGER,
  gmina_id      INTEGER,
  PRIMARY KEY (voivodship_id, gmina_id),
  FOREIGN KEY (voivodship_id) REFERENCES Voivodships (voivodship_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (gmina_id) REFERENCES Gminas (gmina_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION
);

CREATE TABLE IF NOT EXISTS NeighbouringGminas (
  gmina1_id INTEGER,
  gmina2_id INTEGER,
  PRIMARY KEY (gmina1_id, gmina2_id),
  FOREIGN KEY (gmina1_id) REFERENCES Gminas (gmina_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (gmina2_id) REFERENCES Gminas (gmina_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION
);

CREATE TABLE IF NOT EXISTS WorkMigration (
  home_voivodship_id INTEGER,
  home_gmina_id      INTEGER,
  home_powiat_id     INTEGER,
  work_voivodship_id INTEGER,
  work_powiat_id     INTEGER,
  work_gmina_id      INTEGER,
  migrating          INTEGER,
  FOREIGN KEY (home_voivodship_id) REFERENCES Voivodships (voivodship_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (home_powiat_id) REFERENCES Powiats (powiat_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (home_gmina_id) REFERENCES Gminas (gmina_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (work_voivodship_id) REFERENCES Voivodships (voivodship_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (work_powiat_id) REFERENCES Powiats (powiat_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION,
  FOREIGN KEY (work_gmina_id) REFERENCES Gminas (gmina_id)
    ON DELETE CASCADE
    ON UPDATE NO ACTION
);
