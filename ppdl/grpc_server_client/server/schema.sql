CREATE SCHEMA IF NOT EXISTS ppdl;

CREATE TABLE IF NOT EXISTS ppdl.job (
    id bigserial PRIMARY KEY,
    created_at timestamp NOT NULL DEFAULT now()
    );

CREATE TABLE IF NOT EXISTS ppdl.cycle (
    id bigserial PRIMARY KEY,
    job_id bigint NOT NULL REFERENCES ppdl.job(id) ON DELETE CASCADE ON UPDATE CASCADE,
    starts_at timestamp NOT NULL,
    finishes_at timestamp NOT NULL
    );
CREATE INDEX c_job_id_idx ON ppdl.cycle(job_id);

CREATE TABLE IF NOT EXISTS ppdl.parameters_upload (
    id bigserial PRIMARY KEY,
    cycle_id bigint NOT NULL REFERENCES ppdl.cycle(id) ON DELETE CASCADE ON UPDATE CASCADE,
    user_id text NOT NULL,
    created_at timestamp NOT NULL DEFAULT now()
    );
CREATE INDEX pu_cycle_id_idx ON ppdl.parameters_upload(cycle_id);

CREATE TABLE IF NOT EXISTS ppdl.parameter (
    upload_id bigint NOT NULL REFERENCES ppdl.parameters_upload(id) ON DELETE CASCADE ON UPDATE CASCADE,
    dimension int NOT NULL,
    value float8 NOT NULL
    );
CREATE INDEX p_upload_id_idx ON ppdl.parameter(upload_id);
CREATE INDEX p_dimension_idx ON ppdl.parameter(dimension);
