This is the container to be run as the master of a node pool
You must import the `schema.sql` file into your Postgres DB for testing. 
Try `cat schema.sql | psql -U postgres` (or with other options for your database).

## Docker
To build: `docker build -t ppdl-server .`
To run: `docker run -it -p 1453:1453/tcp ppdl-server`
