-- UUIDv7 Function 1: Simple implementation using overlay and bit manipulation
CREATE OR REPLACE FUNCTION uuid_generate_v7()
RETURNS uuid
AS $$
BEGIN
  -- use random v4 uuid as starting point (which has the same variant we need)
  -- then overlay timestamp
  -- then set version 7 by flipping the 2 and 1 bit in the version 4 string
  RETURN encode(
    set_bit(
      set_bit(
        overlay(uuid_send(gen_random_uuid())
                placing substring(int8send(floor(extract(epoch from clock_timestamp()) * 1000)::bigint) from 3)
                from 1 for 6
        ),
        52, 1
      ),
      53, 1
    ),
    'hex')::uuid;
END
$$
LANGUAGE plpgsql
VOLATILE;

-- UUIDv7 Function 2: SQL-based implementation with similar approach
CREATE OR REPLACE FUNCTION uuidv7() 
RETURNS uuid
AS $$
  -- Replace the first 48 bits of a uuidv4 with the current
  -- number of milliseconds since 1970-01-01 UTC
  -- and set the "ver" field to 7 by setting additional bits
  SELECT encode(
    set_bit(
      set_bit(
        overlay(uuid_send(gen_random_uuid()) placing
          substring(int8send((extract(epoch from clock_timestamp())*1000)::bigint) from 3)
          from 1 for 6),
        52, 1),
      53, 1), 'hex')::uuid;
$$ LANGUAGE sql VOLATILE;

-- UUIDv7 Function 3: Sub-millisecond precision version
-- Version with the "rand_a" field containing sub-milliseconds (method 3 of the spec)
-- The uuid is the concatenation of:
-- - 6 bytes with the current Unix timestamp (milliseconds since 1970-01-01 UTC)
-- - 2 bytes with 4 bits for version and 12 bits for fractional milliseconds
-- - 8 bytes of randomness
CREATE OR REPLACE FUNCTION uuidv7_sub_ms() 
RETURNS uuid
AS $$
SELECT encode(
  substring(int8send(floor(t_ms)::int8) from 3) ||
  int2send((7<<12)::int2 | ((t_ms-floor(t_ms))*4096)::int2) ||
  substring(uuid_send(gen_random_uuid()) from 9 for 8)
, 'hex')::uuid
FROM (SELECT extract(epoch from clock_timestamp())*1000 as t_ms) s
$$ LANGUAGE sql VOLATILE;

-- Create benchmark tables
CREATE TABLE IF NOT EXISTS uuid_benchmark (
  id SERIAL PRIMARY KEY,
  function_name TEXT NOT NULL,
  uuid_value UUID NOT NULL,
  generation_time_ns BIGINT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_uuid_value ON uuid_benchmark(uuid_value);
CREATE INDEX idx_function_name ON uuid_benchmark(function_name);

-- Create function to measure generation time
CREATE OR REPLACE FUNCTION benchmark_uuid_function(
  func_name TEXT,
  iterations INTEGER DEFAULT 1000
) RETURNS TABLE (
  function_name TEXT,
  total_iterations INTEGER,
  avg_time_ns NUMERIC,
  min_time_ns BIGINT,
  max_time_ns BIGINT,
  std_dev_ns NUMERIC
) AS $$
DECLARE
  start_time TIMESTAMP;
  end_time TIMESTAMP;
  elapsed_ns BIGINT;
  uuid_val UUID;
BEGIN
  -- Clear previous results for this function
  DELETE FROM uuid_benchmark WHERE uuid_benchmark.function_name = func_name;
  
  -- Run benchmarks
  FOR i IN 1..iterations LOOP
    start_time := clock_timestamp();
    
    CASE func_name
      WHEN 'uuid_generate_v7' THEN
        uuid_val := uuid_generate_v7();
      WHEN 'uuidv7' THEN
        uuid_val := uuidv7();
      WHEN 'uuidv7_sub_ms' THEN
        uuid_val := uuidv7_sub_ms();
    END CASE;
    
    end_time := clock_timestamp();
    elapsed_ns := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000000000;
    
    INSERT INTO uuid_benchmark (function_name, uuid_value, generation_time_ns)
    VALUES (func_name, uuid_val, elapsed_ns);
  END LOOP;
  
  -- Return statistics
  RETURN QUERY
  SELECT 
    b.function_name,
    COUNT(*)::INTEGER as total_iterations,
    AVG(generation_time_ns) as avg_time_ns,
    MIN(generation_time_ns) as min_time_ns,
    MAX(generation_time_ns) as max_time_ns,
    STDDEV(generation_time_ns) as std_dev_ns
  FROM uuid_benchmark b
  WHERE b.function_name = func_name
  GROUP BY b.function_name;
END;
$$ LANGUAGE plpgsql;