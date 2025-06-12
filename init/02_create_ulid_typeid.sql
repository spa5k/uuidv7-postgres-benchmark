-- ULID Implementation
-- Based on spec: https://github.com/ulid/spec
-- 48-bit timestamp + 80-bit randomness

CREATE OR REPLACE FUNCTION ulid_generate()
RETURNS text
AS $$
DECLARE
    -- Crockford's Base32 alphabet (case-insensitive, no ambiguous characters)
    alphabet text := '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
    timestamp_ms bigint;
    ulid_text text := '';
    temp_val bigint;
    i integer;
    random_bytes bytea;
BEGIN
    -- Get current timestamp in milliseconds
    timestamp_ms := floor(EXTRACT(EPOCH FROM clock_timestamp()) * 1000);
    
    -- Encode timestamp (48 bits / 6 bytes) in 10 characters
    temp_val := timestamp_ms;
    FOR i IN 1..10 LOOP
        ulid_text := substring(alphabet FROM (temp_val % 32) + 1 FOR 1) || ulid_text;
        temp_val := temp_val / 32;
    END LOOP;
    
    -- Generate 10 random bytes (80 bits)
    random_bytes := gen_random_bytes(10);
    
    -- Encode random part (80 bits) in 16 characters  
    FOR i IN 0..9 LOOP
        temp_val := get_byte(random_bytes, i);
        ulid_text := ulid_text || substring(alphabet FROM (temp_val % 32) + 1 FOR 1);
        temp_val := temp_val / 32;
        IF temp_val > 0 THEN
            ulid_text := ulid_text || substring(alphabet FROM (temp_val % 32) + 1 FOR 1);
        ELSE
            ulid_text := ulid_text || substring(alphabet FROM 1 FOR 1);
        END IF;
    END LOOP;
    
    RETURN left(ulid_text, 26); -- ULID is always 26 characters
END;
$$ LANGUAGE plpgsql VOLATILE;

-- Optimized ULID implementation using direct bit manipulation
CREATE OR REPLACE FUNCTION ulid_generate_optimized()
RETURNS text
AS $$
DECLARE
    alphabet text := '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
    timestamp_ms bigint;
    random_bytes bytea;
    combined_bytes bytea;
    result text := '';
    byte_val integer;
    carry integer := 0;
    i integer;
BEGIN
    -- Get timestamp
    timestamp_ms := floor(EXTRACT(EPOCH FROM clock_timestamp()) * 1000);
    
    -- Convert timestamp to 6 bytes (big-endian)
    combined_bytes := substring(int8send(timestamp_ms) FROM 3 FOR 6);
    
    -- Add 10 random bytes
    random_bytes := gen_random_bytes(10);
    combined_bytes := combined_bytes || random_bytes;
    
    -- Encode all 16 bytes as base32 (26 characters)
    FOR i IN 15 DOWNTO 0 LOOP
        byte_val := get_byte(combined_bytes, i) + carry * 256;
        result := substring(alphabet FROM (byte_val % 32) + 1 FOR 1) || result;
        carry := byte_val / 32;
    END LOOP;
    
    -- Handle any remaining carry
    WHILE carry > 0 LOOP
        result := substring(alphabet FROM (carry % 32) + 1 FOR 1) || result;
        carry := carry / 32;
    END LOOP;
    
    -- Ensure exactly 26 characters (pad if necessary)
    RETURN lpad(result, 26, '0');
END;
$$ LANGUAGE plpgsql VOLATILE;

-- TypeID Implementation
-- Based on: https://github.com/jetify-com/typeid-sql

-- Create TypeID composite type
DO $$ BEGIN
    CREATE TYPE typeid AS (
        prefix text,
        uuid uuid
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- TypeID validation function
CREATE OR REPLACE FUNCTION typeid_validate_prefix(prefix text)
RETURNS boolean
AS $$
BEGIN
    RETURN prefix ~ '^[a-z]{0,63}$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Generate TypeID as composite type
CREATE OR REPLACE FUNCTION typeid_generate(prefix text DEFAULT '')
RETURNS typeid
AS $$
DECLARE
    result typeid;
BEGIN
    -- Validate prefix
    IF NOT typeid_validate_prefix(prefix) THEN
        RAISE EXCEPTION 'TypeID prefix must be lowercase letters only, max 63 chars';
    END IF;
    
    -- Generate result
    result.prefix := prefix;
    result.uuid := uuid_generate_v7(); -- Using our UUIDv7 implementation
    
    RETURN result;
END;
$$ LANGUAGE plpgsql VOLATILE;

-- Base32 encoding for TypeID (Crockford's encoding)
CREATE OR REPLACE FUNCTION base32_encode_uuid(input_uuid uuid)
RETURNS text
AS $$
DECLARE
    alphabet text := '0123456789abcdefghjkmnpqrstvwxyz';
    uuid_bytes bytea;
    result text := '';
    temp_val numeric := 0;
    i integer;
    byte_val integer;
BEGIN
    -- Convert UUID to bytes
    uuid_bytes := uuid_send(input_uuid);
    
    -- Convert bytes to base32
    FOR i IN 0..15 LOOP
        temp_val := temp_val * 256 + get_byte(uuid_bytes, i);
    END LOOP;
    
    -- Extract base32 characters
    WHILE temp_val > 0 LOOP
        result := substring(alphabet FROM (temp_val % 32)::integer + 1 FOR 1) || result;
        temp_val := floor(temp_val / 32);
    END LOOP;
    
    -- Pad to 26 characters
    RETURN lpad(result, 26, '0');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Generate TypeID as text
CREATE OR REPLACE FUNCTION typeid_generate_text(prefix text DEFAULT '')
RETURNS text
AS $$
DECLARE
    uuid_part uuid;
    encoded_uuid text;
BEGIN
    -- Validate prefix
    IF NOT typeid_validate_prefix(prefix) THEN
        RAISE EXCEPTION 'TypeID prefix must be lowercase letters only, max 63 chars';
    END IF;
    
    -- Generate UUID
    uuid_part := uuid_generate_v7();
    
    -- Encode UUID as base32
    encoded_uuid := base32_encode_uuid(uuid_part);
    
    -- Return formatted TypeID
    IF prefix = '' THEN
        RETURN encoded_uuid;
    ELSE
        RETURN prefix || '_' || encoded_uuid;
    END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

-- Create enhanced benchmark table with new ID types
CREATE TABLE IF NOT EXISTS extended_benchmark (
    id SERIAL PRIMARY KEY,
    function_name TEXT NOT NULL,
    id_value TEXT NOT NULL,  -- Store all IDs as text for comparison
    id_binary BYTEA,         -- Store binary representation where applicable
    generation_time_ns BIGINT NOT NULL,
    storage_size_bytes INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_extended_benchmark_function ON extended_benchmark(function_name);
CREATE INDEX IF NOT EXISTS idx_extended_benchmark_time ON extended_benchmark(generation_time_ns);

-- Function to calculate storage size for different ID types
CREATE OR REPLACE FUNCTION calculate_storage_size(
    id_type TEXT,
    id_value TEXT,
    id_binary BYTEA DEFAULT NULL
) RETURNS INTEGER AS $$
BEGIN
    CASE id_type
        WHEN 'uuid_generate_v7', 'uuidv7', 'uuidv7_sub_ms' THEN
            RETURN 16; -- UUID is always 16 bytes
        WHEN 'ulid_generate', 'ulid_generate_optimized' THEN
            RETURN length(id_value); -- ULID as text (26 chars)
        WHEN 'typeid_generate' THEN
            RETURN 16 + length(split_part(id_value, '_', 1)) + 1; -- UUID + prefix + separator
        WHEN 'typeid_generate_text' THEN
            RETURN length(id_value); -- Full text length
        ELSE
            RETURN length(id_value); -- Default to text length
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Enhanced benchmark function for all ID types
CREATE OR REPLACE FUNCTION benchmark_id_function(
    func_name TEXT,
    iterations INTEGER DEFAULT 1000,
    prefix TEXT DEFAULT 'test'
) RETURNS TABLE (
    function_name TEXT,
    total_iterations INTEGER,
    avg_time_ns NUMERIC,
    min_time_ns BIGINT,
    max_time_ns BIGINT,
    std_dev_ns NUMERIC,
    avg_storage_bytes NUMERIC
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    elapsed_ns BIGINT;
    id_val TEXT;
    id_bin BYTEA;
    storage_size INTEGER;
BEGIN
    -- Clear previous results for this function
    DELETE FROM extended_benchmark WHERE extended_benchmark.function_name = func_name;
    
    -- Run benchmarks
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        
        CASE func_name
            WHEN 'uuid_generate_v7' THEN
                id_val := uuid_generate_v7()::text;
                id_bin := uuid_send(id_val::uuid);
            WHEN 'uuidv7' THEN
                id_val := uuidv7()::text;
                id_bin := uuid_send(id_val::uuid);
            WHEN 'uuidv7_sub_ms' THEN
                id_val := uuidv7_sub_ms()::text;
                id_bin := uuid_send(id_val::uuid);
            WHEN 'ulid_generate' THEN
                id_val := ulid_generate();
                id_bin := NULL;
            WHEN 'ulid_generate_optimized' THEN
                id_val := ulid_generate_optimized();
                id_bin := NULL;
            WHEN 'typeid_generate' THEN
                id_val := (typeid_generate(prefix))::text;
                id_bin := NULL;
            WHEN 'typeid_generate_text' THEN
                id_val := typeid_generate_text(prefix);
                id_bin := NULL;
        END CASE;
        
        end_time := clock_timestamp();
        elapsed_ns := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000000000;
        storage_size := calculate_storage_size(func_name, id_val, id_bin);
        
        INSERT INTO extended_benchmark (function_name, id_value, id_binary, generation_time_ns, storage_size_bytes)
        VALUES (func_name, id_val, id_bin, elapsed_ns, storage_size);
    END LOOP;
    
    -- Return statistics
    RETURN QUERY
    SELECT 
        b.function_name,
        COUNT(*)::INTEGER as total_iterations,
        AVG(generation_time_ns) as avg_time_ns,
        MIN(generation_time_ns) as min_time_ns,
        MAX(generation_time_ns) as max_time_ns,
        STDDEV(generation_time_ns) as std_dev_ns,
        AVG(storage_size_bytes) as avg_storage_bytes
    FROM extended_benchmark b
    WHERE b.function_name = func_name
    GROUP BY b.function_name;
END;
$$ LANGUAGE plpgsql;