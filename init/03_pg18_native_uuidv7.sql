-- PostgreSQL 18 Native UUIDv7 Support
-- This script adds native uuidv7() function testing for PostgreSQL 18

-- Check if we're running on PostgreSQL 18+
DO $$
DECLARE
    pg_version_num INTEGER;
BEGIN
    SELECT current_setting('server_version_num')::INTEGER INTO pg_version_num;
    
    -- PostgreSQL 18 has version number 180000+
    IF pg_version_num >= 180000 THEN
        RAISE NOTICE 'PostgreSQL 18+ detected - native uuidv7() support available';
        
        -- Test native uuidv7() function
        PERFORM uuidv7();
        RAISE NOTICE 'Native uuidv7() function working correctly';
        
        -- Test uuidv4() alias  
        PERFORM uuidv4();
        RAISE NOTICE 'Native uuidv4() alias working correctly';
        
        -- Create wrapper functions for consistent benchmarking
        CREATE OR REPLACE FUNCTION uuidv7_native()
        RETURNS uuid
        AS $$
        BEGIN
            RETURN uuidv7();
        END;
        $$ LANGUAGE plpgsql VOLATILE;
        
        -- Function to generate UUIDv7 with time offset (unique to PG18)
        CREATE OR REPLACE FUNCTION uuidv7_native_offset(time_offset INTERVAL DEFAULT '0 seconds')
        RETURNS uuid
        AS $$
        BEGIN
            RETURN uuidv7(time_offset);
        END;
        $$ LANGUAGE plpgsql VOLATILE;
        
        -- Function to extract timestamp from UUIDv7 (PG18 feature)
        CREATE OR REPLACE FUNCTION test_uuid_extract_features()
        RETURNS TABLE (
            uuid_val uuid,
            version_num INTEGER,
            extracted_timestamp TIMESTAMPTZ
        ) AS $$
        DECLARE
            test_uuid uuid;
        BEGIN
            -- Generate a test UUIDv7
            test_uuid := uuidv7();
            
            RETURN QUERY SELECT 
                test_uuid,
                uuid_extract_version(test_uuid),
                uuid_extract_timestamp(test_uuid);
        END;
        $$ LANGUAGE plpgsql VOLATILE;
        
        RAISE NOTICE 'PostgreSQL 18 native UUIDv7 functions configured successfully';
    ELSE
        RAISE NOTICE 'PostgreSQL version % - native uuidv7() not available', pg_version_num;
        
        -- Create dummy functions that will fail gracefully
        CREATE OR REPLACE FUNCTION uuidv7_native()
        RETURNS uuid
        AS $$
        BEGIN
            RAISE EXCEPTION 'Native uuidv7() not available in PostgreSQL < 18';
        END;
        $$ LANGUAGE plpgsql VOLATILE;
        
        CREATE OR REPLACE FUNCTION uuidv7_native_offset(time_offset INTERVAL DEFAULT '0 seconds')
        RETURNS uuid
        AS $$
        BEGIN
            RAISE EXCEPTION 'Native uuidv7() with offset not available in PostgreSQL < 18';
        END;
        $$ LANGUAGE plpgsql VOLATILE;
        
        CREATE OR REPLACE FUNCTION test_uuid_extract_features()
        RETURNS TABLE (
            uuid_val uuid,
            version_num INTEGER,
            extracted_timestamp TIMESTAMPTZ
        ) AS $$
        BEGIN
            RAISE EXCEPTION 'UUID extraction functions not available in PostgreSQL < 18';
        END;
        $$ LANGUAGE plpgsql VOLATILE;
    END IF;
END;
$$;

-- Function to compare native vs custom implementations (PostgreSQL 18 only)
CREATE OR REPLACE FUNCTION compare_uuidv7_implementations(iterations INTEGER DEFAULT 1000)
RETURNS TABLE (
    implementation TEXT,
    avg_time_ns NUMERIC,
    sample_uuid TEXT,
    has_sub_ms_precision BOOLEAN
) AS $$
DECLARE
    pg_version_num INTEGER;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    times BIGINT[];
    i INTEGER;
    sample UUID;
    uuid1 UUID;
    uuid2 UUID;
    has_precision BOOLEAN;
BEGIN
    SELECT current_setting('server_version_num')::INTEGER INTO pg_version_num;
    
    IF pg_version_num < 180000 THEN
        RAISE EXCEPTION 'This comparison requires PostgreSQL 18+';
    END IF;
    
    -- Test native implementation
    times := ARRAY[]::BIGINT[];
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        sample := uuidv7();
        end_time := clock_timestamp();
        times := array_append(times, EXTRACT(EPOCH FROM (end_time - start_time)) * 1000000000);
    END LOOP;
    
    -- Check for sub-millisecond precision by generating two rapid UUIDs
    uuid1 := uuidv7();
    uuid2 := uuidv7();
    has_precision := uuid1 != uuid2; -- Should be different due to sub-ms precision
    
    RETURN QUERY SELECT 
        'native_uuidv7',
        (SELECT AVG(t) FROM unnest(times) AS t),
        sample::TEXT,
        has_precision;
    
    -- Test custom implementation 1
    times := ARRAY[]::BIGINT[];
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        sample := uuid_generate_v7();
        end_time := clock_timestamp();
        times := array_append(times, EXTRACT(EPOCH FROM (end_time - start_time)) * 1000000000);
    END LOOP;
    
    RETURN QUERY SELECT 
        'custom_uuid_generate_v7',
        (SELECT AVG(t) FROM unnest(times) AS t),
        sample::TEXT,
        FALSE; -- Custom implementation doesn't guarantee sub-ms precision
        
    -- Test custom implementation 2
    times := ARRAY[]::BIGINT[];
    FOR i IN 1..iterations LOOP
        start_time := clock_timestamp();
        sample := uuidv7_sub_ms();
        end_time := clock_timestamp();
        times := array_append(times, EXTRACT(EPOCH FROM (end_time - start_time)) * 1000000000);
    END LOOP;
    
    RETURN QUERY SELECT 
        'custom_uuidv7_sub_ms',
        (SELECT AVG(t) FROM unnest(times) AS t),
        sample::TEXT,
        TRUE; -- This implementation has sub-ms precision
END;
$$ LANGUAGE plpgsql;

-- Add PostgreSQL version info to benchmark metadata
CREATE OR REPLACE FUNCTION get_postgres_version_info()
RETURNS TABLE (
    version_string TEXT,
    version_number INTEGER,
    has_native_uuidv7 BOOLEAN,
    has_uuid_extract BOOLEAN
) AS $$
DECLARE
    pg_version_num INTEGER;
BEGIN
    SELECT current_setting('server_version_num')::INTEGER INTO pg_version_num;
    
    RETURN QUERY SELECT 
        version(),
        pg_version_num,
        pg_version_num >= 180000,
        pg_version_num >= 180000;
END;
$$ LANGUAGE plpgsql;

-- Enhanced benchmark function that includes native uuidv7() when available
CREATE OR REPLACE FUNCTION benchmark_id_function_enhanced(
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
    avg_storage_bytes NUMERIC,
    postgresql_version INTEGER,
    is_native_function BOOLEAN
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    elapsed_ns BIGINT;
    id_val TEXT;
    id_bin BYTEA;
    storage_size INTEGER;
    pg_version_num INTEGER;
    is_native BOOLEAN := FALSE;
BEGIN
    SELECT current_setting('server_version_num')::INTEGER INTO pg_version_num;
    
    -- Check if this is a native function
    IF func_name = 'uuidv7_native' AND pg_version_num >= 180000 THEN
        is_native := TRUE;
    END IF;
    
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
            WHEN 'uuidv7_native' THEN
                IF pg_version_num >= 180000 THEN
                    id_val := uuidv7_native()::text;
                    id_bin := uuid_send(id_val::uuid);
                ELSE
                    RAISE EXCEPTION 'Native uuidv7() not available in PostgreSQL < 18';
                END IF;
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
        AVG(storage_size_bytes) as avg_storage_bytes,
        pg_version_num,
        is_native
    FROM extended_benchmark b
    WHERE b.function_name = func_name
    GROUP BY b.function_name;
END;
$$ LANGUAGE plpgsql;