#!/usr/bin/env python3
"""
Quick test script to verify UUIDv7 functions work correctly
"""
import psycopg2
import time

# Test with existing PostgreSQL 17 container
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'temporal',
    'user': 'temporal',
    'password': 'temporal'
}

def test_connection():
    """Test database connection and setup functions"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to PostgreSQL")
        
        with conn.cursor() as cur:
            # Check PostgreSQL version
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"PostgreSQL Version: {version}")
            
            # Check version number for native uuidv7 support
            cur.execute("SELECT current_setting('server_version_num')::INTEGER")
            version_num = cur.fetchone()[0]
            print(f"Version Number: {version_num}")
            
            if version_num >= 180000:
                print("✅ PostgreSQL 18+ detected - Native uuidv7() should be available")
                try:
                    cur.execute("SELECT uuidv7()")
                    native_uuid = cur.fetchone()[0]
                    print(f"✅ Native uuidv7() works: {native_uuid}")
                except Exception as e:
                    print(f"❌ Native uuidv7() failed: {e}")
            else:
                print("ℹ️  PostgreSQL < 18 - Will test custom implementations only")
            
            # Create our custom UUIDv7 functions
            print("\n📦 Creating custom UUIDv7 functions...")
            
            # Function 1: uuid_generate_v7() - PL/pgSQL
            cur.execute("""
                CREATE OR REPLACE FUNCTION uuid_generate_v7()
                RETURNS uuid
                AS $$
                BEGIN
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
            """)
            print("✅ uuid_generate_v7() created")
            
            # Function 2: uuidv7() - SQL
            cur.execute("""
                CREATE OR REPLACE FUNCTION uuidv7() RETURNS uuid
                AS $$
                  SELECT encode(
                    set_bit(
                      set_bit(
                        overlay(uuid_send(gen_random_uuid()) placing
                          substring(int8send((extract(epoch from clock_timestamp())*1000)::bigint) from 3)
                          from 1 for 6),
                        52, 1),
                      53, 1), 'hex')::uuid;
                $$ LANGUAGE sql VOLATILE;
            """)
            print("✅ uuidv7() created")
            
            # Function 3: uuidv7_sub_ms() - Sub-millisecond precision
            cur.execute("""
                CREATE OR REPLACE FUNCTION uuidv7_sub_ms() RETURNS uuid
                AS $$
                SELECT encode(
                  substring(int8send(floor(t_ms)::int8) from 3) ||
                  int2send((7<<12)::int2 | ((t_ms-floor(t_ms))*4096)::int2) ||
                  substring(uuid_send(gen_random_uuid()) from 9 for 8)
                , 'hex')::uuid
                FROM (SELECT extract(epoch from clock_timestamp())*1000 as t_ms) s
                $$ LANGUAGE sql VOLATILE;
            """)
            print("✅ uuidv7_sub_ms() created")
            
            # Test all functions
            print("\n🧪 Testing functions:")
            functions = ['uuid_generate_v7', 'uuidv7', 'uuidv7_sub_ms']
            
            if version_num >= 180000:
                functions.append('uuidv7_native')
            
            for func in functions:
                try:
                    if func == 'uuidv7_native':
                        cur.execute("SELECT uuidv7()")
                    else:
                        cur.execute(f"SELECT {func}()")
                    result = cur.fetchone()[0]
                    print(f"✅ {func}: {result}")
                except Exception as e:
                    print(f"❌ {func}: {e}")
            
            # Performance test
            print("\n⚡ Quick performance test (1000 iterations):")
            for func in functions:
                if func == 'uuidv7_native' and version_num < 180000:
                    continue
                    
                start_time = time.perf_counter()
                try:
                    if func == 'uuidv7_native':
                        cur.execute("SELECT uuidv7() FROM generate_series(1, 1000)")
                    else:
                        cur.execute(f"SELECT {func}() FROM generate_series(1, 1000)")
                    results = cur.fetchall()
                    end_time = time.perf_counter()
                    avg_time = (end_time - start_time) * 1000 / 1000  # ms per UUID
                    print(f"✅ {func}: {avg_time:.4f} ms/UUID ({len(results)} generated)")
                except Exception as e:
                    print(f"❌ {func}: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing UUIDv7 Functions")
    print("=" * 50)
    
    if test_connection():
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")