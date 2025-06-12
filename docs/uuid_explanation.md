# UUIDv7 Implementation Deep Dive

## Understanding UUIDv7 Structure

UUIDv7 is designed to be time-sortable while maintaining uniqueness. The structure is:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           unix_ts_ms                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          unix_ts_ms           |  ver  |       rand_a          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|var|                        rand_b                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            rand_b                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

- **unix_ts_ms**: 48-bit big-endian unsigned timestamp (milliseconds since Unix epoch)
- **ver**: 4-bit version field (0111 binary = 7 decimal)
- **rand_a**: 12-bit random data (or sub-millisecond precision in method 3)
- **var**: 2-bit variant field (10 binary)
- **rand_b**: 62-bit random data

## Function 1: uuid_generate_v7() Explained

```sql
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
```

### Step-by-step breakdown:

1. **Generate base UUID**: `gen_random_uuid()` creates a UUIDv4
2. **Convert to binary**: `uuid_send()` converts to 16-byte binary format
3. **Get timestamp**: `extract(epoch from clock_timestamp()) * 1000` gets milliseconds
4. **Convert timestamp**: `int8send()` converts to 8-byte big-endian, take bytes 3-8 (6 bytes)
5. **Overlay timestamp**: Replace first 6 bytes of UUID with timestamp
6. **Set version bits**: Bits 52 and 53 set to 1 (making version field 0111 = 7)
7. **Convert to hex**: Final UUID in standard format

### Visual representation:

```
Original UUIDv4:  [RRRRRRRR-RRRR-4RRR-VRRR-RRRRRRRRRRRR]
                   ^^^^^^^^^ ^^^^ ^^^^ ^^^^ ^^^^^^^^^^^^

After overlay:    [TTTTTTTT-TTTT-4RRR-VRRR-RRRRRRRRRRRR]
                   ^^^^^^^^^ ^^^^
                   timestamp

After version:    [TTTTTTTT-TTTT-7RRR-VRRR-RRRRRRRRRRRR]
                                ↑
                            version 7
```

## Function 2: uuidv7() Explained

This function is nearly identical to Function 1 but written as a single SQL statement:

```sql
SELECT encode(
  set_bit(
    set_bit(
      overlay(uuid_send(gen_random_uuid()) placing
        substring(int8send((extract(epoch from clock_timestamp())*1000)::bigint) from 3)
        from 1 for 6),
      52, 1),
    53, 1), 'hex')::uuid;
```

The main differences:

- Pure SQL implementation (no PL/pgSQL wrapper)
- Slightly more compact syntax
- May have marginally different performance characteristics

## Function 3: uuidv7_sub_ms() Explained

```sql
SELECT encode(
  substring(int8send(floor(t_ms)::int8) from 3) ||
  int2send((7<<12)::int2 | ((t_ms-floor(t_ms))*4096)::int2) ||
  substring(uuid_send(gen_random_uuid()) from 9 for 8)
, 'hex')::uuid
FROM (SELECT extract(epoch from clock_timestamp())*1000 as t_ms) s
```

### Unique features:

1. **Sub-millisecond precision**: Uses fractional part of timestamp
2. **Manual construction**: Builds UUID from scratch rather than overlaying
3. **Bit manipulation for version**: `(7<<12)` shifts version bits into position

### Breakdown:

```
Part 1: 6 bytes timestamp (48 bits)
Part 2: 2 bytes containing:
  - 4 bits version (7)
  - 12 bits sub-millisecond data
Part 3: 8 bytes random data

Result: [TIMESTAMP_MS][VER+SUBSEC][RANDOM_DATA]
```

## Performance Characteristics

### Function 1 & 2 (overlay method):

- **Pros**:
  - Reuses existing UUID generation
  - Simple bit manipulation
  - Good performance
- **Cons**:
  - No sub-millisecond precision
  - Slightly more operations

### Function 3 (construction method):

- **Pros**:
  - Sub-millisecond precision
  - Direct construction
  - Better time ordering within same millisecond
- **Cons**:
  - More complex logic
  - Potentially slower due to multiple concatenations

## Collision Analysis

All functions use sufficient randomness:

- Functions 1 & 2: 74 bits of randomness (12 + 62)
- Function 3: 62 bits of randomness

Probability of collision within same millisecond:

- Functions 1 & 2: ~1 in 2^74 ≈ 1 in 10^22
- Function 3: ~1 in 2^62 ≈ 1 in 10^18

Even generating 1 billion UUIDs per millisecond, collisions remain astronomically unlikely.

## Time Ordering Guarantees

- **Between milliseconds**: All functions guarantee ordering
- **Within millisecond**:
  - Functions 1 & 2: Random ordering
  - Function 3: Sub-millisecond ordering (better granularity)

## Use Case Recommendations

1. **High-volume inserts**: Function 2 (pure SQL, minimal overhead)
2. **Precise time ordering**: Function 3 (sub-millisecond precision)
3. **General purpose**: Function 1 (good balance, clear implementation)
4. **Distributed systems**: Any function (all provide sufficient uniqueness)
