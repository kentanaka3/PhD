# ObsPy Mass Downloader Architectural Findings and Reference

This document records technical findings, mathematical derivations, source-code behaviors, and bug mitigations discovered in `obspy.clients.fdsn.mass_downloader` during seismological data acquisition in the OGS pipeline.

---

## 1. Mass Downloader Chunking Mechanics & Compression

### 1.1 Source-Level Evidence [OBSERVED]
In `obspy.clients.fdsn.mass_downloader.download_helpers.py` (lines 820–830), ObsPy calculates the estimated payload size of waveform requests using an explicit Steim compression assumption:

```python
# Assume that each sample needs 4 byte, STEIM
# compression reduces size to about a third.
# chunk size is in MB
duration = interval.end - interval.start
curr_chunks_mb += \
    sr * duration * 4.0 / 3.0 / 1024.0 / 1024.0
if curr_chunks_mb >= chunk_size_in_mb or \
        len(chunks_curr) >= max_chunk_length:
    chunks.append(chunks_curr)
    chunks_curr = []
    curr_chunks_mb = 0
```

ObsPy enforces two independent chunk boundary conditions:
1. **Payload Estimate Ceiling**: `curr_chunks_mb >= chunk_size_in_mb` (default: 25 MB).
2. **Channel Count Ceiling**: `len(chunks_curr) >= max_chunk_length` (hardcoded to `50` in line 795).

---

## 2. Mathematical Derivation for 100 Hz Daily Windows

### 2.1 Daily Sample Count [OBSERVED]
For a 1-day acquisition window ($24\text{ hours} = 86,400\text{ seconds}$) at a sampling rate of $100\text{ Hz}$:
$$\text{Samples per channel} = 100\text{ samples/s} \times 86,400\text{ s} = 8,640,000\text{ samples}$$

### 2.2 Single-Channel Size
- **Uncompressed Raw (32-bit / 4 bytes per sample)**:
  $$\frac{8,640,000 \times 4\text{ B}}{1024^2\text{ B/MiB}} = \frac{34,560,000}{1,048,576} \approx 32.96\text{ MB}$$
- **Steim Compressed (ObsPy Factor: $4/3 \approx 1.33$ bytes per sample)**:
  $$\frac{8,640,000 \times (4/3)\text{ B}}{1024^2\text{ B/MiB}} = \frac{11,520,000}{1,048,576} \approx 10.99\text{ MB}$$

### 2.3 50 Channels (ObsPy Request Ceiling: `max_chunk_length = 50`)
- **Uncompressed (4 bytes per sample)**:
  $$50 \times 32.96\text{ MB} \approx 1,647.95\text{ MB}\; (\approx 1.61\text{ GiB})$$
- **Steim Compressed ($4/3$ bytes per sample)**:
  $$50 \times 10.99\text{ MB} \approx 549.32\text{ MB}$$

### 2.4 Invariance of Chunk Size $\ge 550\text{ MB}$ [INFERENCE]
Because ObsPy evaluates `len(chunks_curr) >= max_chunk_length` (50 channels), any value of `download_chunk_size_in_mb >= 550` (such as 600, 733, or 1000) will **always hit the 50-channel hard ceiling first** when processing 1-day 100 Hz data. Consequently, values $\ge 550$ result in identical chunking behavior (50 channels per bulk HTTP POST request).

---

## 3. Provider Rate Limiting (HTTP 429) Root Cause

### 3.1 Mechanism [OBSERVED]
With ObsPy's default `download_chunk_size_in_mb = 25`:
1. Even with Steim compression, 3 channels reach $\approx 33\text{ MB} > 25\text{ MB}$.
2. ObsPy splices station channel requests into numerous tiny HTTP POST requests (e.g. 31 single-channel requests for a daily network).
3. When multiple worker threads query public FDSN servers (e.g. `data.raspberryshake.org`), the rapid burst of distinct HTTP requests trips server-side rate limits, returning:
   ```text
   HTTP 429: Sent too many requests in a given amount of time ('rate limiting'). Wait before making a new request.
   ```

### 3.2 Mitigation [PROPOSAL]
Setting `download_chunk_size_in_mb >= 550` (e.g. 600 or 733) aggregates all channels for a provider (up to 50 channels) into a single bulk HTTP request per day, reducing HTTP transaction frequency by over 90% and avoiding rate-limiting thresholds.

---

## 4. ObsPy `raise_on_error` Bug and Encapsulated Patch

### 4.1 Bug Description [OBSERVED]
In `obspy.clients.fdsn.client.Client.raise_on_error(code, data)`:
When an FDSN endpoint encounters a connection timeout or connection reset (e.g. `http://158.110.30.217:8080`), ObsPy catches `URLError` / `ConnectionResetError` and forwards the exception instance into `raise_on_error(code, data)` as the `data` parameter.
ObsPy calls `data.splitlines()`, assuming `data` is a byte string or text payload. This causes an unhandled crash:
```text
AttributeError: 'ConnectionResetError' object has no attribute 'splitlines'
```
This unhandled `AttributeError` crashed worker processes in multiprocessing pools.

### 4.2 Solution in `OGS/src/ogsdownloader.py` [OBSERVED]
In `OGS/src/ogsdownloader.py`, `ObsPyDownloader._safe_raise_on_error` intercepts calls:
- If `isinstance(data, Exception)`, it re-raises the exception directly, allowing `obspy.clients.fdsn.mass_downloader.utils.ERRORS` to catch and record the network error gracefully.
- Otherwise, it delegates to ObsPy's standard `raise_on_error(code, data)`.

---

## 5. Metadata Column Warnings (`CR.*`) [OBSERVED]

Stations federated through EIDA from Croatian networks (`CR.*`) often emit:
```text
UserWarning: Empty float value in column 3 (depth) or 4 (dip) or 5 (azimuth).
```
These warnings originate from missing optional float fields in EIDA station-text tables. They are non-fatal and do not interrupt waveform downloads.

