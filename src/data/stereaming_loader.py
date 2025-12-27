"""
Ultra-memory-efficient loader for 76M+ row parquet files.
Uses streaming to avoid loading entire file at once.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterator, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NETFLOW_FEATURES = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES',
    'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES',
    'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES',
    'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
    'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE'
]  # 40 features (removed SRC_FRAGMENTS, DST_FRAGMENTS)


class StreamingParquetLoader:
    """
    Streams parquet data in small chunks to avoid OOM.
    Does NOT load the entire file at once.
    """
    
    def __init__(self,
                 data_path: str = 'data/parquet/nf_uq/NF-UQ-NIDS-v2.parquet',
                 chunk_size: int = 50_000,
                 normalize: bool = False):
        """
        Args:
            data_path: Path to parquet file
            chunk_size: How many rows to buffer at once (default 50k = safe for 2GB RAM)
            normalize: Whether to normalize (requires pre-computed stats)
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.feature_stats = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        logger.info(f"Parquet file: {self.data_path}")
        logger.info(f"Chunk size: {chunk_size:,} rows")
    
    def stream_batches(self, 
                      limit_rows: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Stream batches of (X, y) from parquet file.
        Memory usage: Only one chunk at a time (~50MB per 50k rows).
        
        Args:
            limit_rows: Maximum rows to stream (None = all)
            
        Yields:
            (X_batch, y_batch) tuples
        """
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.data_path)
            
            total_rows = parquet_file.metadata.num_rows
            if limit_rows:
                total_rows = min(limit_rows, total_rows)
            
            logger.info(f"Total rows available: {total_rows:,}")
            
            rows_read = 0
            for i in range(parquet_file.num_row_groups):
                if rows_read >= total_rows:
                    break
                
                # Read one row group at a time
                table = parquet_file.read_row_group(i, columns=NETFLOW_FEATURES + ['Attack'])
                df = table.to_pandas()
                
                # Process in smaller chunks if needed
                for j in range(0, len(df), self.chunk_size):
                    if rows_read >= total_rows:
                        break
                    
                    end_idx = min(j + self.chunk_size, len(df))
                    chunk = df.iloc[j:end_idx]
                    
                    # Convert to float64 first to avoid overflow, then to float32
                    X = chunk[NETFLOW_FEATURES].fillna(0).values.astype(np.float64)
                    # Clip extreme values before converting to float32
                    X = np.clip(X, -1e6, 1e6)
                    X = X.astype(np.float32)
                    
                    y = chunk['Attack'].values
                    
                    # Normalize if needed
                    if self.normalize and self.feature_stats is not None:
                        mean, std = self.feature_stats
                        X = (X - mean) / (std + 1e-8)
                    
                    rows_read += len(X)
                    logger.info(f"Streamed {rows_read:,} rows...")
                    
                    yield X, y
                
                # Cleanup
                del df, table
                
        except ImportError:
            # Fallback if PyArrow not available
            logger.warning("PyArrow not available, using slower pandas method")
            # Delegate to the pandas-based generator
            yield from self._stream_batches_pandas(limit_rows)
    
    def _stream_batches_pandas(self, limit_rows: Optional[int]):
        """Fallback: try to read with pandas using available parquet engines.

        Notes:
            - pandas does not support streaming parquet natively; this fallback
              will read the file into a DataFrame and then yield chunked
              slices. This may use more memory than the pyarrow path.
        """
        # Try available engines in order
        engines_tried = []
        df = None
        for engine in ('pyarrow', 'fastparquet', None):
            try:
                if engine is None:
                    # Let pandas pick a suitable engine
                    df = pd.read_parquet(self.data_path, columns=NETFLOW_FEATURES + ['Attack'])
                else:
                    df = pd.read_parquet(self.data_path, columns=NETFLOW_FEATURES + ['Attack'], engine=engine)
                logger.info(f"Read parquet with pandas using engine={engine}")
                break
            except Exception as e:
                engines_tried.append((engine, str(e)))
                df = None

        if df is None:
            msg_lines = ["Failed to read parquet with pandas fallback. Engines tried:"]
            for eng, err in engines_tried:
                msg_lines.append(f"  - engine={eng}: {err}")
            msg_lines.append("Install 'pyarrow' (recommended) or 'fastparquet', or run inside the project venv that has pyarrow installed.")
            logger.error('\n'.join(msg_lines))
            raise RuntimeError('\n'.join(msg_lines))

        total = len(df)
        rows_read = 0
        for start in range(0, total, self.chunk_size):
            if limit_rows and rows_read >= limit_rows:
                break

            end = min(start + self.chunk_size, total)
            chunk = df.iloc[start:end]

            if limit_rows and rows_read + len(chunk) > limit_rows:
                chunk = chunk.iloc[: (limit_rows - rows_read)]

            X = chunk[NETFLOW_FEATURES].fillna(0).values.astype(np.float32)
            y = chunk['Attack'].values

            if self.normalize and self.feature_stats is not None:
                mean, std = self.feature_stats
                X = (X - mean) / (std + 1e-8)

            rows_read += len(X)
            logger.info(f"Streamed {rows_read:,} rows (pandas fallback)...")
            yield X, y


# For compatibility with old code
MemoryEfficientDataLoader = StreamingParquetLoader
