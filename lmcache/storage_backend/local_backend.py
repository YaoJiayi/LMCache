from typing import Tuple, Optional, Iterator
import re
import io
import torch
import os

from lmcache.utils import CacheEngineKey, KVCache
from lmcache.config import LMCacheEngineConfig
from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

class LMCLocalBackend(LMCBackendInterface):
    """
    Cache engine for storing the KV cache of the tokens in the local cpu/gpu memory.
    """
    def __init__(
            self, 
            config: LMCacheEngineConfig
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """
        super().__init__()

        #self.chunk_size = config.chunk_size 
        self.config = config
        self.dict = {}
        
        #TODO: need to pass this from config
        
        self.path = "cache.pt"
        
        # restore the cache from disk to local pinned cpu mem
        if os.path.exists(self.path):
            self.restore()
        
    def contains(
            self, 
            key: CacheEngineKey,
        ) -> bool:
        """
        Check if the cache engine contains the key.

        Input:
            key: the key of the token chunk, including prefix hash and format

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        return key in self.dict

    def put(
            self, 
            key: CacheEngineKey,
            kv_chunk,
            blocking: bool = True,
        ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            key: the key of the token chunk, including prefix hash and format
            kv_chunk: the kv cache of the token chunk, in the format of nested tuples

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        if not blocking:
            logger.warn("Non-blocking is not implemented for local backend")
        self.dict[key] = kv_chunk
    
    def dump(
        self
    ):
        for key in self.dict.keys():
            logger.info(f"saving chunk {key}")
            logger.info(f"chunk_shape {self.dict[key][0].shape}")
            logger.info(f"chunk_shape {self.dict[key][1].shape}")
        torch.save(self.dict, self.path)
        logger.info(f"saving {len(self.dict.keys())} chunk(s)")
    
    def restore(
        self
    ):
        self.dict = torch.load(self.path)
        for key, val in self.dict.items():
            self.dict[key] = val.cpu().pin_memory()
            

    @_lmcache_nvtx_annotate
    def get(
            self,
            key: CacheEngineKey,
        ) -> Optional[KVCache]:
        """
        Retrive the KV cache chunk by the given key 

        Input:
            key: the key of the token chunk, including prefix hash and format
        Output: 
            the kv cache of the token chunk, in the format of nested tuples
            None if the key is not found
        """
        return self.dict.get(key, None)
