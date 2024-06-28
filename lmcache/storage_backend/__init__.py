from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.storage_backend.local_backend import LMCLocalBackend
#from lmcache.storage_backend.remote_backend import LMCRemoteBackend
#from lmcache.storage_backend.hybrid_backend import LMCHybridBackend#, LMCPipelinedHybridBackend
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger

logger = init_logger(__name__)


def CreateStorageBackend(
        config: LMCacheEngineConfig, 
        metadata: LMCacheEngineMetadata
    ) -> LMCBackendInterface:
    match config:
        case LMCacheEngineConfig(local_device=str(p)) if p is not None:
            # local only
            logger.info("Initializing local-only backend")
            return LMCLocalBackend(config)

        case _:
            raise ValueError(f"Invalid configuration: {config}")

#__all__ = [
#    "LMCBackendInterface",
#    "LMCLocalBackend",
#    "LMCRemoteBackend",
#]
