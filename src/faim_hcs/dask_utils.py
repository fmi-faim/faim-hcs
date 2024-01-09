from multiprocessing import Process, Queue
from time import sleep

import numpy as np


class LocalClusterFactory:
    """Creates a local dask cluster in a sub-process."""

    def __init__(
        self,
        n_workers: int = None,
        threads_per_worker: int = None,
        processes: bool = None,
        memory_limit: str = None,
        local_directory: str = None,
    ):
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.memory_limit = memory_limit
        self.local_directory = local_directory
        self._queue = Queue(1)
        self._scheduler_address = None

        self._subprocess = Process(
            target=self._run_cluster,
            args=(
                self._queue,
                self.n_workers,
                self.threads_per_worker,
                self.processes,
                self.memory_limit,
                self.local_directory,
            ),
        )
        self._subprocess.start()
        self._scheduler_address = None
        self._client = None

    def _get_scheduler_address(self):
        if self._scheduler_address is None:
            scheduler_adress = self._queue.get()
            self._scheduler_address = scheduler_adress
            self._queue.close()
            self._queue = None

        return self._scheduler_address

    def _shutdown(self):
        if self._client is not None and self._client.scheduler is not None:
            self._client.shutdown()
            self._client = None

        if self._subprocess is not None and self._subprocess.is_alive():
            self._subprocess.join()

    def __del__(self):
        self._shutdown()

    @staticmethod
    def _run_cluster(
        queue: Queue,
        n_workers: int,
        threads_per_worker: int,
        processes: bool,
        memory_limit: str,
        local_directory: str,
    ):
        import dask
        import distributed

        dask.config.set({"distributed.workers.memory.spill": 0.90})
        dask.config.set({"distributed.workers.memory.target": 0.85})
        dask.config.set({"distributed.workers.memory.terminate": 0.98})

        client = distributed.Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            memory_limit=memory_limit,
            local_directory=local_directory,
        )
        queue.put(client.cluster.scheduler.address)
        while client.cluster.scheduler.status.value != "closed":
            sleep(5)

    def get_client(self):
        """Get a dask client for the local cluster."""
        if self._client is None:
            import distributed

            self._client = distributed.Client(self._get_scheduler_address())

        return self._client


def mean_cast_to(target_dtype):
    """
    Wrap np.mean to cast the result to a given dtype.
    """

    def _mean(
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        *,
        where=np._NoValue,
    ):
        return np.mean(
            a=a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        ).astype(target_dtype)

    return _mean
