from multiprocessing import Process, Queue

from distributed import Client

from faim_hcs.dask_utils import LocalClusterFactory


def test_local_cluster_factory(tmp_path_factory):
    fac = LocalClusterFactory(
        n_workers=1,
        threads_per_worker=1,
        memory_limit="1GB",
        local_directory=str(tmp_path_factory.mktemp("dask")),
    )

    client = fac.get_client()
    assert client.scheduler_info() is not None
    assert client.scheduler_info()["workers"] is not None
    assert len(client.scheduler_info()["workers"]) == 1
    for worker in client.scheduler_info()["workers"].values():
        assert worker["nthreads"] == 1
        assert worker["memory_limit"] == 1e9

    assert fac._subprocess.is_alive()

    c2 = fac.get_client()
    assert client == c2
    assert fac._scheduler_address == fac._get_scheduler_address()

    assert fac._subprocess.is_alive()

    fac._shutdown()

    assert not fac._subprocess.is_alive()
    assert fac._subprocess.exitcode == 0

    fac._shutdown()


def test__start_local_cluster(tmp_path_factory):
    queue = Queue(1)
    p = Process(
        target=LocalClusterFactory._run_cluster,
        args=(
            queue,
            1,
            1,
            "1GB",
            str(tmp_path_factory.mktemp("dask")),
        ),
    )
    p.start()
    sa = queue.get()
    assert sa is not None
    client = Client(sa)
    client.shutdown()
    p.join()
    assert p.exitcode == 0
