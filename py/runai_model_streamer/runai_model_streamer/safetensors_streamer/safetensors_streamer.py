# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Collection, Dict, Iterator, Optional, Tuple
from collections import Counter
import torch
import glob
import os
import fcntl
import shutil
import logging
import humanize
from timeit import default_timer as timer
from typing import List

from runai_model_streamer.file_streamer import FileChunks
from runai_model_streamer.file_streamer.file_streamer import host_or_device

from runai_model_streamer.distributed_streamer import DistributedStreamer

import runai_model_streamer.safetensors_streamer.safetensors_pytorch as safetensors_pytorch

from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
    is_s3_path,
    is_gs_path,
    is_azure_path,
    s3_glob,
    s3_pull_files,
    gcs_glob,
    gcs_pull_files,
    azure_glob,
    azure_pull_files,
)

logger = logging.getLogger(__name__)

SAFETENSORS_PATTERN = "*.safetensors"

def list_safetensors(path: str, s3_credentials : Optional[S3Credentials] = None) -> List[str]:
    """
    List all safetensors files in the given path.
    This function is not recursive.
    """
    if is_s3_path(path):
        files = s3_glob(path, [SAFETENSORS_PATTERN], s3_credentials)
    elif is_gs_path(path):
        files = gcs_glob(path, [SAFETENSORS_PATTERN])
    elif is_azure_path(path):
        files = azure_glob(path, [SAFETENSORS_PATTERN])
    else:
        files = glob.glob(os.path.join(path, SAFETENSORS_PATTERN))
    
    return files

def pull_files(model_path: str,
                dst: str,
                allow_pattern: Optional[List[str]] = None,
                ignore_pattern: Optional[List[str]] = None,
                s3_credentials : Optional[S3Credentials] = None) -> None:
    """
    Pull all safetensors files in the given path.
    This function is recursive.
    """
    if is_s3_path(model_path):
        return s3_pull_files(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)
    if is_gs_path(model_path):
        return gcs_pull_files(model_path, dst, allow_pattern, ignore_pattern)
    if is_azure_path(model_path):
        return azure_pull_files(model_path, dst, allow_pattern, ignore_pattern)
    raise NotImplementedError("pull files is not implemented for file system paths")

class ObjectStorageModel:
    """
    Process-safe, idempotent wrapper for downloading model files from object storage.

    Multiple processes calling pull_files() concurrently with the same dst will
    serialize via a file lock. The first process downloads; the rest wait and skip
    (sentinel-based idempotency).

    Use as a context manager. The sentinel is written and the lock is released on
    clean exit. If an exception is raised inside the block the lock is still released
    but the sentinel is NOT written, so the next process will retry the download.

    Locking mechanism:
        Uses fcntl.flock, which is supported on Linux only (not Windows).
        The lock file is placed at dst + ".lock".

        Supported: multiple processes on the same machine, whether dst is a local
        disk or a network-mounted filesystem (NFS, EFS, etc.) — the kernel manages
        the lock locally.

        Not supported: processes on different machines sharing the same dst over
        a network filesystem. fcntl.flock does not provide reliable cross-host
        locking on NFS (NFSv3 and earlier silently ignore it; NFSv4 depends on
        mount options and server support).

    Example::

        with ObjectStorageModel(model_path=url, dst=cache_dir) as obj:
            obj.pull_files(allow_pattern=["*.safetensors"])
            obj.pull_files(ignore_pattern=["*.safetensors"])
    """

    SENTINEL_NAME = ".runai_complete"

    def __init__(
        self,
        model_path: str,
        dst: str,
        s3_credentials: Optional[S3Credentials] = None,
    ) -> None:
        if not (is_s3_path(model_path) or is_gs_path(model_path) or is_azure_path(model_path)):
            raise ValueError(
                f"model_path {model_path!r} is not a supported object storage path "
                "(expected s3://, gs://, or az://)"
            )
        self.dir = dst.rstrip("/") or "/"
        self._model_path = model_path if model_path.endswith("/") else model_path + "/"
        self._s3_credentials = s3_credentials
        self._lock_path = self.dir + ".lock"
        self._sentinel = os.path.join(self.dir, self.SENTINEL_NAME)
        self._lock_file = None
        self._lock_file = open(self._lock_path, "a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_SH)  # shared: fast path for already-downloaded
            if os.path.exists(self._sentinel):
                self._skip = True
            else:
                # Release SH before requesting EX. Direct SH→EX promotion can
                # deadlock if two processes both try to upgrade simultaneously:
                # each blocks waiting for the other to release its SH, and
                # neither ever runs again. flock(2) does not detect this cycle.
                # Re-check sentinel after acquiring EX since another process may
                # have completed the download in the gap.
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
                if os.path.exists(self._sentinel):
                    self._skip = True
                    # Downgrade to SH: no further writes needed, allowing other
                    # waiting processes to proceed in parallel rather than serialize.
                    fcntl.flock(self._lock_file, fcntl.LOCK_SH)
                else:
                    self._skip = False
                    if os.path.exists(self.dir):
                        shutil.rmtree(self.dir)
                    os.makedirs(self.dir, exist_ok=True)
        except BaseException:
            # BaseException (not Exception) is intentional: KeyboardInterrupt and
            # SystemExit must also release the lock, otherwise sibling processes
            # waiting on flock will block indefinitely after a Ctrl+C or SIGTERM.
            if self._lock_file is not None:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                self._lock_file.close()
            raise

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        if self._skip:
            return
        pull_files(self._model_path, self.dir, allow_pattern, ignore_pattern, self._s3_credentials)

    def __del__(self) -> None:
        # Use getattr in case __init__ raised before self._lock_file was assigned.
        lock_file = getattr(self, '_lock_file', None)
        if lock_file is not None and not lock_file.closed:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def __enter__(self) -> ObjectStorageModel:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None and not self._skip:
                downloaded = [f for f in os.listdir(self.dir) if f != self.SENTINEL_NAME]
                if not downloaded:
                    raise RuntimeError(
                        f"No files were downloaded to {self.dir!r} — "
                        "verify that the model path is correct"
                    )
                try:
                    with open(self._sentinel, "w"):
                        pass
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to write download sentinel {self._sentinel!r}: {exc}"
                    ) from exc
        finally:
            lock_file = getattr(self, '_lock_file', None)
            if lock_file is not None:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                finally:
                    lock_file.close()
        return False


class SafetensorsStreamer:
    def __init__(self) -> None:
        self.file_streamer = DistributedStreamer()
        self.files_to_tensors_metadata = {}
        self.total_size = 0
        self.device_str = None
        self.start_time = None
        # Tensors this session had to copy because their address was unusable for their dtype. Per
        # session, not per module: a process can load two models, and a threaded consumer would lose
        # increments to a shared counter.
        self.unaligned_copies = 0
        # Maps (file_index, chunk_index) -> caller-owned CPU tensor when tensor_dsts is used.
        self._direct_dsts: Dict[Tuple[int, int], torch.Tensor] = {}
        # Checkpoint tensor names excluded during the last stream_files() call.
        self.excluded_tensor_names: set[str] = set()

    def __enter__(self) -> SafetensorsStreamer:
        self.file_streamer.__enter__()
        return self

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        return self.file_streamer.__exit__(exc_type, exc_value, traceback)

    def _log_throughput(self) -> None:
        # The session-level throughput: one SafetensorsStreamer session may drive several FileStreamer
        # requests (memory-limited chunking today, concurrent submissions with the ring buffer), so the
        # overall figure belongs here, not in FileStreamer. Emitted from get_tensors() completion (active
        # consumption) rather than __exit__, which under the vllm generator loader can run at interpreter
        # shutdown after the logging handlers are torn down (record silently dropped).
        if self.start_time is None:
            # stream_files() never ran, so there is nothing to report. A throughput log must never raise
            # (it would break get_tensors()), so bail out instead of computing timer() - None.
            return
        size = self.total_size
        elapsed_time = timer() - self.start_time
        throughput = size / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"[RunAI Streamer] Overall time to stream {humanize.naturalsize(size, binary=True)} of all files to {self.device_str}: {round(elapsed_time, 2)}s, {humanize.naturalsize(throughput, binary=True)}/s"
        )
        # Only when it happened. A zero every load would train the reader to skip the line, which is
        # the one case where it carries information.
        if self.unaligned_copies:
            logger.debug(
                f"[RunAI Streamer] Copied {self.unaligned_copies} tensor(s) whose address their dtype "
                f"could not use"
            )

    def stream_file(
            self,
            path: str,
            s3_credentials : Optional[S3Credentials] = None,
            device: Optional[str] = "cpu",
            is_distributed: bool = False,
            tensor_names: Optional[Collection[str]] = None,
            owned: bool = True,
            exclude_tensor_names: Optional[Collection[str]] = None,
            tensor_slices: Optional[Dict[str, Tuple[int, int]]] = None,
            tensor_dsts: Optional[Dict[str, torch.Tensor]] = None,
            memory_limit: Optional[int] = None,
        ) -> None:
        # By keyword, so that adding a parameter to stream_files cannot silently shift these into the
        # wrong slots. is_distributed and owned are both bools, so a shift would still run.
        return self.stream_files(
            [path],
            s3_credentials=s3_credentials,
            device=device,
            is_distributed=is_distributed,
            tensor_names=tensor_names,
            owned=owned,
            exclude_tensor_names=exclude_tensor_names,
            tensor_slices=tensor_slices,
            tensor_dsts=tensor_dsts,
            memory_limit=memory_limit,
        )


    def stream_files(
            self,
            paths: List[str],
            s3_credentials : Optional[S3Credentials] = None,
            device: Optional[str] = "cpu",
            is_distributed: bool = False,
            tensor_names: Optional[Collection[str]] = None,
            owned: bool = True,
            exclude_tensor_names: Optional[Collection[str]] = None,
            tensor_slices: Optional[Dict[str, Tuple[int, int]]] = None,
            tensor_dsts: Optional[Dict[str, torch.Tensor]] = None,
            memory_limit: Optional[int] = None,
        ) -> None:
        """Stream the tensors of one or more safetensors files.

        owned=True, the default, gives every yielded tensor to the caller: it stays valid for as
        long as the caller holds it, so there is no need to copy one in order to keep it.

        owned=False yields views into buffers the streamer reuses, so each must be consumed before
        the iteration advances. Retaining one then reads whatever was written next, silently.

        exclude_tensor_names skips the named tensors during this stream (mutually exclusive with
        tensor_names). Unknown names in exclude_tensor_names are ignored.

        tensor_slices maps tensor name -> (start_row, end_row) along dimension 0 so only that row
        range is fetched from storage.

        tensor_dsts maps tensor name -> pre-allocated contiguous CPU torch.Tensor, streaming bytes
        directly into the caller's memory with zero staging ring allocation (requires device="cpu"
        and is_distributed=False). When tensor_names and exclude_tensor_names are both None,
        tensor_dsts implicitly selects tensor_dsts.keys().

        memory_limit overrides RUNAI_STREAMER_MEMORY_LIMIT for a non-distributed host ring without
        mutating os.environ.

        Alignment is not under this flag, and nor is a device destination without distributed
        streaming - see docs/src/usage.md.
        """
        if tensor_names is not None and exclude_tensor_names is not None:
            raise ValueError("Cannot specify both tensor_names and exclude_tensor_names")
        # None = no filter (today's behavior). str satisfies Collection[str] structurally, so
        # reject it - otherwise `in` below silently becomes substring matching.
        if isinstance(tensor_names, str):
            raise ValueError("tensor_names must be a collection of names, not a single str")
        if isinstance(exclude_tensor_names, str):
            raise ValueError("exclude_tensor_names must be a collection of names, not a single str")
        # Empty = certainly a caller bug, reject it up front.
        if tensor_names is not None and len(tensor_names) == 0:
            raise ValueError("tensor_names must not be empty - pass None to load every tensor")

        if tensor_dsts is not None:
            if host_or_device(device) != "cpu":
                raise ValueError(
                    f"tensor_dsts requires device='cpu', got device={device!r}"
                )
            if is_distributed:
                raise ValueError("tensor_dsts is not supported with is_distributed=True")
            if len(tensor_dsts) == 0:
                raise ValueError("tensor_dsts must not be empty")
            if tensor_names is None and exclude_tensor_names is None:
                tensor_names = tensor_dsts.keys()

        # Normalize once for O(1) membership below.
        if tensor_names is not None:
            tensor_names = frozenset(tensor_names)
        if exclude_tensor_names is not None:
            exclude_tensor_names = frozenset(exclude_tensor_names)

        if is_distributed:
            # Every rank turns this list's ORDER into FileChunks ids, and the ranks then exchange
            # tensors by that id without ever comparing their lists. Two ranks that enumerated the
            # directory differently would hold different files behind one id, and the weights would
            # be wrong with nothing raised. Callers reach here from glob, which does not sort.
            #
            # Not under the ordering flag: the ranks have to agree whether or not the yield is
            # ordered. A single process reading for itself needs no agreement, so its order is left
            # exactly as the caller gave it.
            #
            # Not a yield order either. Ranks broadcast one batch each per round, so the tensors a
            # caller sees are interleaved by rank whatever this list's order is.
            paths = sorted(paths)

        self.files_to_tensors_metadata = {}
        self.total_size = 0
        self.device_str = device
        self.unaligned_copies = 0
        self._direct_dsts = {}
        self.excluded_tensor_names = set()

        file_stream_requests: List[FileChunks] = []
        file_dst_tensors: List[List[torch.Tensor]] = []

        # Start the throughput clock here, right before the first submission: prepare_request issues the
        # metadata reads (the first submissions), so the measurement spans first submission -> last response.
        self.start_time = timer()

        # metadata is created on cpu and each process reads it individually (or shards
        # across ranks for object storage when is_distributed=True).
        safetensors_metadatas = safetensors_pytorch.prepare_request(
            self.file_streamer, paths, s3_credentials, discover_ranks=is_distributed
        )
        prepare_elapsed = timer() - self.start_time
        if prepare_elapsed >= 1.0:
            logger.info(
                f"[RunAI Streamer] Prepared headers for {len(paths)} file(s) to {device} in {prepare_elapsed:.2f}s"
            )

        # Every kept name, WITH duplicates - only tensor_names carries the expectation of one
        # tensor per requested name, so this is never tracked for the unfiltered load.
        all_kept_names: List[str] = [] if tensor_names is not None else None
        matched_slice_names = set()
        needs_custom_offsets = (
            tensor_names is not None
            or exclude_tensor_names is not None
            or tensor_slices is not None
            or tensor_dsts is not None
        )

        for i in range(len(paths)):
            (file_offset, tensors_metadata, tensor_sizes) = safetensors_metadatas[i]
            path = paths[i]

            if not needs_custom_offsets:
                self.files_to_tensors_metadata[i] = tensors_metadata
                self.total_size += sum(tensor_sizes)
                file_stream_requests.append(FileChunks.contiguous(i, path, file_offset, tensor_sizes))
                continue

            if tensor_names is not None:
                selected = [tm for tm in tensors_metadata if tm.name in tensor_names]
            elif exclude_tensor_names is not None:
                selected = []
                for tm in tensors_metadata:
                    if tm.name in exclude_tensor_names:
                        self.excluded_tensor_names.add(tm.name)
                    else:
                        selected.append(tm)
            else:
                selected = list(tensors_metadata)

            kept_metadata = []
            kept_offsets = []
            kept_sizes = []
            kept_dsts: List[torch.Tensor] = []

            for tm in selected:
                if tensor_slices is not None and tm.name in tensor_slices:
                    start_row, end_row = tensor_slices[tm.name]
                    sliced_tm, rel_offset, byte_size = tm.slice_dim0(start_row, end_row)
                    matched_slice_names.add(tm.name)
                    out_tm = sliced_tm
                    abs_offset = file_offset + rel_offset
                else:
                    out_tm = tm
                    abs_offset = file_offset + tm.offsets.start
                    byte_size = tm.get_bytesize()

                if tensor_dsts is not None:
                    if tm.name not in tensor_dsts:
                        raise ValueError(
                            f"tensor_dsts is missing destination tensor for '{tm.name}'"
                        )
                    dst = tensor_dsts[tm.name]
                    if not torch.is_tensor(dst) or dst.device.type != "cpu":
                        raise ValueError(
                            f"tensor_dsts['{tm.name}'] must be a CPU torch.Tensor"
                        )
                    if not dst.is_contiguous():
                        raise ValueError(
                            f"tensor_dsts['{tm.name}'] must be contiguous"
                        )
                    if dst.nbytes != byte_size:
                        raise ValueError(
                            f"tensor_dsts['{tm.name}'] has {dst.nbytes} bytes, "
                            f"expected {byte_size} bytes for shape {out_tm.shape} ({out_tm.dtype})"
                        )
                    self._direct_dsts[(i, len(kept_metadata))] = dst
                    kept_dsts.append(dst)

                kept_metadata.append(out_tm)
                kept_offsets.append(abs_offset)
                kept_sizes.append(byte_size)

            if all_kept_names is not None:
                all_kept_names.extend(tm.name for tm in kept_metadata)

            self.files_to_tensors_metadata[i] = kept_metadata
            self.total_size += sum(kept_sizes)
            file_stream_requests.append(FileChunks(i, path, kept_offsets, kept_sizes))
            if tensor_dsts is not None:
                file_dst_tensors.append(kept_dsts)

        if tensor_names is not None:
            # A name only counts as "unknown" if it matched in NONE of the files, not just the
            # one being processed at the time - so dedupe across all of them first.
            kept_set = set(all_kept_names)

            missing = tensor_names - kept_set
            if missing:
                raise ValueError(f"tensor_names not found in checkpoint: {sorted(missing)}")

            # Only pay for finding WHICH names collided when the gate actually trips.
            if len(kept_set) != len(all_kept_names):
                counts = Counter(all_kept_names)
                duplicates = sorted(name for name, count in counts.items() if count > 1)
                raise ValueError(
                    f"tensor_names requested tensors that appear in more than one file: {duplicates}"
                )

        if tensor_slices is not None:
            unused_slices = set(tensor_slices) - matched_slice_names
            if unused_slices:
                raise ValueError(
                    f"tensor_slices keys not found in streamed tensors: {sorted(unused_slices)}"
                )

        if tensor_dsts is not None:
            self.file_streamer.stream_files_into(
                file_stream_requests,
                file_dst_tensors,
                credentials=s3_credentials,
            )
            return

        self.file_streamer.stream_files(
            file_stream_requests,
            credentials=s3_credentials,
            device=device,
            is_distributed=is_distributed,
            owned=owned,
            discover_ranks=is_distributed,
            memory_limit=memory_limit,
        )
        self._log_ring()

    def _log_ring(self) -> None:
        """Report the ring once per load, at the session boundary.

        Here rather than where the ring is built, for two reasons. The rank: a distributed load builds
        one ring per rank, and FilesRequestsIteratorWithBuffer cannot know which rank it is, so at INFO
        it would emit N unattributable copies. And the count: stream_files runs three times per load,
        twice for the safetensors metadata (`1 x 8 Bytes`), so only this one describes the model read.

        Worth INFO at all because the depth is DERIVED - from the limit, the ranks on the node and the
        largest tensor - not taken from the setting, so a ring clamped to one buffer is otherwise
        indistinguishable from the configured four, and they perform very differently. A failure to
        report must never break a load, hence the broad except."""
        try:
            ring = self.file_streamer.ring_info()
            if ring is None:
                return
            rank, num_buffers, buffer_size, label = ring
            logger.info(
                f"[RunAI Streamer] Rank {rank}: {label} ring {num_buffers} x "
                f"{humanize.naturalsize(buffer_size, binary=True)} = "
                f"{humanize.naturalsize(num_buffers * buffer_size, binary=True)} "
                f"for {humanize.naturalsize(self.total_size, binary=True)} in {len(self.files_to_tensors_metadata)} file(s)"
            )
        except Exception:
            logger.debug("[RunAI Streamer] could not report the ring", exc_info=True)

    def get_tensors(self) -> Iterator[torch.tensor]:
        for file_index, ready_chunk_index, buffer in self.file_streamer.get_chunks():
            tensor_metadata = self.files_to_tensors_metadata[file_index][ready_chunk_index]

            direct_dst = self._direct_dsts.get((file_index, ready_chunk_index))
            if direct_dst is not None:
                yield tensor_metadata.name, direct_dst
                continue

            # Copied BEFORE the dtype view, and on the uint8 buffer: a copy afterwards would already
            # have had to interpret the address this is here to avoid. Only what has to be copied is,
            # so a model whose tensors all sit where their dtypes can use them pays nothing.
            if safetensors_pytorch.requires_alignment_copy(buffer, tensor_metadata):
                buffer = buffer.clone()
                self.unaligned_copies += 1

            tensor = safetensors_pytorch.create_torch_tensor(buffer, tensor_metadata)

            # AUTHORITATIVE, where the check above only decides whether to copy. Making the copy
            # aligned relies on the allocator handing back a suitably aligned block, which is an
            # implementation detail of torch and not a promise to us. Verified rather than assumed, so
            # the day it stops holding is a failed load and not a model that is quietly wrong.
            element_size = tensor_metadata.get_element_size()
            if tensor.data_ptr() % element_size:
                raise ValueError(
                    f"Tensor '{tensor_metadata.name}' is at {tensor.data_ptr():#x}, which "
                    f"{tensor_metadata.dtype} cannot be read from ({element_size}-byte elements)"
                )

            yield tensor_metadata.name, tensor

        # All tensors delivered (the underlying get_chunks() is exhausted). Log the session throughput here,
        # during active consumption while the logging system is still alive.
        self._log_throughput()
