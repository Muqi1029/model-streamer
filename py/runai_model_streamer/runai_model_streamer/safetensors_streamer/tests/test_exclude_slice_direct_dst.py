# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
    RunaiStreamerMemoryLimitException,
)
from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
)


class TestExcludeSliceAndDirectDst(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _save_shard(self, filename: str, tensors: dict) -> str:
        path = os.path.join(self.test_dir, filename)
        save_file(tensors, path)
        return path

    def test_exclude_tensor_names_filters_across_files_and_ignores_unknown(self):
        t_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        t_b = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        t_c = torch.arange(20, dtype=torch.uint8).reshape(5, 4)
        p1 = self._save_shard("s1.safetensors", {"a": t_a, "b": t_b})
        p2 = self._save_shard("s2.safetensors", {"c": t_c})

        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                [p1, p2],
                exclude_tensor_names={"b", "not_in_checkpoint"},
            )
            loaded = dict(streamer.get_tensors())

        self.assertEqual(set(loaded.keys()), {"a", "c"})
        self.assertTrue(torch.equal(loaded["a"], t_a))
        self.assertTrue(torch.equal(loaded["c"], t_c))

    def test_exclude_all_tensors_yields_empty(self):
        p1 = self._save_shard("only.safetensors", {"a": torch.ones(4, dtype=torch.float32)})
        with SafetensorsStreamer() as streamer:
            streamer.stream_files([p1], exclude_tensor_names={"a"})
            loaded = list(streamer.get_tensors())
        self.assertEqual(loaded, [])

    def test_exclude_tensor_names_validation(self):
        p1 = self._save_shard("v.safetensors", {"a": torch.ones(4, dtype=torch.float32)})
        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "str"):
                streamer.stream_files([p1], exclude_tensor_names="a")
            with self.assertRaisesRegex(ValueError, "both"):
                streamer.stream_files(
                    [p1], tensor_names={"a"}, exclude_tensor_names={"a"}
                )

    def test_tensor_slices_reads_exact_dim0_rows_and_empty_slice(self):
        table = torch.arange(80, dtype=torch.int32).reshape(10, 8)
        small = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        p1 = self._save_shard("sliced.safetensors", {"table": table, "small": small})

        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                [p1],
                tensor_slices={"table": (3, 7)},
            )
            loaded = dict(streamer.get_tensors())

        self.assertEqual(loaded["table"].shape, (4, 8))
        self.assertTrue(torch.equal(loaded["table"], table[3:7]))
        self.assertTrue(torch.equal(loaded["small"], small))

        # Empty slice [5, 5) should yield a 0-row tensor of the expected shape & dtype.
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                [p1],
                tensor_names={"table"},
                tensor_slices={"table": (5, 5)},
            )
            loaded_empty = dict(streamer.get_tensors())

        self.assertEqual(loaded_empty["table"].shape, (0, 8))
        self.assertEqual(loaded_empty["table"].dtype, torch.int32)

    def test_tensor_slices_validation(self):
        table = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        scalar = torch.tensor(3.0, dtype=torch.float32)
        p1 = self._save_shard("bad_slice.safetensors", {"table": table, "scalar": scalar})

        with SafetensorsStreamer() as streamer:
            with self.assertRaises(ValueError):
                streamer.stream_files([p1], tensor_slices={"table": (2, 5)})
            with self.assertRaises(ValueError):
                streamer.stream_files([p1], tensor_slices={"table": (3, 1)})
            with self.assertRaises(ValueError):
                streamer.stream_files([p1], tensor_slices={"scalar": (0, 1)})
            with self.assertRaisesRegex(ValueError, "missing_key"):
                streamer.stream_files([p1], tensor_slices={"missing_key": (0, 1)})

    def test_tensor_dsts_streams_directly_with_slices_bypassing_memory_limit(self):
        # Simulates a huge CPU-resident Engram table sharded across ranks:
        # even with RUNAI_STREAMER_MEMORY_LIMIT=1 byte, direct dst streaming succeeds
        # with zero staging ring allocation and writes straight into dst memory.
        weight = torch.arange(1024, dtype=torch.uint8).reshape(16, 64)
        scale = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
        gpu_w = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        p1 = self._save_shard(
            "engram.safetensors",
            {"engram.weight": weight, "engram.scale": scale, "gpu.w": gpu_w},
        )

        # Shared host buffer (per_node style): rank writes into its own contiguous row slice.
        shared_w = torch.zeros_like(weight)
        shared_s = torch.zeros_like(scale)
        dst_w = shared_w[4:10]
        dst_s = shared_s[4:10]

        # 512 bytes is larger than the safetensors header (~200 bytes) but smaller than
        # engram.weight (1024 bytes), mirroring the real 40 GB limit vs 91.55 GiB Engram tensor.
        with patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "512"}):
            # Confirm normal host-ring streaming fails under a 512-byte memory limit.
            with SafetensorsStreamer() as streamer:
                with self.assertRaises(RunaiStreamerMemoryLimitException):
                    streamer.stream_files([p1], tensor_names={"engram.weight"})

            # Direct-into-dst with row slicing succeeds and writes straight into shared_w[4:10].
            with SafetensorsStreamer() as streamer:
                streamer.stream_files(
                    [p1],
                    device="cpu",
                    is_distributed=False,
                    tensor_slices={"engram.weight": (4, 10), "engram.scale": (4, 10)},
                    tensor_dsts={"engram.weight": dst_w, "engram.scale": dst_s},
                )
                loaded = dict(streamer.get_tensors())

        self.assertIs(loaded["engram.weight"], dst_w)
        self.assertIs(loaded["engram.scale"], dst_s)
        self.assertTrue(torch.equal(shared_w[4:10], weight[4:10]))
        self.assertTrue(torch.equal(shared_s[4:10], scale[4:10]))
        # Untouched rows remain zero.
        self.assertEqual(shared_w[:4].sum().item(), 0)
        self.assertEqual(shared_w[10:].sum().item(), 0)

    def test_tensor_dsts_empty_slice(self):
        weight = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
        p1 = self._save_shard("empty_dst.safetensors", {"w": weight})
        dst = torch.empty((0, 8), dtype=torch.uint8)

        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                [p1],
                tensor_slices={"w": (2, 2)},
                tensor_dsts={"w": dst},
            )
            loaded = dict(streamer.get_tensors())

        self.assertIs(loaded["w"], dst)
        self.assertEqual(loaded["w"].shape, (0, 8))

    def test_tensor_dsts_validation(self):
        weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        p1 = self._save_shard("val_dst.safetensors", {"w": weight})

        with SafetensorsStreamer() as streamer:
            with self.assertRaisesRegex(ValueError, "cpu"):
                streamer.stream_files(
                    [p1], device="cuda:0", tensor_dsts={"w": torch.empty_like(weight)}
                )
            with self.assertRaisesRegex(ValueError, "is_distributed"):
                streamer.stream_files(
                    [p1], is_distributed=True, tensor_dsts={"w": torch.empty_like(weight)}
                )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                streamer.stream_files(
                    [p1], tensor_dsts={"w": torch.empty(8, 4, dtype=torch.float32).t()}
                )
            with self.assertRaisesRegex(ValueError, "bytes"):
                streamer.stream_files(
                    [p1], tensor_dsts={"w": torch.empty(2, 8, dtype=torch.float32)}
                )

    def test_memory_limit_parameter_overrides_env_without_mutating_it(self):
        # 4 KiB tensor; env limit is 100 bytes (too small), explicit memory_limit=0 succeeds.
        big = torch.arange(1024, dtype=torch.float32)
        p1 = self._save_shard("mem_override.safetensors", {"big": big})

        with patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: "100"}):
            with SafetensorsStreamer() as streamer:
                streamer.stream_files([p1], memory_limit=0)
                loaded = dict(streamer.get_tensors())
            self.assertEqual(os.environ[RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME], "100")

        self.assertTrue(torch.equal(loaded["big"], big))


if __name__ == "__main__":
    unittest.main()
