import pytest

from latentcore.storage.database import init_database
from latentcore.storage.latent_store import LatentStore
from latentcore.utils.serialization import bytes_to_indices, indices_to_bytes


class TestSerialization:
    def test_roundtrip(self):
        original = [0, 100, 255, 8191, 1, 42]
        blob = indices_to_bytes(original)
        recovered = bytes_to_indices(blob)
        assert recovered == original

    def test_compact_size(self):
        indices = list(range(1000))
        blob = indices_to_bytes(indices)
        # int16 = 2 bytes per index
        assert len(blob) == 2000

    def test_empty_list(self):
        blob = indices_to_bytes([])
        recovered = bytes_to_indices(blob)
        assert recovered == []


class TestLatentStore:
    @pytest.fixture
    async def store(self):
        db = await init_database(":memory:")
        yield LatentStore(db)
        await db.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        indices = [10, 20, 30, 40, 50]
        ref_id = await store.save_ref(
            indices=indices,
            original_token_count=500,
            metadata={"source": "test"},
        )

        loaded = await store.load_ref(ref_id)
        assert loaded is not None
        assert loaded["indices"] == indices
        assert loaded["original_token_count"] == 500
        assert loaded["num_indices"] == 5
        assert loaded["metadata"]["source"] == "test"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load_ref("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        ref_id = await store.save_ref(indices=[1, 2, 3], original_token_count=10)
        assert await store.delete_ref(ref_id) is True
        assert await store.load_ref(ref_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        assert await store.delete_ref("nonexistent") is False

    @pytest.mark.asyncio
    async def test_list_refs(self, store):
        for i in range(5):
            await store.save_ref(indices=[i], original_token_count=10)

        refs = await store.list_refs(limit=3)
        assert len(refs) == 3

    @pytest.mark.asyncio
    async def test_count_refs(self, store):
        assert await store.count_refs() == 0
        await store.save_ref(indices=[1], original_token_count=10)
        await store.save_ref(indices=[2], original_token_count=20)
        assert await store.count_refs() == 2

    @pytest.mark.asyncio
    async def test_access_tracking(self, store):
        ref_id = await store.save_ref(indices=[1, 2], original_token_count=10)

        # Load twice
        await store.load_ref(ref_id)
        loaded = await store.load_ref(ref_id)

        assert loaded["access_count"] == 2

    @pytest.mark.asyncio
    async def test_compression_ratio_calculated(self, store):
        ref_id = await store.save_ref(indices=[1, 2, 3, 4, 5], original_token_count=100)
        loaded = await store.load_ref(ref_id)
        assert loaded["compression_ratio"] == 20.0  # 100 / 5
