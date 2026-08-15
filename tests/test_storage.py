import math

from raglab import ConvertedDocument, LineProvenance, ProvenanceStatus
from raglab.storage.postgres import _chunk_page_range, _vector


def test_vector_serialization_is_explicit_and_precise() -> None:
    assert _vector([1.0, -0.25, math.pi]) == "[1,-0.25,3.1415926535897931]"


def test_chunk_page_range_uses_only_mapped_canonical_lines() -> None:
    document = ConvertedDocument(
        "memory://guide",
        "one\ntwo\nthree",
        "hash",
        "test",
        "1",
        "guide.pdf",
        line_provenance=(
            LineProvenance(1, page_number=3),
            LineProvenance(2),
            LineProvenance(3, page_number=4),
        ),
        provenance_status=ProvenanceStatus.PARTIAL,
    )

    assert _chunk_page_range(document, 1, 3) == (3, 4)
    assert _chunk_page_range(document, 2, 2) == (None, None)
    assert _chunk_page_range(document, None, None) == (None, None)
