import pytest

from lit_screening.domain_packs.loader import load_domain_pack, list_domain_packs


def test_list_domain_packs_includes_materials_magnetism():
    assert "materials_magnetism" in list_domain_packs()


def test_load_materials_magnetism_domain_pack():
    pack = load_domain_pack("materials_magnetism")

    assert pack.domain_name == "materials_magnetism"
    assert "surface_magnetization" in pack.concepts


def test_surface_magnetization_synonyms_include_boundary_magnetization():
    pack = load_domain_pack("materials_magnetism")

    assert "boundary magnetization" in pack.concepts["surface_magnetization"].synonyms


def test_materials_magnetism_methods_include_expected_probe_terms():
    pack = load_domain_pack("materials_magnetism")

    assert "SPLEEM" in pack.methods
    assert "XMCD-PEEM" in pack.methods
    assert "NV magnetometry" in pack.methods


def test_materials_magnetism_false_positive_terms_include_clinical_screening():
    pack = load_domain_pack("materials_magnetism")

    assert "clinical screening" in pack.false_positive_terms


def test_missing_domain_pack_raises_clear_error():
    with pytest.raises(ValueError, match="does not exist"):
        load_domain_pack("missing_domain")
