from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_release_reuses_full_ci_for_the_tag_commit_before_building():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "workflow_call:" in ci
    assert "ci:\n    uses: ./.github/workflows/ci.yml" in release
    assert "build:\n    needs: ci" in release


def test_release_asserts_tag_matches_distribution_version():
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "Verify tag matches package version" in release
    assert "GITHUB_REF_NAME" in release
    assert '["project"]["version"]' in release
