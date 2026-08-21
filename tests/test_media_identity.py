import os
from pathlib import Path

from app.services.media_identity import CatalogueClient, MediaLibrary, parse_media_filename, score_candidate
from app.store import JobStore


def test_scene_movie_name_is_cleaned():
    parsed = parse_media_filename("f1-Waterboys.2001.Japanese.EnglishSubtitles.mkv")
    assert parsed.title == "Waterboys"
    assert parsed.year == 2001
    assert parsed.media_type == "movie"


def test_tv_episode_conventions_are_understood():
    scene = parse_media_filename("The.Last.of.Us.S01E03.2160p.WEB-DL.DDP5.1.H.265-NTb.mkv")
    alternate = parse_media_filename("The Last of Us - 1x04 - Please Hold to My Hand.mkv")
    assert (scene.title, scene.season, scene.episode) == ("The Last of Us", 1, 3)
    assert (alternate.normalized_title, alternate.season, alternate.episode) == ("the last of us", 1, 4)


def test_numeric_titles_are_not_mistaken_for_release_years():
    assert parse_media_filename("1917.1080p.BluRay.mkv").title == "1917"
    odyssey = parse_media_filename("2001.A.Space.Odyssey.1968.2160p.mkv")
    assert odyssey.title == "2001 A Space Odyssey"
    assert odyssey.year == 1968


def test_embedded_catalogue_ids_and_multi_episode_are_parsed():
    parsed = parse_media_filename("Slow.Horses.S04E01-E02.{tmdb=95480}.mkv")
    assert parsed.tmdb_id == 95480
    assert (parsed.season, parsed.episode, parsed.episode_end) == (4, 1, 2)
    assert parse_media_filename("Dark.S01E01.{tvmaze=178}.mkv").tvmaze_id == 178


def test_candidate_scoring_rewards_exact_title_year_and_type():
    parsed = parse_media_filename("Waterboys.2001.mkv")
    exact = score_candidate(parsed, {"title": "Waterboys", "year": 2001, "media_type": "movie"})
    wrong = score_candidate(parsed, {"title": "Waterboy", "year": 1998, "media_type": "movie"})
    television = score_candidate(parsed, {"title": "Waterboys", "year": 2001, "media_type": "tv"})
    assert exact >= 0.95
    assert wrong < exact
    assert television < exact


def test_same_title_resolves_to_one_project(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    library = MediaLibrary(store, tmp_path)
    for job_id, filename in (("one", "Waterboys.2001.1080p.mkv"), ("two", "Waterboys (2001) Remux.mkv")):
        store.create({"id": job_id, "filename": filename, "status": "queued", "progress": 0, "cues": []})
        library.attach(job_id, filename)
    assert store.get("one")["project_id"] == store.get("two")["project_id"]
    assert len(store.list_projects()) == 1


def test_range_suffix_and_missing_year_join_one_unambiguous_project(tmp_path: Path):
    store = JobStore(tmp_path / "ranges.sqlite3")
    library = MediaLibrary(store, tmp_path / "data")
    for job_id, filename in (("full", "Waterboys.2001.mkv"), ("range", "Waterboys.-20m20s.mkv")):
        store.create({"id": job_id, "filename": filename, "status": "queued", "progress": 0, "cues": []})
        library.attach(job_id, filename)
    assert store.get("full")["project_id"] == store.get("range")["project_id"]
    assert len(store.list_projects()) == 1


def test_yearless_title_does_not_cross_two_known_remakes(tmp_path: Path):
    store = JobStore(tmp_path / "remakes.sqlite3")
    library = MediaLibrary(store, tmp_path / "data")
    for job_id, filename in (("old", "The.Thing.1982.mkv"), ("new", "The.Thing.2011.mkv"), ("unknown", "The.Thing.mkv")):
        store.create({"id": job_id, "filename": filename, "status": "queued", "progress": 0, "cues": []})
        library.attach(job_id, filename)
    assert len({store.get(key)["project_id"] for key in ("old", "new", "unknown")}) == 3


def test_kodi_style_nfo_and_local_poster_take_priority(tmp_path: Path):
    media = tmp_path / "Odd.Release.Name.1080p.mkv"
    media.write_bytes(b"media")
    media.with_suffix(".nfo").write_text(
        "<movie><title>The Correct Film</title><year>2007</year>"
        "<uniqueid type='tmdb'>12345</uniqueid></movie>", encoding="utf-8"
    )
    (tmp_path / "Odd.Release.Name.1080p-poster.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 300)
    store = JobStore(tmp_path / "library.sqlite3")
    store.create({"id": "nfo", "filename": media.name, "status": "queued", "progress": 0, "cues": []})
    library = MediaLibrary(store, tmp_path / "data")
    project = library.attach("nfo", media.name, media)
    job = store.get("nfo")
    assert project["title"] == "The Correct Film"
    assert project["year"] == 2007
    assert job["media_identity"]["tmdb_id"] == 12345
    assert store.get_project(job["project_id"])["artwork_source"] == "local"
    assert library.poster_path(job["project_id"]).is_file()


def test_show_folder_supplies_title_for_episode_only_filename(tmp_path: Path):
    source = tmp_path / "Severance (2022)" / "Season 01" / "S01E04.mkv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"media")
    store = JobStore(tmp_path / "shows.sqlite3")
    store.create({"id": "episode", "filename": source.name, "status": "queued", "progress": 0, "cues": []})
    library = MediaLibrary(store, tmp_path / "data")
    project = library.attach("episode", source.name, source)
    assert (project["title"], project["year"], project["media_type"]) == ("Severance", 2022, "tv")
    assert store.get("episode")["media_identity"]["episode"] == 4


def test_managed_application_key_is_added_without_bearer_header(monkeypatch):
    monkeypatch.delenv("TMDB_TOKEN", raising=False)
    monkeypatch.delenv("DUBLINE_TMDB_TOKEN", raising=False)
    monkeypatch.setenv("DUBLINE_TMDB_API_KEY", "application-key")
    client = CatalogueClient()
    assert client.tmdb_available
    assert client._tmdb_url("https://example.test/search?q=film").endswith("q=film&api_key=application-key")


def test_the_bundled_token_fills_an_empty_slot_but_never_beats_the_user(monkeypatch, tmp_path):
    # Released builds carry Dubline's own TMDB credential so a non-technical
    # user gets covers with no account.  It must sit underneath everything else.
    from app import config

    blank_env = tmp_path / ".env"
    blank_env.write_text("DUBLINE_TMDB_TOKEN=\n", encoding="utf-8")
    monkeypatch.setattr(config, "BUNDLED_TMDB_TOKEN", "shipped-application-token")

    for name in ("TMDB_TOKEN", "DUBLINE_TMDB_TOKEN", "DUBLINE_TMDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config.load_local_environment(blank_env)
    summary = config.media_lookup_summary()
    assert summary["managed"] is True
    assert summary["movie_lookup"] is True
    assert summary["display"] == "Included with Dubline"
    assert CatalogueClient().token == "shipped-application-token"

    # A token the user saved in Setup outranks the shipped one.
    monkeypatch.setenv("TMDB_TOKEN", "a-personal-token-of-sufficient-length")
    personal = config.media_lookup_summary()
    assert personal["managed"] is False
    assert CatalogueClient().token == "a-personal-token-of-sufficient-length"


def test_a_fork_without_a_bundled_token_still_looks_up_television(monkeypatch, tmp_path):
    from app import config

    monkeypatch.setattr(config, "BUNDLED_TMDB_TOKEN", "")
    for name in ("TMDB_TOKEN", "DUBLINE_TMDB_TOKEN", "DUBLINE_TMDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config.load_local_environment(tmp_path / "absent.env")
    summary = config.media_lookup_summary()
    assert summary["managed"] is False
    assert summary["movie_lookup"] is False
    assert summary["tv_lookup"] is True
    assert "DUBLINE_TMDB_TOKEN" not in os.environ
