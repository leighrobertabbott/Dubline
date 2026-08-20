from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.config import media_lookup_summary, save_setting, validate_tmdb_token
from app.store import JobStore


_VIDEO_SUFFIXES = {
    ".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".m2ts",
    ".mpeg", ".mpg", ".wmv", ".mxf", ".vob", ".3gp", ".flv", ".ogm", ".iso",
}
_TV_PATTERNS = (
    re.compile(r"(?i)(?<![a-z0-9])s(?P<season>\d{1,2})[ ._-]*e(?P<episode>\d{1,3})(?:[ ._-]*(?:e|-)[ ._-]*(?P<episode_end>\d{1,3}))?"),
    re.compile(r"(?i)(?<!\d)(?P<season>\d{1,2})x(?P<episode>\d{1,3})(?:-(?P<episode_end>\d{1,3}))?"),
    re.compile(r"(?i)\bseason[ ._-]*(?P<season>\d{1,2})[ ._-]*(?:episode|ep)[ ._-]*(?P<episode>\d{1,3})\b"),
)
_ID_PATTERNS = {
    "imdb_id": re.compile(r"(?i)(?<![a-z0-9])(tt\d{7,10})(?!\d)"),
    "tmdb_id": re.compile(r"(?i)(?:\{|\[|\()?tmdb(?:id)?[ ._:=/-]*(\d{1,9})(?:\}|\]|\))?"),
    "tvdb_id": re.compile(r"(?i)(?:\{|\[|\()?tvdb(?:id)?[ ._:=/-]*(\d{1,9})(?:\}|\]|\))?"),
    "tvmaze_id": re.compile(r"(?i)(?:\{|\[|\()?tvmaze(?:id)?[ ._:=/-]*(\d{1,9})(?:\}|\]|\))?"),
}
_RELEASE_TOKEN = re.compile(
    r"""(?ix)^(
      (?:360|480|576|720|1080|1440|2160|4320)p|4k|8k|uhd|hdr10\+?|hdr|dv|dolby[ ._-]?vision|
      x26[45]|h[ ._-]?26[45]|hevc|avc|av1|xvid|divx|10bit|8bit|hi10p|
      web[ ._-]?(?:dl|rip)|bluray|blu[ ._-]?ray|b[dr]rip|remux|hdtv|pdtv|dvdrip|dvd|cam|telesync|webrip|
      dts(?:[ ._-]?hd)?|truehd|atmos|ddp?\d(?:\.\d)?|eac3|ac3|aac\d(?:\.\d)?|flac|opus|mp3|
      proper|repack|rerip|internal|extended|unrated|directors?|theatrical|criterion|imax|restored|remastered|
      \d{1,3}m\d{1,2}s|\d{1,2}h\d{1,2}m(?:\d{1,2}s)?|
      multi|dual|dubbed|subbed|subtitles?|englishsubtitles?|engsub|japanese|korean|chinese|mandarin|cantonese|
      french|german|spanish|italian|russian|hindi|arabic|portuguese|dutch|swedish|norwegian|danish|finnish|
      nf|amzn|dsnp|hmax|atvp|hulu|ma|yts|yify|rarbg|evo|ctrlhd|framestor|tigole|qxr
    )$"""
)
_GROUP_SUFFIX = re.compile(r"(?i)-[a-z0-9]{2,18}$")
_YEAR = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedMediaName:
    source: str
    title: str
    normalized_title: str
    year: int | None
    media_type: str
    season: int | None
    episode: int | None
    episode_end: int | None
    imdb_id: str | None
    tmdb_id: int | None
    tvdb_id: int | None
    tvmaze_id: int | None
    parse_confidence: float

    @property
    def lookup_key(self) -> str:
        if self.media_type == "tv":
            return f"tv:{self.normalized_title}:{self.year or ''}"
        return f"movie:{self.normalized_title}:{self.year or ''}"


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return _SPACE.sub(" ", value).strip()


def _display_title(value: str) -> str:
    words = value.split()
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    output = []
    for index, word in enumerate(words):
        if word.isupper() and len(word) <= 5:
            output.append(word)
        elif index and word.lower() in small:
            output.append(word.lower())
        else:
            output.append(word[:1].upper() + word[1:])
    return " ".join(output)


def parse_media_filename(filename: str) -> ParsedMediaName:
    """Parse common scene, P2P, disc, broadcast and personal library names conservatively."""
    source = Path(filename).name.strip()
    stem = source
    while Path(stem).suffix.lower() in _VIDEO_SUFFIXES:
        stem = Path(stem).stem
    stem = re.sub(r"(?i)^f\d+[-_]", "", stem)  # Dubline's resumable upload prefix.
    ids: dict[str, Any] = {"imdb_id": None, "tmdb_id": None, "tvdb_id": None, "tvmaze_id": None}
    for name, pattern in _ID_PATTERNS.items():
        match = pattern.search(stem)
        if match:
            ids[name] = match.group(1) if name == "imdb_id" else int(match.group(1))
            stem = pattern.sub(" ", stem)

    season = episode = episode_end = None
    tv_match = None
    for pattern in _TV_PATTERNS:
        tv_match = pattern.search(stem)
        if tv_match:
            season, episode = int(tv_match.group("season")), int(tv_match.group("episode"))
            episode_end = int(tv_match.group("episode_end")) if tv_match.groupdict().get("episode_end") else None
            before, after = stem[:tv_match.start()], stem[tv_match.end():]
            # Project identity is the series, not the individual episode title.
            stem = before if re.search(r"[A-Za-z]", before) else after
            break

    # Bracketed checksums and release notes rarely form part of a catalogue title.
    stem = re.sub(r"[\[(]([^\])]{1,80})[\])]",
                  lambda match: f" {match.group(1)} " if _YEAR.fullmatch(match.group(1).strip()) else " ", stem)
    stem = _GROUP_SUFFIX.sub("", stem)
    stem = re.sub(r"[._]+", " ", stem)
    stem = re.sub(r"\s+-\s+", " ", stem)
    stem = re.sub(r"(?i)\b(?:final[ ._-]?cut|director'?s[ ._-]?cut|special[ ._-]?edition|extended[ ._-]?edition)\b", " ", stem)
    year = None
    year_matches = list(_YEAR.finditer(stem))
    if year_matches:
        release_year = year_matches[-1]
        without_year = stem[:release_year.start()] + " " + stem[release_year.end():]
        meaningful = [token for token in re.split(r"\s+", without_year.strip())
                      if token and not _RELEASE_TOKEN.fullmatch(token.strip("-–—,:;{}[]()"))]
        # A filename such as "1917.1080p.mkv" is a numeric title, not a release year with no title.
        if meaningful:
            year = int(release_year.group(1))
            stem = without_year

    tokens = [token for token in re.split(r"\s+", stem.strip()) if token]
    cleaned: list[str] = []
    for token in tokens:
        bare = token.strip("-–—,:;{}[]()")
        is_technical = bool(_RELEASE_TOKEN.fullmatch(bare)) or bool(re.fullmatch(r"\d{3,4}p", bare, re.I))
        if is_technical:
            break
        cleaned.append(bare)
    title_raw = _SPACE.sub(" ", " ".join(cleaned)).strip(" -._")
    if not title_raw:
        title_raw = _SPACE.sub(" ", re.sub(r"[._-]+", " ", Path(source).stem)).strip() or "Untitled media"
    normalized = normalize_title(title_raw)
    title = _display_title(title_raw)
    confidence = 0.96 if year or tv_match or ids["imdb_id"] else (0.84 if len(normalized) >= 4 else 0.55)
    return ParsedMediaName(
        source=source, title=title, normalized_title=normalized, year=year,
        media_type="tv" if tv_match or ids["tvdb_id"] or ids["tvmaze_id"] else "movie",
        season=season, episode=episode, episode_end=episode_end,
        imdb_id=ids["imdb_id"], tmdb_id=ids["tmdb_id"], tvdb_id=ids["tvdb_id"], tvmaze_id=ids["tvmaze_id"],
        parse_confidence=confidence,
    )


def score_candidate(parsed: ParsedMediaName, candidate: dict, runtime_seconds: float | None = None) -> float:
    wanted = parsed.normalized_title
    names = [candidate.get("title", ""), candidate.get("original_title", "")]
    normalized_names = [normalize_title(name) for name in names if name]
    sequence = max((SequenceMatcher(None, wanted, name).ratio() for name in normalized_names), default=0.0)
    wanted_tokens = set(wanted.split())
    overlap = max((len(wanted_tokens & set(name.split())) / max(1, len(wanted_tokens | set(name.split()))) for name in normalized_names), default=0.0)
    title_score = max(sequence, overlap)
    score = title_score * 0.72
    candidate_year = candidate.get("year")
    if parsed.year and candidate_year:
        difference = abs(int(parsed.year) - int(candidate_year))
        score += 0.19 if difference == 0 else (0.07 if difference == 1 else -min(0.20, difference * 0.04))
    elif not parsed.year:
        score += 0.04
    if candidate.get("media_type") == parsed.media_type:
        score += 0.07
    else:
        score -= 0.11
    candidate_runtime = candidate.get("runtime_seconds")
    if runtime_seconds and candidate_runtime:
        delta = abs(float(runtime_seconds) - float(candidate_runtime)) / max(float(runtime_seconds), 1)
        score += 0.08 if delta <= 0.04 else (0.03 if delta <= 0.10 else -min(0.12, delta * 0.2))
    return round(max(0.0, min(1.0, score)), 4)


def parsed_identity(job: dict) -> ParsedMediaName:
    identity = job.get("media_identity") or {}
    fields = ParsedMediaName.__dataclass_fields__
    if all(name in identity for name in fields):
        return ParsedMediaName(**{name: identity[name] for name in fields})
    return parse_media_filename(job.get("filename") or "Untitled media")


class CatalogueClient:
    USER_AGENT = "Dubline/1.0 local media organiser"

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("TMDB_TOKEN", "").strip() or os.getenv("DUBLINE_TMDB_TOKEN", "").strip()
        self.api_key = os.getenv("DUBLINE_TMDB_API_KEY", "").strip()

    @property
    def tmdb_available(self) -> bool:
        return bool(self.token or self.api_key)

    def _tmdb_url(self, url: str) -> str:
        if self.token or not self.api_key:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{urllib.parse.urlencode({'api_key': self.api_key})}"

    def _json(self, url: str, *, bearer: bool = False) -> dict | list:
        headers = {"User-Agent": self.USER_AGENT, "Accept": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=9) as response:
                    if int(response.headers.get("Content-Length", "0") or 0) > 2_000_000:
                        raise RuntimeError("The catalogue response was unexpectedly large")
                    return json.loads(response.read(2_000_001))
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 2:
                    time.sleep(min(3, 0.75 * (attempt + 1)))
                    continue
                if exc.code in {401, 403}:
                    raise ValueError("TMDB rejected that API Read Access Token") from exc
                raise RuntimeError(f"The media catalogue returned HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt < 2:
                    time.sleep(0.45 * (attempt + 1))
                    continue
                raise RuntimeError("The media catalogue could not be reached. Your local projects still work.") from exc
        raise RuntimeError("The media catalogue could not be reached")

    def validate_tmdb(self, token: str) -> None:
        self.token = validate_tmdb_token(token)
        self._json("https://api.themoviedb.org/3/configuration", bearer=True)

    def tmdb_candidates(self, parsed: ParsedMediaName, query: str | None = None, media_type: str | None = None) -> list[dict]:
        if not self.tmdb_available:
            return []
        if (parsed.imdb_id or parsed.tvdb_id) and query is None:
            external_id = parsed.imdb_id or str(parsed.tvdb_id)
            source = "imdb_id" if parsed.imdb_id else "tvdb_id"
            url = "https://api.themoviedb.org/3/find/" + urllib.parse.quote(external_id) + "?external_source=" + source
            data = self._json(self._tmdb_url(url), bearer=bool(self.token))
            rows = [("movie", row) for row in data.get("movie_results", [])] + [("tv", row) for row in data.get("tv_results", [])]
        elif parsed.tmdb_id and query is None:
            kinds = [parsed.media_type]
            rows = []
            for kind in kinds:
                try:
                    endpoint = self._tmdb_url(f"https://api.themoviedb.org/3/{kind}/{parsed.tmdb_id}")
                    rows.append((kind, self._json(endpoint, bearer=bool(self.token))))
                except RuntimeError:
                    pass
        else:
            search = query or parsed.title
            kinds = [media_type] if media_type in {"movie", "tv"} else ([parsed.media_type, "tv" if parsed.media_type == "movie" else "movie"])
            rows = []
            for kind in kinds:
                params = {"query": search, "include_adult": "false", "language": "en-GB"}
                if parsed.year and not query:
                    params["year" if kind == "movie" else "first_air_date_year"] = str(parsed.year)
                endpoint = self._tmdb_url(f"https://api.themoviedb.org/3/search/{kind}?{urllib.parse.urlencode(params)}")
                data = self._json(endpoint, bearer=bool(self.token))
                rows.extend((kind, row) for row in data.get("results", [])[:10])
        candidates = []
        for kind, row in rows:
            date = row.get("release_date") if kind == "movie" else row.get("first_air_date")
            candidates.append({
                "provider": "tmdb", "id": int(row["id"]), "media_type": kind,
                "title": row.get("title") or row.get("name") or "Untitled",
                "original_title": row.get("original_title") or row.get("original_name"),
                "year": int(date[:4]) if date and re.match(r"^\d{4}", date) else None,
                "poster_url": f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if row.get("poster_path") else None,
                "overview": (row.get("overview") or "")[:600], "popularity": float(row.get("popularity") or 0),
            })
        return candidates

    def tvmaze_candidates(self, parsed: ParsedMediaName, query: str | None = None) -> list[dict]:
        if parsed.media_type != "tv" and not query:
            return []
        exact = bool(parsed.tvmaze_id and not query)
        url = f"https://api.tvmaze.com/shows/{parsed.tvmaze_id}" if exact else "https://api.tvmaze.com/search/shows?" + urllib.parse.urlencode({"q": query or parsed.title})
        response = self._json(url)
        data = [{"score": 1.0, "show": response}] if exact else response
        candidates = []
        for result in data[:8]:
            row = result.get("show", {})
            premiered = row.get("premiered") or ""
            image = row.get("image") or {}
            candidates.append({
                "provider": "tvmaze", "id": int(row["id"]), "media_type": "tv",
                "title": row.get("name") or "Untitled", "original_title": None,
                "year": int(premiered[:4]) if re.match(r"^\d{4}", premiered) else None,
                "poster_url": image.get("original") or image.get("medium"),
                "overview": re.sub(r"<[^>]+>", "", row.get("summary") or "")[:600],
                "popularity": float(result.get("score") or 0), "site_url": row.get("url"),
            })
        return candidates

    def candidate_by_id(self, provider: str, media_type: str, external_id: int) -> dict | None:
        if provider == "tvmaze":
            data = self._json(f"https://api.tvmaze.com/shows/{external_id}")
            premiered = data.get("premiered") or ""
            image = data.get("image") or {}
            return {
                "provider": "tvmaze", "id": int(data["id"]), "media_type": "tv",
                "title": data.get("name") or "Untitled", "original_title": None,
                "year": int(premiered[:4]) if re.match(r"^\d{4}", premiered) else None,
                "poster_url": image.get("original") or image.get("medium"),
                "overview": re.sub(r"<[^>]+>", "", data.get("summary") or "")[:600],
                "popularity": 0.0, "site_url": data.get("url"),
            }
        if provider != "tmdb" or not self.tmdb_available:
            return None
        endpoint = self._tmdb_url(f"https://api.themoviedb.org/3/{media_type}/{external_id}?language=en-GB")
        row = self._json(endpoint, bearer=bool(self.token))
        date = row.get("release_date") if media_type == "movie" else row.get("first_air_date")
        return {
            "provider": "tmdb", "id": int(row["id"]), "media_type": media_type,
            "title": row.get("title") or row.get("name") or "Untitled",
            "original_title": row.get("original_title") or row.get("original_name"),
            "year": int(date[:4]) if date and re.match(r"^\d{4}", date) else None,
            "poster_url": f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if row.get("poster_path") else None,
            "overview": (row.get("overview") or "")[:600], "popularity": float(row.get("popularity") or 0),
        }


class MediaLibrary:
    def __init__(self, store: JobStore, data: Path):
        self.store = store
        self.artwork = data / "artwork"
        self.artwork.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._queued: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="media-library", daemon=True)
        self._thread.start()
        for job in self.store.list_summaries(1000):
            if not job.get("project_id") or not (job.get("media_identity") or {}).get("matched"):
                full = self.store.get(job["id"]) or job
                source = Path(full["input_path"]) if full.get("input_path") and Path(full["input_path"]).is_file() else None
                self.attach(job["id"], job.get("filename") or "Untitled media", source)
            self.schedule(job["id"])

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=4)

    def attach(self, job_id: str, filename: str, source_path: Path | None = None) -> dict:
        parsed = parse_media_filename(filename)
        metadata_source = "filename"
        if source_path:
            parsed_from_folder = self._prefer_media_folder(Path(source_path), parsed)
            if parsed_from_folder != parsed:
                parsed = parsed_from_folder
                metadata_source = "folder"
            local = self._read_local_nfo(source_path, parsed)
            if local:
                parsed = local
                metadata_source = "nfo"
        provider, external_id = self._external_identity(parsed)
        external_key = f"{provider}:{parsed.media_type}:{external_id}" if provider and external_id else None
        lookup_key = self._compatible_lookup_key(parsed)
        project = self.store.resolve_project(lookup_key, {
            "title": parsed.title, "year": parsed.year, "media_type": parsed.media_type,
            "provider": provider, "external_id": external_id,
            "metadata_source": metadata_source,
        }, external_key=external_key)
        if source_path and self._cache_local_artwork(project["id"], source_path):
            project = self.store.update_project(project["id"], poster_ready=True, artwork_source="local")
        already_known = bool(project.get("catalogue_matched"))
        identity = {**asdict(parsed), "matched": already_known, "needs_confirmation": False,
                    "match_confidence": 1.0 if already_known else None, "candidates": [],
                    "provider": project.get("provider"), "external_id": project.get("external_id")}
        self.store.assign_project(job_id, project["id"], identity)
        if not already_known:
            self.schedule(job_id)
        return project

    def _compatible_lookup_key(self, parsed: ParsedMediaName) -> str:
        """Join a yearless task to one unambiguous same-title project, never across known remakes."""
        matches = []
        for project in self.store.list_projects(1000):
            if project.get("media_type") != parsed.media_type:
                continue
            if normalize_title(project.get("title") or "") != parsed.normalized_title:
                continue
            project_year = project.get("year")
            if parsed.year and project_year and int(parsed.year) != int(project_year):
                continue
            matches.append(project)
        return matches[0]["lookup_key"] if len(matches) == 1 else parsed.lookup_key

    @staticmethod
    def _prefer_media_folder(source: Path, parsed: ParsedMediaName) -> ParsedMediaName:
        generic = {"movie", "film", "video", "title", "episode", "sample", "feature", "source", "s01e01"}
        parent = source.parent
        if re.fullmatch(r"(?i)season[ ._-]*\d+|s\d{1,2}", parent.name):
            parent = parent.parent
        folder = parse_media_filename(parent.name)
        has_folder_evidence = bool(folder.year or folder.tmdb_id or folder.tvdb_id or folder.tvmaze_id or folder.imdb_id)
        weak_filename = parsed.normalized_title in generic or parsed.normalized_title.startswith("s01e")
        container_folder = folder.normalized_title in {"downloads", "download", "movies", "films", "videos", "tv", "tv shows", "media"}
        if weak_filename and container_folder:
            return parsed
        if not weak_filename and not (parsed.media_type == "tv" and has_folder_evidence):
            return parsed
        return replace(
            folder, media_type=parsed.media_type, season=parsed.season, episode=parsed.episode,
            episode_end=parsed.episode_end, imdb_id=parsed.imdb_id or folder.imdb_id,
            tmdb_id=parsed.tmdb_id or folder.tmdb_id, tvdb_id=parsed.tvdb_id or folder.tvdb_id,
            tvmaze_id=parsed.tvmaze_id or folder.tvmaze_id,
        )

    @staticmethod
    def _external_identity(parsed: ParsedMediaName) -> tuple[str | None, int | str | None]:
        if parsed.tmdb_id:
            return "tmdb", parsed.tmdb_id
        if parsed.tvmaze_id:
            return "tvmaze", parsed.tvmaze_id
        if parsed.imdb_id:
            return "imdb", parsed.imdb_id
        if parsed.tvdb_id:
            return "tvdb", parsed.tvdb_id
        return None, None

    def _read_local_nfo(self, source_path: Path, parsed: ParsedMediaName) -> ParsedMediaName | None:
        source = Path(source_path)
        candidates = [source.with_suffix(".nfo"), source.parent / "movie.nfo", source.parent / "tvshow.nfo"]
        nfo = next((item for item in candidates if item.is_file() and item.stat().st_size <= 2 * 1024 * 1024), None)
        if not nfo:
            return None
        try:
            raw = nfo.read_bytes()
            if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                return None
            root = ET.fromstring(raw)
        except (OSError, ET.ParseError, UnicodeError):
            return None
        root_name = root.tag.lower().rsplit("}", 1)[-1]
        media_type = "tv" if root_name in {"tvshow", "episodedetails"} or root.find("showtitle") is not None else parsed.media_type
        title = (root.findtext("showtitle") if media_type == "tv" else None) or root.findtext("title") or parsed.title
        year_text = root.findtext("year") or root.findtext("premiered") or ""
        year_match = re.search(r"(?:18|19|20)\d{2}", year_text)
        values: dict[str, Any] = {}
        for element in root.findall(".//uniqueid"):
            kind, value = (element.get("type") or "").lower(), (element.text or "").strip()
            if kind == "imdb" and re.fullmatch(r"tt\d{7,10}", value, re.I):
                values["imdb_id"] = value.lower()
            elif kind in {"tmdb", "tvdb", "tvmaze"} and value.isdigit():
                values[f"{kind}_id"] = int(value)
        for tag, field in (("tmdbid", "tmdb_id"), ("tvdbid", "tvdb_id"), ("imdbid", "imdb_id")):
            value = (root.findtext(tag) or "").strip()
            if field == "imdb_id" and re.fullmatch(r"tt\d{7,10}", value, re.I):
                values[field] = value.lower()
            elif field != "imdb_id" and value.isdigit():
                values[field] = int(value)
        legacy_id = (root.findtext("id") or "").strip()
        if re.fullmatch(r"tt\d{7,10}", legacy_id, re.I):
            values.setdefault("imdb_id", legacy_id.lower())
        return replace(
            parsed, title=_SPACE.sub(" ", title).strip()[:240], normalized_title=normalize_title(title),
            year=int(year_match.group()) if year_match else parsed.year, media_type=media_type, **values,
        )

    def _cache_local_artwork(self, project_id: str, source_path: Path) -> bool:
        source = Path(source_path)
        stems = (f"{source.stem}-poster", "poster", "folder")
        candidates = [source.parent / f"{stem}{suffix}" for stem in stems for suffix in (".jpg", ".jpeg", ".png", ".webp")]
        artwork = next((item for item in candidates if item.is_file() and 256 <= item.stat().st_size <= 12 * 1024 * 1024), None)
        if not artwork:
            return False
        try:
            body = artwork.read_bytes()
        except OSError:
            return False
        image_type = None
        if body.startswith(b"\xff\xd8\xff"):
            image_type = ".jpg"
        elif body.startswith(b"\x89PNG\r\n\x1a\n"):
            image_type = ".png"
        elif body[:4] == b"RIFF" and body[8:12] == b"WEBP":
            image_type = ".webp"
        if not image_type:
            return False
        for old in self.artwork.glob(f"{project_id}.*"):
            old.unlink(missing_ok=True)
        temporary = self.artwork / f".{project_id}{image_type}.part"
        temporary.write_bytes(body)
        temporary.replace(self.artwork / f"{project_id}{image_type}")
        return True

    def schedule(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._queued:
                return
            self._queued.add(job_id)
        self._queue.put(job_id)

    def _run(self) -> None:
        while not self._stopping.is_set():
            job_id = self._queue.get()
            if job_id is None:
                break
            try:
                self.enrich(job_id)
            except Exception as exc:  # Lookup failures must never disrupt dubbing.
                job = self.store.get(job_id)
                if job:
                    identity = {**job.get("media_identity", {}), "lookup_error": str(exc)[:240]}
                    self.store.update(job_id, media_identity=identity)
            finally:
                with self._lock:
                    self._queued.discard(job_id)

    def enrich(self, job_id: str) -> None:
        settings = media_lookup_summary()
        if not settings["enabled"]:
            return
        job = self.store.get(job_id)
        if not job:
            return
        if (job.get("media_identity") or {}).get("matched"):
            return
        parsed = parsed_identity(job)
        client = CatalogueClient()
        candidates = client.tmdb_candidates(parsed)
        if parsed.media_type == "tv" and not candidates:
            candidates = client.tvmaze_candidates(parsed)
        runtime = (job.get("media") or {}).get("duration")
        ranked = []
        for candidate in candidates:
            candidate = {**candidate, "score": score_candidate(parsed, candidate, runtime)}
            ranked.append(candidate)
        ranked.sort(key=lambda item: (item["score"], item.get("popularity", 0)), reverse=True)
        best = ranked[0] if ranked else None
        runner_up = ranked[1]["score"] if len(ranked) > 1 else 0
        exact_id = bool(parsed.imdb_id or parsed.tmdb_id or parsed.tvdb_id or parsed.tvmaze_id)
        accepted = bool(best and (exact_id or (best["score"] >= 0.78 and best["score"] - runner_up >= 0.07)))
        if accepted:
            self.apply_candidate(job_id, best, automatic=True)
            return
        identity = {**job.get("media_identity", asdict(parsed)), "matched": False,
                    "needs_confirmation": bool(ranked), "match_confidence": best["score"] if best else None,
                    "candidates": [self.public_candidate(item) for item in ranked[:5]], "lookup_error": None}
        self.store.update(job_id, media_identity=identity)

    def search(self, query: str, media_type: str | None = None, year: int | None = None) -> list[dict]:
        settings = media_lookup_summary()
        if not settings["enabled"]:
            raise ValueError("Connect TMDB in Setup to search films and television")
        query = _SPACE.sub(" ", query).strip()[:160]
        if len(query) < 2:
            raise ValueError("Enter at least two characters")
        parsed = ParsedMediaName(
            source=query, title=query, normalized_title=normalize_title(query), year=year,
            media_type=media_type or "movie", season=None, episode=None, episode_end=None,
            imdb_id=None, tmdb_id=None, tvdb_id=None, tvmaze_id=None, parse_confidence=1.0,
        )
        client = CatalogueClient()
        candidates = client.tmdb_candidates(parsed, query=query, media_type=media_type)
        if media_type == "tv" and not candidates:
            candidates = client.tvmaze_candidates(parsed, query=query)
        if media_type != "tv" and not candidates and not client.tmdb_available:
            raise ValueError("Film catalogue access is not included in this development build yet")
        for candidate in candidates:
            candidate["score"] = score_candidate(parsed, candidate)
        candidates.sort(key=lambda item: (item["score"], item.get("popularity", 0)), reverse=True)
        return [self.public_candidate(item) for item in candidates[:12]]

    def apply_candidate(self, job_id: str, candidate: dict, automatic: bool = False) -> dict:
        job = self.store.get(job_id)
        if not job:
            raise KeyError(job_id)
        parsed = parsed_identity(job)
        provider = candidate["provider"]
        external_key = f"{provider}:{candidate['media_type']}:{int(candidate['id'])}"
        existing = self.store.get_project(job.get("project_id", "")) or {}
        local_metadata = existing.get("metadata_source") == "nfo"
        project = self.store.resolve_project(parsed.lookup_key, {
            "title": existing.get("title") if local_metadata else candidate["title"],
            "year": existing.get("year") if local_metadata else candidate.get("year"),
            "media_type": candidate["media_type"], "provider": provider,
            "external_id": int(candidate["id"]), "overview": candidate.get("overview") or "",
            "site_url": candidate.get("site_url"), "catalogue_matched": True,
            "metadata_source": "nfo" if local_metadata else provider,
        }, external_key=external_key)
        local_artwork = project.get("poster_ready") and project.get("artwork_source") == "local"
        poster_ready = bool(local_artwork) or self._download_poster(project["id"], candidate.get("poster_url"))
        if poster_ready:
            project = self.store.update_project(project["id"], poster_ready=True,
                                                artwork_source="local" if local_artwork else provider)
        identity = {**asdict(parsed), "matched": True, "needs_confirmation": False,
                    "match_confidence": candidate.get("score"), "automatic": automatic,
                    "provider": provider, "external_id": int(candidate["id"]), "candidates": [], "lookup_error": None}
        self.store.assign_project(job_id, project["id"], identity)
        return project

    def choose(self, job_id: str, provider: str, media_type: str, external_id: int) -> dict:
        if provider not in {"tmdb", "tvmaze"} or media_type not in {"movie", "tv"}:
            raise ValueError("That media match is not valid")
        job = self.store.get(job_id)
        if not job:
            raise KeyError(job_id)
        identity = job.get("media_identity") or {}
        candidates = identity.get("candidates") or []
        candidate = next((item for item in candidates if item.get("provider") == provider and
                          item.get("media_type") == media_type and int(item.get("id", -1)) == external_id), None)
        if candidate is None:
            client = CatalogueClient()
            candidate = client.candidate_by_id(provider, media_type, external_id)
        if candidate is None:
            raise ValueError("That catalogue result could not be verified")
        return self.apply_candidate(job_id, candidate, automatic=False)

    @staticmethod
    def public_candidate(candidate: dict) -> dict:
        return {key: candidate.get(key) for key in (
            "provider", "id", "media_type", "title", "original_title", "year", "poster_url",
            "overview", "site_url", "score",
        )}

    def _download_poster(self, project_id: str, url: str | None) -> bool:
        if not url:
            return False
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in {"image.tmdb.org", "static.tvmaze.com"}:
            return False
        request = urllib.request.Request(url, headers={"User-Agent": CatalogueClient.USER_AGENT, "Accept": "image/jpeg,image/png,image/webp"})
        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                content_type = response.headers.get_content_type().lower()
                if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                    return False
                if int(response.headers.get("Content-Length", "0") or 0) > 8 * 1024 * 1024:
                    return False
                body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024 or len(body) < 256:
                return False
            suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
            for old in self.artwork.glob(f"{project_id}.*"):
                old.unlink(missing_ok=True)
            temporary = self.artwork / f".{project_id}{suffix}.part"
            temporary.write_bytes(body)
            temporary.replace(self.artwork / f"{project_id}{suffix}")
            return True
        except (OSError, urllib.error.URLError):
            return False

    def poster_path(self, project_id: str) -> Path | None:
        if not re.fullmatch(r"p[a-f0-9]{15}", project_id):
            return None
        for suffix in (".jpg", ".png", ".webp"):
            candidate = (self.artwork / f"{project_id}{suffix}").resolve()
            if candidate.is_file() and self.artwork.resolve() in candidate.parents:
                return candidate
        return None

    def save_tmdb_token(self, token: str) -> dict:
        client = CatalogueClient(token)
        client.validate_tmdb(token)
        save_setting("TMDB_TOKEN", client.token)
        save_setting("MEDIA_LOOKUP_ENABLED", "1")
        for job in self.store.list_summaries(1000):
            self.schedule(job["id"])
        return media_lookup_summary()

    def set_enabled(self, enabled: bool) -> dict:
        save_setting("MEDIA_LOOKUP_ENABLED", "1" if enabled else "0")
        if enabled:
            for job in self.store.list_summaries(1000):
                self.schedule(job["id"])
        return media_lookup_summary()
