"""
skill_match_engine.py

Implements the weighted skill-proficiency rubric and match-percentage
calculation against a normalized job-skill CSV pool.

-------------------------------------------------------------------------
JOB CSV SCHEMA (wide format, one row per job)
-------------------------------------------------------------------------
job_id       : unique id for the job posting
job_title    : normalized title, e.g. "Data Analyst"
skills       : stringified Python list of extracted skills, e.g.
               "['Azure', 'Data Integration', 'SQL', 'PySpark',
                 'Dynamics 365', 'CI/CD', 'Documentation']"

Example rows:
job_id,job_title,skills
J1001,Data Analyst,"['SQL', 'Python', 'Tableau']"
J1002,Data Analyst,"['SQL', 'Excel']"
J1003,Data Engineer,"['Azure', 'PySpark', 'CI/CD']"

(A long format -- one row per skill per job, with a plain "skill" column
-- is also supported and auto-detected; see JobSkillPool._load below.)

Skill "importance" per job_title is derived automatically: it's the
fraction of jobs (with that title) that require the skill.
e.g. if SQL appears in 90/100 "Data Analyst" postings -> importance 0.9

-------------------------------------------------------------------------
CANDIDATE INPUT SCHEMA (Python objects, not CSV -- typically built from
resume/profile extraction upstream)
-------------------------------------------------------------------------
Each candidate skill entry needs:
    skill           : str, normalized skill name
    context         : "list" | "mentioned" | "central"
                        - list      -> only appears in a skills list/tags
                        - mentioned -> appears in a project/experience description
                        - central   -> central to a project title/role description
    duration_months : int or None  -> time spent using the skill on that evidence
    role             : "contributor" | "lead" | None
    academic_year    : int or None -> year of study at time of the evidence
                        (e.g. 1, 2, 3, 4; use highest completed year if unknown)
    months_since_used: int or None -> recency of last evidenced use

If a candidate has multiple entries for the same skill (e.g. used in two
projects), the MAX skill score across entries is taken (prevents
resume-stuffing from inflating scores by repetition).
-------------------------------------------------------------------------
NORMALIZATION
-------------------------------------------------------------------------
Both job-pool skills and candidate skills pass through normalize_skill()
before scoring/matching, so string variants collapse to one canonical
key:
    1. Alias lookup  - "GCP" / "Google Cloud" -> "google cloud platform"
                        (edit SKILL_ALIASES below to extend this map)
    2. Fuzzy fallback - if a required job skill has no exact/alias match
                        against the candidate's skills, compute_match()
                        tries a fuzzy string match (difflib) before
                        calling it a gap. This catches near-duplicates
                        that weren't anticipated in the alias map (e.g.
                        "Dynamics 365 CRM" vs "Dynamics 365"). Any fuzzy
                        match is flagged in SkillMatchDetail.fuzzy_matched
                        so it stays visible/auditable rather than silent.
-------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import csv
import ast
import difflib
from collections import defaultdict


# =========================================================================
# 0. SKILL NORMALIZATION
# =========================================================================

# Extend this map as you discover new variants in your data. Keys and
# values should both be lowercase; normalize_skill() lowercases input
# before lookup, so casing in the raw data doesn't matter.
SKILL_ALIASES: Dict[str, str] = {
    # cloud / platforms
    "gcp": "google cloud platform",
    "google cloud": "google cloud platform",
    "aws": "amazon web services",
    "amazon web services (aws)": "amazon web services",
    "ms azure": "azure",
    "microsoft azure": "azure",
    "azure cloud": "azure",
    "azure devops": "azure devops",
    # dynamics
    "dynamics365": "dynamics 365",
    "ms dynamics 365": "dynamics 365",
    "microsoft dynamics 365": "dynamics 365",
    "dynamics 365 crm": "dynamics 365",
    # office / excel
    "ms excel": "excel",
    "microsoft excel": "excel",
    "excel (advanced)": "excel",
    "advanced excel": "excel",
    # python variants
    "python (pandas)": "python",
    "python programming": "python",
    "pandas/python": "python",
    # ci/cd
    "cicd": "ci/cd",
    "ci-cd": "ci/cd",
    "continuous integration/continuous deployment": "ci/cd",
    "continuous integration / continuous delivery": "ci/cd",
    # sql
    "structured query language": "sql",
    "ms sql": "sql",
    "t-sql": "sql",
    "tsql": "sql",
    "sql server": "sql",
    # data integration / ETL
    "data integration & etl": "data integration",
    "etl": "data integration",
    "etl pipelines": "data integration",
    # docs
    "technical documentation": "documentation",
    "documentation writing": "documentation",
    # ML / AI
    "ml": "machine learning",
    "machine learning (ml)": "machine learning",
    "llm": "large language models",
    "llms": "large language models",
    "prompt eng": "prompt engineering",
}


def normalize_skill(raw: str) -> str:
    """
    Canonicalizes a raw skill string so variants collapse to one key:
      1. lowercase + collapse internal whitespace
      2. strip stray leading/trailing punctuation
      3. map through SKILL_ALIASES if a known variant

    Applied to BOTH job-pool skills and candidate skills so e.g. "GCP",
    "Google Cloud", and "Google Cloud Platform" all resolve to the same
    canonical key before importance weighting / matching.
    """
    if not raw:
        return ""
    s = " ".join(str(raw).strip().lower().split())
    s = s.strip(" .,;:")
    return SKILL_ALIASES.get(s, s)


# =========================================================================
# 1. CANDIDATE SKILL SCORING
# =========================================================================

@dataclass
class CandidateSkillEvidence:
    skill: str
    context: str                      # "list" | "mentioned" | "central"
    duration_months: Optional[int] = None
    role: Optional[str] = None        # "contributor" | "lead"
    academic_year: Optional[int] = None
    months_since_used: Optional[int] = None


BASE_PRESENCE = {
    "list": 0.4,
    "mentioned": 0.6,
    "central": 0.8,
}

FINAL_YEAR_THRESHOLD = 3   # academic_year >= this counts as "senior"


def depth_multiplier(ev: CandidateSkillEvidence) -> float:
    """
    Depth multiplier captures 'time spent / ownership' signal.
    Base 1.0, up to +0.15 duration, +0.15 leadership, +0.10 seniority.
    Missing data -> neutral (no bonus, no penalty).
    Capped at 1.4.
    """
    bonus = 0.0

    if ev.duration_months is not None and ev.duration_months >= 4:
        bonus += 0.15

    if ev.role is not None and ev.role.lower() in ("lead", "led", "owner", "founder"):
        bonus += 0.15

    if ev.academic_year is not None and ev.academic_year >= FINAL_YEAR_THRESHOLD:
        bonus += 0.10

    return min(1.0 + bonus, 1.4)


def recency_multiplier(ev: CandidateSkillEvidence) -> float:
    """
    Skills atrophy in relevance over time. Missing data -> assume recent (1.0),
    don't penalize for lack of a date.
    """
    if ev.months_since_used is None:
        return 1.0
    if ev.months_since_used <= 12:
        return 1.0
    if ev.months_since_used <= 24:
        return 0.85
    return 0.7


def score_single_evidence(ev: CandidateSkillEvidence) -> float:
    base = BASE_PRESENCE.get(ev.context, 0.4)
    score = base * depth_multiplier(ev) * recency_multiplier(ev)
    return min(score, 1.0)


def compute_candidate_skill_scores(
    evidences: List[CandidateSkillEvidence],
) -> Dict[str, float]:
    """
    Collapses multiple evidence entries per skill into a single score
    per skill, taking the MAX across entries.
    Returns: { normalized_skill_name: score (0.0 - 1.0) }
    """
    scores: Dict[str, float] = {}
    for ev in evidences:
        s = score_single_evidence(ev)
        key = ev.skill.strip().lower()
        scores[key] = max(scores.get(key, 0.0), s)
    return scores


# =========================================================================
# 2. JOB POOL LOADING + IMPORTANCE WEIGHTS
# =========================================================================

class JobSkillPool:
    """
    Loads the normalized job-skill CSV and computes, per job_title,
    the importance weight of each skill = (# jobs with that title
    requiring the skill) / (# jobs with that title).
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        # job_title -> set of job_ids seen
        self._title_job_ids: Dict[str, set] = defaultdict(set)
        # job_title -> skill -> set of job_ids requiring it
        self._title_skill_job_ids: Dict[str, Dict[str, set]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._load()

    @staticmethod
    def _parse_skill_list(raw: str) -> List[str]:
        """
        Parses a stringified Python list of skills, e.g.
        "['Azure', 'Data Integration', 'SQL']" -> ['Azure', 'Data Integration', 'SQL']

        Falls back to comma-splitting if the value isn't a valid literal
        list (e.g. "Azure, SQL, PySpark" without brackets/quotes).
        """
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple, set)):
                return [str(s).strip() for s in parsed if str(s).strip()]
            # single scalar value wrapped in quotes, e.g. "'SQL'"
            return [str(parsed).strip()]
        except (ValueError, SyntaxError):
            # not a valid literal -- treat as a plain delimited string
            stripped = raw.strip("[]")
            return [s.strip().strip("'\"") for s in stripped.split(",") if s.strip()]

    def _load(self):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])

            base_required = {"job_id", "job_title"}
            if not base_required.issubset(fieldnames):
                raise ValueError(
                    f"CSV must contain columns {base_required}, "
                    f"found {reader.fieldnames}"
                )

            has_skills_list_col = "skills" in fieldnames
            has_single_skill_col = "skill" in fieldnames

            if not (has_skills_list_col or has_single_skill_col):
                raise ValueError(
                    "CSV must contain either a 'skills' column (list per job, "
                    "wide format) or a 'skill' column (one row per skill, "
                    f"long format). Found columns: {reader.fieldnames}"
                )

            for row in reader:
                job_id = row["job_id"].strip()
                title = row["job_title"].strip().lower()
                if not job_id or not title:
                    continue

                if has_skills_list_col:
                    skills = self._parse_skill_list(row["skills"])
                else:
                    skills = [row["skill"].strip()]

                self._title_job_ids[title].add(job_id)
                for skill in skills:
                    skill = skill.strip().lower()
                    if not skill:
                        continue
                    self._title_skill_job_ids[title][skill].add(job_id)

    def titles(self) -> List[str]:
        return list(self._title_job_ids.keys())

    def skill_importance(self, job_title: str) -> Dict[str, float]:
        """
        Returns { skill: importance_weight } for a given job_title,
        where importance = fraction of postings of that title requiring
        the skill (0.0 - 1.0).
        """
        title = job_title.strip().lower()
        total_jobs = len(self._title_job_ids.get(title, set()))
        if total_jobs == 0:
            return {}
        importance = {}
        for skill, job_ids in self._title_skill_job_ids[title].items():
            importance[skill] = len(job_ids) / total_jobs
        return importance


# =========================================================================
# 3. MATCHING
# =========================================================================

CRITICAL_THRESHOLD = 0.8  # required by >=80% of jobs for the title


@dataclass
class SkillMatchDetail:
    skill: str
    importance: float
    candidate_score: float   # 0 if candidate doesn't have it
    is_gap: bool
    is_critical: bool


@dataclass
class MatchResult:
    job_title: str
    match_percent: float
    details: List[SkillMatchDetail]

    def gaps(self) -> List[SkillMatchDetail]:
        return [d for d in self.details if d.is_gap]

    def matching_skills(self) -> List[SkillMatchDetail]:
        return [d for d in self.details if not d.is_gap]


def compute_match(
    job_title: str,
    candidate_scores: Dict[str, float],
    pool: JobSkillPool,
) -> MatchResult:
    importance = pool.skill_importance(job_title)

    numerator = 0.0
    denominator = 0.0
    details: List[SkillMatchDetail] = []

    for skill, weight in importance.items():
        cand_score = candidate_scores.get(skill, 0.0)
        numerator += cand_score * weight
        denominator += weight

        details.append(
            SkillMatchDetail(
                skill=skill,
                importance=round(weight, 3),
                candidate_score=round(cand_score, 3),
                is_gap=(cand_score == 0.0),
                is_critical=(weight >= CRITICAL_THRESHOLD),
            )
        )

    match_percent = (numerator / denominator * 100) if denominator > 0 else 0.0

    # sort: gaps first (critical gaps first), then matching skills by importance
    details.sort(key=lambda d: (not d.is_gap, not d.is_critical, -d.importance))

    return MatchResult(
        job_title=job_title,
        match_percent=round(match_percent, 1),
        details=details,
    )


def rank_job_titles_for_candidate(
    candidate_scores: Dict[str, float],
    pool: JobSkillPool,
    top_n: int = 3,
) -> List[MatchResult]:
    results = [
        compute_match(title, candidate_scores, pool) for title in pool.titles()
    ]
    results.sort(key=lambda r: r.match_percent, reverse=True)
    return results[:top_n]


# =========================================================================
# 4. EXAMPLE USAGE
# =========================================================================

if __name__ == "__main__":
    # --- Example candidate: Sarah, 2nd-year student ---
    sarah_evidence = [
        CandidateSkillEvidence(
            skill="SQL", context="central", duration_months=8,
            role="contributor", academic_year=2, months_since_used=2,
        ),
        CandidateSkillEvidence(
            skill="Excel", context="list", duration_months=None,
            role=None, academic_year=2, months_since_used=1,
        ),
        CandidateSkillEvidence(
            skill="Python", context="mentioned", duration_months=6,
            role="lead", academic_year=2, months_since_used=1,
        ),
        CandidateSkillEvidence(
            skill="Tableau", context="central", duration_months=5,
            role="lead", academic_year=2, months_since_used=1,
        ),
    ]

    scores = compute_candidate_skill_scores(sarah_evidence)
    print("Candidate skill scores:")
    for skill, score in scores.items():
        print(f"  {skill}: {score:.2f}")

    # --- Load job pool ---
    # Expects a CSV at this path with columns: job_id, job_title, skill
    pool = JobSkillPool("jobs_skill_dataset.csv")

    top_matches = rank_job_titles_for_candidate(scores, pool, top_n=3)

    print("\nTop job title matches:")
    for result in top_matches:
        print(f"\n{result.job_title.title()} — {result.match_percent}% match")
        print("  Gaps:")
        for d in result.gaps():
            tag = " (Critical)" if d.is_critical else ""
            print(f"    - {d.skill}{tag} [importance {d.importance}]")
        print("  Matching skills:")
        for d in result.matching_skills():
            print(f"    - {d.skill} [score {d.candidate_score}, importance {d.importance}]")