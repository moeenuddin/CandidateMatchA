import os
import csv
import pandas as pd
import streamlit as st

# Import classes and methods from your engine script
from skill_match_engine import (
    CandidateSkillEvidence,
    JobSkillPool,
    compute_candidate_skill_scores,
    rank_job_titles_for_candidate,
    normalize_skill,
)

# -----------------------------------------------------------------------------
# CONFIG & MOCK DATA GENERATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Candidate Assessment & Job Matcher",
    page_icon="🎯",
    layout="wide"
)

CSV_FILE = "jobs_skill_dataset.csv"

def ensure_sample_csv():
    """Generates a sample CSV dataset if none exists."""
    if not os.path.exists(CSV_FILE):
        data = [
            ["job_id", "job_title", "skills"],
            ["J1001", "Data Analyst", "['SQL', 'Python', 'Tableau', 'Excel']"],
            ["J1002", "Data Analyst", "['SQL', 'Excel', 'Power BI']"],
            ["J1003", "Data Analyst", "['SQL', 'Python', 'Documentation']"],
            ["J1004", "Data Engineer", "['SQL', 'PySpark', 'Azure', 'CI/CD']"],
            ["J1005", "Data Engineer", "['Python', 'PySpark', 'SQL', 'Data Integration']"],
            ["J1006", "Data Engineer", "['Azure', 'CI/CD', 'Docker']"],
            ["J1007", "Machine Learning Engineer", "['Python', 'Machine Learning', 'SQL', 'PyTorch']"],
            ["J1008", "Machine Learning Engineer", "['Python', 'Large Language Models', 'Prompt Engineering']"],
        ]
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)

ensure_sample_csv()

@st.cache_resource
def load_pool():
    return JobSkillPool(CSV_FILE)

try:
    pool = load_pool()
except Exception as e:
    st.error(f"Failed to load job pool CSV: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "evidences" not in st.session_state:
    st.session_state.evidences = [
        # Default starter skills for demonstration
        CandidateSkillEvidence(
            skill="SQL", context="central", duration_months=8,
            role="contributor", academic_year=2, months_since_used=2
        ),
        CandidateSkillEvidence(
            skill="Python", context="mentioned", duration_months=6,
            role="lead", academic_year=2, months_since_used=1
        ),
    ]

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
st.title("🎯 Student Skill Assessment & Job Matching")
st.markdown("Assess your skill proficiency based on practical evidence and match against target role titles.")

col_left, col_right = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: ASSESSMENTS ENTRY ---
with col_left:
    st.header("1. Candidate Assessment")

    with st.expander("➕ Add New Skill Evidence", expanded=True):
        with st.form("add_skill_form", clear_on_submit=True):
            skill_input = st.text_input("Skill Name", placeholder="e.g. SQL, GCP, PySpark")
            
            f1, f2 = st.columns(2)
            context_val = f1.selectbox(
                "Usage Context", 
                ["central", "mentioned", "list"],
                help="central: Core to project/role | mentioned: Used in project description | list: Listed in resume tags"
            )
            role_val = f2.selectbox("Role Taken", ["contributor", "lead"])

            f3, f4, f5 = st.columns(3)
            duration_val = f3.number_input("Duration (Months)", min_value=1, max_value=120, value=6)
            year_val = f4.number_input("Academic Year", min_value=1, max_value=6, value=2)
            recency_val = f5.number_input("Months Since Last Used", min_value=0, max_value=60, value=1)

            submitted = st.form_submit_button("Add Skill Evidence")
            if submitted:
                if skill_input.strip():
                    new_ev = CandidateSkillEvidence(
                        skill=skill_input.strip(),
                        context=context_val,
                        duration_months=int(duration_val),
                        role=role_val,
                        academic_year=int(year_val),
                        months_since_used=int(recency_val)
                    )
                    st.session_state.evidences.append(new_ev)
                    st.success(f"Added skill evidence for '{skill_input.strip()}'!")
                    st.rerun()
                else:
                    st.warning("Please enter a valid skill name.")

    st.subheader("Logged Skill Evidences")
    if not st.session_state.evidences:
        st.info("No skill evidences added yet. Use the form above.")
    else:
        for idx, ev in enumerate(st.session_state.evidences):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(
                    f"**{ev.skill}** — *{ev.context}* | {ev.duration_months or 0} mos | "
                    f"Role: {ev.role} | Year: {ev.academic_year} | {ev.months_since_used or 0} mos ago"
                )
            with c2:
                if st.button("Delete", key=f"del_{idx}"):
                    st.session_state.evidences.pop(idx)
                    st.rerun()

# --- RIGHT COLUMN: COMPUTED MATCHES ---
with col_right:
    st.header("2. Matching Results")

    if not st.session_state.evidences:
        st.info("Add candidate skills to compute job matches.")
    else:
        # Pre-process raw skill inputs using engine's normalize_skill function
        normalized_evidences = []
        for ev in st.session_state.evidences:
            norm_ev = CandidateSkillEvidence(
                skill=normalize_skill(ev.skill),
                context=ev.context,
                duration_months=ev.duration_months,
                role=ev.role,
                academic_year=ev.academic_year,
                months_since_used=ev.months_since_used
            )
            normalized_evidences.append(norm_ev)

        candidate_scores = compute_candidate_skill_scores(normalized_evidences)

        st.subheader("Computed Skill Scores")
        score_df = pd.DataFrame([
            {"Normalized Skill": skill.title(), "Calculated Score": score}
            for skill, score in candidate_scores.items()
        ])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        top_n = st.slider("Top Job Title Matches to Display", min_value=1, max_value=10, value=3)
        top_matches = rank_job_titles_for_candidate(candidate_scores, pool, top_n=top_n)

        st.subheader("Job Matches")
        for match in top_matches:
            with st.expander(f"**{match.job_title.title()}** — {match.match_percent}% Match", expanded=True):
                st.progress(match.match_percent / 100.0)

                gaps = match.gaps()
                if gaps:
                    st.markdown("**Skill Gaps:**")
                    for g in gaps:
                        crit = "🔴 *(Critical)*" if g.is_critical else "🟡"
                        st.write(f"- {crit} **{g.skill.title()}** (Job Importance: {g.importance})")
                else:
                    st.success("No skill gaps for this job role!")

                matches = match.matching_skills()
                if matches:
                    st.markdown("**Matching Skills:**")
                    for m in matches:
                        st.write(
                            f"- 🟢 **{m.skill.title()}** — Your Score: {m.candidate_score} "
                            f"(Job Importance: {m.importance})"
                        )