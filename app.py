import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
import time

# Import our AI Interviewer models
from models import InterviewerAgent, TechnicalSkillsModel, BehavioralSkillsModel

class AIInterviewerApp:
    """Main application class for AI Interviewer"""
    
    def __init__(self):
        self.interviewer = InterviewerAgent()
        self.candidates = self._load_candidates()
        self.current_candidate = None
        self.interview_in_progress = False
    
    def _load_candidates(self):
        """Load candidate data from file"""
        try:
            with open('data/candidates.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return empty list if file not found
            return []
    
    def _save_candidates(self):
        """Save candidate data to file"""
        os.makedirs('data', exist_ok=True)
        with open('data/candidates.json', 'w') as f:
            json.dump(self.candidates, f)
    
    def add_candidate(self, candidate_info):
        """Add a new candidate to the system"""
        candidate_id = f"C{len(self.candidates) + 1}"
        candidate_info["id"] = candidate_id
        candidate_info["interview_date"] = datetime.now().strftime("%Y-%m-%d")
        candidate_info["status"] = "Pending"
        candidate_info["assessment"] = None
        
        self.candidates.append(candidate_info)
        self._save_candidates()
        return candidate_id
    
    def start_interview(self, candidate_id):
        """Start interview for a candidate"""
        # Find candidate
        candidate = next((c for c in self.candidates if c["id"] == candidate_id), None)
        if not candidate:
            return False, "Candidate not found"
        
        self.current_candidate = candidate
        self.interview_in_progress = True
        
        # Determine focus areas based on position
        technical_focus = candidate["position"] in ["Software Engineer", "Data Scientist", "DevOps Engineer"]
        behavioral_focus = True  # Always include behavioral questions
        
        # Start interview through the interviewer agent
        next_question = self.interviewer.start_interview(
            candidate_id, 
            technical_focus=technical_focus,
            behavioral_focus=behavioral_focus
        )
        
        return True, next_question
    
    def process_response(self, response_text):
        """Process candidate's response"""
        if not self.interview_in_progress:
            return False, "No interview in progress"
        
        result = self.interviewer.process_response(response_text)
        
        # If no more questions, interview is complete
        if result["next_question"] is None:
            self.interview_in_progress = False
            self._update_candidate_assessment()
        
        return True, result
    
    def _update_candidate_assessment(self):
        """Update candidate record with assessment results"""
        assessment = self.interviewer.get_candidate_assessment()
        
        # Find candidate in list and update
        for i, candidate in enumerate(self.candidates):
            if candidate["id"] == self.current_candidate["id"]:
                self.candidates[i]["status"] = "Interviewed"
                self.candidates[i]["assessment"] = assessment["assessment"]
                self.candidates[i]["interview_data"] = {
                    "technical_responses": assessment["technical_responses"],
                    "behavioral_responses": assessment["behavioral_responses"]
                }
                break
        
        self._save_candidates()
    
    def get_candidate_rankings(self):
        """Return sorted list of candidates based on assessment scores"""
        # Filter candidates with assessments
        assessed_candidates = [c for c in self.candidates if c.get("assessment")]
        
        if not assessed_candidates:
            return []
        
        # Sort by overall score
        sorted_candidates = sorted(
            assessed_candidates,
            key=lambda x: x["assessment"]["overall_score"],
            reverse=True
        )
        
        return sorted_candidates
    
    def get_candidate_details(self, candidate_id):
        """Get detailed information for a specific candidate"""
        candidate = next((c for c in self.candidates if c["id"] == candidate_id), None)
        return candidate


def run_streamlit_dashboard():
    """Run the Streamlit dashboard for the AI Interviewer"""
    st.set_page_config(page_title="AI Interviewer Dashboard", layout="wide")
    
    # Initialize app
    app = AIInterviewerApp()
    
    # Sidebar navigation
    st.sidebar.title("AI Interviewer")
    page = st.sidebar.selectbox("Select Page", ["Dashboard", "New Interview", "Candidate Details"])
    
    if page == "Dashboard":
        st.title("AI Interviewer Dashboard")
        
        # Get candidate rankings
        rankings = app.get_candidate_rankings()
        
        if not rankings:
            st.info("No candidates have been interviewed yet.")
        else:
            st.subheader("Candidate Rankings")
            
            # Create DataFrame for display
            df = pd.DataFrame([
                {
                    "ID": c["id"],
                    "Name": c["name"],
                    "Position": c["position"],
                    "Technical Score": round(c["assessment"]["overall_technical_score"] * 100),
                    "Behavioral Score": round(c["assessment"]["overall_behavioral_score"] * 100),
                    "Overall Score": round(c["assessment"]["overall_score"] * 100),
                    "Recommendation": c["assessment"]["recommendation"]
                }
                for c in rankings
            ])
            
            # Display as table
            st.dataframe(df)
            
            # Visualization
            st.subheader("Score Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(df))
            width = 0.35
            
            technical = ax.bar(x - width/2, df["Technical Score"], width, label="Technical")
            behavioral = ax.bar(x + width/2, df["Behavioral Score"], width, label="Behavioral")
            
            ax.set_ylabel("Score")
            ax.set_title("Candidate Scores")
            ax.set_xticks(x)
            ax.set_xticklabels(df["Name"])
            ax.legend()
            
            st.pyplot(fig)
    
    elif page == "New Interview":
        st.title("New Interview")
        
        # Form for adding new candidate
        with st.form("new_candidate_form"):
            st.subheader("Candidate Information")
            
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            position = st.selectbox(
                "Position",
                ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "Other"]
            )
            experience = st.slider("Years of Experience", 0, 20, 3)
            
            submitted = st.form_submit_button("Add Candidate")
            
            if submitted:
                if name and email:
                    candidate_info = {
                        "name": name,
                        "email": email,
                        "position": position,
                        "experience": experience
                    }
                    
                    candidate_id = app.add_candidate(candidate_info)
                    st.success(f"Candidate added successfully! ID: {candidate_id}")
                    
                    # Option to start interview immediately
                    start_now = st.button("Start Interview Now")
                    
                    if start_now:
                        st.session_state.interview_candidate_id = candidate_id
                        st.session_state.interview_active = True
                        st.session_state.current_question = None
                        st.experimental_rerun()
                else:
                    st.error("Please fill in all required fields")
        
        # Interview section
        st.subheader("Start Interview")
        
        # Get candidates with pending status
        pending_candidates = [c for c in app.candidates if c["status"] == "Pending"]
        
        if not pending_candidates:
            st.info("No pending candidates available for interview.")
        else:
            # Only show if not already in interview
            if not st.session_state.get("interview_active", False):
                candidate_options = {f"{c['name']} ({c['id']})": c["id"] for c in pending_candidates}
                selected_candidate = st.selectbox("Select Candidate", list(candidate_options.keys()))
                selected_id = candidate_options[selected_candidate]
                
                if st.button("Start Interview"):
                    success, result = app.start_interview(selected_id)
                    
                    if success:
                        st.session_state.interview_candidate_id = selected_id
                        st.session_state.interview_active = True
                        st.session_state.current_question = result
                        st.experimental_rerun()
                    else:
                        st.error(result)
            
            # Interview in progress
            if st.session_state.get("interview_active", False):
                st.subheader("Interview in Progress")
                
                # Get current candidate
                candidate = app.get_candidate_details(st.session_state.interview_candidate_id)
                st.write(f"Interviewing: {candidate['name']}")
                
                # Display current question
                current_q = st.session_state.get("current_question")
                
                if current_q:
                    q_type = current_q["type"].capitalize()
                    st.write(f"**{q_type} Question:**")
                    st.write(current_q["text"])
                    
                    # Get response from user (simulating candidate)
                    response = st.text_area("Answer (simulating candidate response)")
                    
                    if st.button("Submit Response"):
                        if response:
                            success, result = app.process_response(response)
                            
                            if success:
                                if result["next_question"]:
                                    st.session_state.current_question = result["next_question"]
                                    st.success(f"Response processed. {result['remaining_questions']} questions remaining.")
                                    st.experimental_rerun()
                                else:
                                    st.success("Interview completed!")
                                    st.session_state.interview_active = False
                                    st.session_state.current_question = None
                                    st.experimental_rerun()
                            else:
                                st.error(result)
                        else:
                            st.warning("Please enter a response")
                else:
                    st.info("Interview complete or no questions available")
                    st.session_state.interview_active = False
    
    elif page == "Candidate Details":
        st.title("Candidate Details")
        
        # Get all candidates
        all_candidates = app.candidates
        
        if not all_candidates:
            st.info("No candidates in the system.")
        else:
            # Create selection options
            candidate_options = {f"{c['name']} ({c['id']})": c["id"] for c in all_candidates}
            selected_candidate = st.selectbox("Select Candidate", list(candidate_options.keys()))
            selected_id = candidate_options[selected_candidate]
            
            # Get detailed info
            candidate = app.get_candidate_details(selected_id)
            
            if not candidate:
                st.error("Candidate not found")
            else:
                # Display candidate info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Basic Information")
                    st.write(f"**Name:** {candidate['name']}")
                    st.write(f"**Email:** {candidate['email']}")
                    st.write(f"**Position:** {candidate['position']}")
                    st.write(f"**Experience:** {candidate['experience']} years")
                    st.write(f"**Status:** {candidate['status']}")
                    st.write(f"**Interview Date:** {candidate.get('interview_date', 'N/A')}")
                
                # Assessment info (if available)
                if candidate.get("assessment"):
                    with col2:
                        st.subheader("Assessment Summary")
                        st.write(f"**Overall Score:** {round(candidate['assessment']['overall_score'] * 100)}%")
                        st.write(f"**Technical Score:** {round(candidate['assessment']['overall_technical_score'] * 100)}%")
                        st.write(f"**Behavioral Score:** {round(candidate['assessment']['overall_behavioral_score'] * 100)}%")
                        st.write(f"**Recommendation:** {candidate['assessment']['recommendation']}")
                    
                    # Category breakdown
                    st.subheader("Skill Category Breakdown")
                    
                    if candidate['assessment'].get('category_scores'):
                        categories = list(candidate['assessment']['category_scores'].keys())
                        scores = list(candidate['assessment']['category_scores'].values())
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        y_pos = np.arange(len(categories))
                        
                        bars = ax.barh(y_pos, [s * 100 for s in scores])
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(categories)
                        ax.set_xlabel('Score (%)')
                        ax.set_title('Technical Skill Categories')
                        
                        # Add score labels
                        for i, bar in enumerate(bars):
                            ax.text(
                                bar.get_width() + 1, 
                                bar.get_y() + bar.get_height()/2, 
                                f"{scores[i]*100:.1f}%", 
                                va='center'
                            )
                        
                        st.pyplot(fig)
                    else:
                        st.info("No category scores available")
                    
                    # Interview responses
                    if candidate.get("interview_data"):
                        st.subheader("Interview Responses")
                        
                        tabs = st.tabs(["Technical", "Behavioral"])
                        
                        with tabs[0]:
                            if candidate["interview_data"].get("technical_responses"):
                                for i, resp in enumerate(candidate["interview_data"]["technical_responses"]):
                                    with st.expander(f"Question {i+1}: {resp['question'][:50]}..."):
                                        st.write("**Question:**")
                                        st.write(resp["question"])
                                        st.write("**Response:**")
                                        st.write(resp["response"])
                                        st.write(f"**Score:** {round(resp['score'] * 100)}%")
                                        st.write(f"**Feedback:** {resp['feedback']}")
                            else:
                                st.info("No technical responses recorded")
                        
                        with tabs[1]:
                            if candidate["interview_data"].get("behavioral_responses"):
                                for i, resp in enumerate(candidate["interview_data"]["behavioral_responses"]):
                                    with st.expander(f"Question {i+1}: {resp['question'][:50]}..."):
                                        st.write("**Question:**")
                                        st.write(resp["question"])
                                        st.write("**Response:**")
                                        st.write(resp["response"])
                                        st.write(f"**Score:** {round(resp['score'] * 100)}%")
                                        st.write(f"**Feedback:** {resp['feedback']}")
                                        if resp.get("sentiment"):
                                            sentiment = resp["sentiment"]
                                            st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
                            else:
                                st.info("No behavioral responses recorded")
                else:
                    st.info("Candidate has not been interviewed yet")


if __name__ == "__main__":
    # Initialize session state
    if 'interview_active' not in st.session_state:
        st.session_state.interview_active = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'interview_candidate_id' not in st.session_state:
        st.session_state.interview_candidate_id = None
    
    run_streamlit_dashboard()
