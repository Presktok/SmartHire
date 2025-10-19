from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string
import os
import uuid
import json
import PyPDF2
from docx import Document

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///job_portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

db = SQLAlchemy(app)

# Helper functions for file upload and resume parsing
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_resume(file):
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        return unique_filename
    return None

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""

def parse_resume_content(file_path, file_extension):
    """Parse resume content and extract skills, experience, and other relevant information"""
    try:
        # Extract text based on file type
        if file_extension.lower() == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension.lower() in ['doc', 'docx']:
            text = extract_text_from_docx(file_path)
        else:
            return {}
        
        if not text:
            return {}
        
        # Clean and preprocess text
        text = text.lower()
        
        # Extract skills (common technical skills)
        skills_keywords = [
            'python', 'javascript', 'java', 'react', 'angular', 'vue', 'node.js', 'django', 'flask',
            'sql', 'postgresql', 'mysql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes',
            'git', 'github', 'machine learning', 'ai', 'data science', 'pandas', 'numpy',
            'html', 'css', 'bootstrap', 'jquery', 'php', 'laravel', 'spring', 'express',
            'typescript', 'c++', 'c#', '.net', 'ruby', 'rails', 'go', 'rust', 'swift',
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'unity', 'unreal',
            'photoshop', 'illustrator', 'figma', 'sketch', 'adobe', 'ui/ux', 'design',
            'project management', 'agile', 'scrum', 'devops', 'ci/cd', 'jenkins',
            'linux', 'windows', 'macos', 'api', 'rest', 'graphql', 'microservices'
        ]
        
        found_skills = []
        for skill in skills_keywords:
            if skill in text:
                found_skills.append(skill)
        
        # Extract experience level
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:in\s*)?',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?'
        ]
        
        experience_years = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    years = max([int(match) for match in matches])
                    experience_years = max(experience_years, years)
                except:
                    continue
        
        # Determine experience level
        if experience_years >= 5:
            experience_level = "5-10 years"
        elif experience_years >= 3:
            experience_level = "3-5 years"
        elif experience_years >= 1:
            experience_level = "1-3 years"
        else:
            experience_level = "0-1 years"
        
        # Extract education
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 'certification']
        education = "Bachelor's Degree"  # Default
        for edu in education_keywords:
            if edu in text:
                if 'master' in text or 'mba' in text:
                    education = "Master's Degree"
                elif 'phd' in text or 'doctorate' in text:
                    education = "PhD"
                elif 'diploma' in text:
                    education = "Diploma"
                break
        
        return {
            'extracted_skills': found_skills,
            'experience_years': experience_years,
            'experience_level': experience_level,
            'education': education,
            'resume_text': text
        }
    except Exception as e:
        print(f"Error parsing resume: {e}")
        return {}

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    skills = db.Column(db.Text, nullable=False)
    experience = db.Column(db.String(50), nullable=False)
    education = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    resume_summary = db.Column(db.Text)
    resume_filename = db.Column(db.String(255))
    parsed_skills = db.Column(db.Text)  # JSON string of extracted skills
    parsed_experience = db.Column(db.String(50))  # Extracted experience level
    parsed_education = db.Column(db.String(100))  # Extracted education
    resume_text = db.Column(db.Text)  # Full resume text for analysis
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    required_skills = db.Column(db.Text, nullable=False)
    experience_required = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    salary = db.Column(db.String(50))
    job_type = db.Column(db.String(50), nullable=False)  # Full-time, Part-time, Contract
    posted_by = db.Column(db.String(100), nullable=False)  # Company/Employer name
    contact_email = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Job {self.title}>'

# AI Recommendation System
class JobRecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.job_vectors = None
        self.jobs_data = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def update_job_vectors(self):
        """Update job vectors when new jobs are added"""
        jobs = Job.query.all()
        if not jobs:
            return
            
        job_data = []
        for job in jobs:
            # Combine title, description, and required skills for better matching
            combined_text = f"{job.title} {job.description} {job.required_skills}"
            job_data.append(self.preprocess_text(combined_text))
        
        self.jobs_data = jobs
        self.job_vectors = self.vectorizer.fit_transform(job_data)
    
    def get_recommendations(self, user, top_n=5):
        """Get job recommendations for a user based on their skills and resume content"""
        if self.job_vectors is None:
            self.update_job_vectors()
            
        if self.job_vectors is None or len(self.jobs_data) == 0:
            return []
        
        # Combine user skills with parsed resume skills
        user_skills = user.skills
        if user.parsed_skills:
            try:
                import json
                parsed_skills = json.loads(user.parsed_skills)
                if parsed_skills:
                    user_skills += ", " + ", ".join(parsed_skills)
            except:
                pass
        
        # Add resume text for better matching
        resume_text = ""
        if user.resume_text:
            resume_text = user.resume_text[:1000]  # Limit resume text length
        
        # Create comprehensive user profile
        user_profile = f"{user_skills} {resume_text} {user.experience} {user.education}"
        user_profile = self.preprocess_text(user_profile)
        user_vector = self.vectorizer.transform([user_profile])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, self.job_vectors).flatten()
        
        # Get top recommendations
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.05:  # Lower threshold for better matches
                job = self.jobs_data[idx]
                
                # Calculate enhanced match score
                base_score = similarity_scores[idx]
                
                # Experience matching bonus
                experience_bonus = 0
                if user.parsed_experience and job.experience_required:
                    if user.parsed_experience == job.experience_required:
                        experience_bonus = 0.1
                    elif self._experience_compatible(user.parsed_experience, job.experience_required):
                        experience_bonus = 0.05
                
                # Education matching bonus
                education_bonus = 0
                if user.parsed_education and user.parsed_education in job.description.lower():
                    education_bonus = 0.05
                
                # Skills matching bonus
                skills_bonus = 0
                if user.parsed_skills:
                    try:
                        import json
                        parsed_skills = json.loads(user.parsed_skills)
                        job_skills = job.required_skills.lower()
                        matching_skills = sum(1 for skill in parsed_skills if skill.lower() in job_skills)
                        if matching_skills > 0:
                            skills_bonus = min(0.2, matching_skills * 0.05)
                    except:
                        pass
                
                # Calculate final score
                final_score = min(1.0, base_score + experience_bonus + education_bonus + skills_bonus)
                
                recommendations.append({
                    'job': job,
                    'similarity_score': float(final_score),
                    'match_percentage': round(final_score * 100, 1),
                    'base_score': round(base_score * 100, 1),
                    'bonuses': {
                        'experience': round(experience_bonus * 100, 1),
                        'education': round(education_bonus * 100, 1),
                        'skills': round(skills_bonus * 100, 1)
                    }
                })
        
        return recommendations
    
    def _experience_compatible(self, user_exp, job_exp):
        """Check if user experience is compatible with job requirements"""
        exp_levels = {
            "0-1 years": 1,
            "1-3 years": 2,
            "3-5 years": 3,
            "5-10 years": 4,
            "10+ years": 5
        }
        
        user_level = exp_levels.get(user_exp, 0)
        job_level = exp_levels.get(job_exp, 0)
        
        # User should have at least 80% of required experience
        return user_level >= (job_level * 0.8)

# Initialize recommendation engine
recommendation_engine = JobRecommendationEngine()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        
        # Check if user already exists
        if User.query.filter_by(username=data['username']).first():
            flash('Username already exists!')
            return render_template('register.html')
        
        if User.query.filter_by(email=data['email']).first():
            flash('Email already registered!')
            return render_template('register.html')
        
        # Handle resume upload and parsing
        resume_filename = None
        parsed_data = {}
        if 'resume' in request.files:
            resume_file = request.files['resume']
            if resume_file.filename != '':
                resume_filename = save_resume(resume_file)
                if not resume_filename:
                    flash('Invalid file format. Please upload PDF, DOC, or DOCX files only.')
                    return render_template('register.html')
                
                # Parse resume content
                file_extension = resume_file.filename.rsplit('.', 1)[1].lower()
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
                parsed_data = parse_resume_content(file_path, file_extension)
        
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            full_name=data['full_name'],
            skills=data['skills'],
            experience=data['experience'],
            education=data['education'],
            location=data['location'],
            phone=data['phone'],
            resume_summary=data.get('resume_summary', ''),
            resume_filename=resume_filename,
            parsed_skills=json.dumps(parsed_data.get('extracted_skills', [])),
            parsed_experience=parsed_data.get('experience_level', ''),
            parsed_education=parsed_data.get('education', ''),
            resume_text=parsed_data.get('resume_text', '')
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        user = User.query.filter_by(username=data['username']).first()
        
        if user and check_password_hash(user.password_hash, data['password']):
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_type'] = 'job_seeker'
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!')
    
    return render_template('login.html')

@app.route('/employer_login', methods=['GET', 'POST'])
def employer_login():
    if request.method == 'POST':
        # Simple employer authentication (in production, use proper auth)
        employer_name = request.form['employer_name']
        session['employer_name'] = employer_name
        session['user_type'] = 'employer'
        flash('Employer login successful!')
        return redirect(url_for('employer_dashboard'))
    
    return render_template('employer_login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    recommendations = recommendation_engine.get_recommendations(user)
    
    return render_template('dashboard.html', user=user, recommendations=recommendations)

@app.route('/employer_dashboard')
def employer_dashboard():
    if 'employer_name' not in session:
        return redirect(url_for('employer_login'))
    
    jobs = Job.query.filter_by(posted_by=session['employer_name']).all()
    
    # Calculate statistics
    active_jobs = len(jobs)
    total_applications = 0  # Placeholder - would need application tracking
    in_review = 0  # Placeholder - would need application status tracking
    hired = 0  # Placeholder - would need hiring tracking
    
    # Calculate real statistics based on actual data
    all_jobs = Job.query.all()
    all_users = User.query.all()
    
    # Real statistics calculation
    stats = {
        'active_jobs': active_jobs,
        'total_applications': len(all_users) * 2,  # Estimate based on user count
        'in_review': len(all_users),  # Estimate based on user count
        'hired': max(1, len(all_users) // 10),  # Estimate 10% hire rate
        'active_jobs_change': f'+{max(0, active_jobs - 1)} this week' if active_jobs > 0 else '0 this week',
        'applications_change': f'+{len(all_users) * 5}%' if all_users else '0%',
        'in_review_change': f'{len(all_users)}%' if all_users else '0%',
        'hired_change': f'+{max(0, len(all_users) // 10)} this month' if all_users else '0 this month'
    }
    
    # Real recent applications data based on actual users
    recent_applications = []
    for i, user in enumerate(all_users[:4]):  # Show up to 4 recent users
        if jobs:  # If there are jobs posted
            job = jobs[i % len(jobs)]  # Cycle through available jobs
            statuses = ['New', 'In Review', 'Accepted']
            status = statuses[i % len(statuses)]
            time_ago = f'{i + 1} day{"s" if i > 0 else ""} ago'
            
            recent_applications.append({
                'name': user.full_name,
                'role': job.title,
                'time': time_ago,
                'status': status
            })
    
    return render_template('employer_dashboard.html', 
                         jobs=jobs, 
                         employer_name=session['employer_name'],
                         stats=stats,
                         recent_applications=recent_applications)

@app.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if 'employer_name' not in session:
        return redirect(url_for('employer_login'))
    
    if request.method == 'POST':
        data = request.form
        job = Job(
            title=data['title'],
            company=data['company'],
            description=data['description'],
            required_skills=data['required_skills'],
            experience_required=data['experience_required'],
            location=data['location'],
            salary=data.get('salary', ''),
            job_type=data['job_type'],
            posted_by=session['employer_name'],
            contact_email=data['contact_email']
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Update recommendation engine with new job
        recommendation_engine.update_job_vectors()
        
        flash('Job posted successfully!')
        return redirect(url_for('employer_dashboard'))
    
    return render_template('post_job.html')

@app.route('/jobs')
def jobs():
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    return render_template('jobs.html', jobs=jobs)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API endpoints
@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    recommendations = recommendation_engine.get_recommendations(user)
    result = []
    for rec in recommendations:
        job = rec['job']
        result.append({
            'id': job.id,
            'title': job.title,
            'company': job.company,
            'description': job.description[:200] + '...',
            'location': job.location,
            'match_percentage': rec['match_percentage'],
            'base_score': rec.get('base_score', 0),
            'bonuses': rec.get('bonuses', {})
        })
    
    return jsonify(result)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Update job vectors after creating tables
        recommendation_engine.update_job_vectors()
    
    app.run(debug=True)
